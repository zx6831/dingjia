import sqlite3
import threading
import time
import uuid
import queue
import traceback
import json
import asyncio
import inspect
from typing import Callable, Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from utils.logger import get_logger
from enum import Enum
import os

logger = get_logger(log_dir= "../logs/taskQueue")

class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    FINISHED = "finished"
    ERROR = "error"


class Task:
    def __init__(self, function: Callable, kwargs: Dict[str, Any] = None, task_id: str = None):
        self.task_id = task_id or str(uuid.uuid4())
        self.function = function
        self.kwargs = kwargs or {}
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now()
        self.started_at = None
        self.finished_at = None
        self.result = None
        self.error_message = None
        self.traceback = None
        self.is_async = inspect.iscoroutinefunction(function)

    def to_dict(self) -> Dict[str, Any]:
        """将任务转换为字典格式"""
        return {
            "taskId": self.task_id,
            "status": self.status.value,
            "function": self.function.__name__ if hasattr(self.function, '__name__') else str(self.function),
            "kwargs": self.kwargs,
            "is_async": self.is_async,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "result": self.result,
            "error_message": self.error_message,
            "traceback": self.traceback
        }

    async def execute_async(self, on_status_change: Optional[Callable[['Task'], None]] = None):
        """异步执行任务（新增状态回调 on_status_change）"""
        try:
            self.status = TaskStatus.PROCESSING
            self.started_at = datetime.now()
            if on_status_change:
                on_status_change(self)

            if self.is_async:
                self.result = await self.function(**self.kwargs)
            else:
                loop = asyncio.get_event_loop()
                self.result = await loop.run_in_executor(None, lambda: self.function(**self.kwargs))

            self.status = TaskStatus.FINISHED
        except Exception as e:
            self.status = TaskStatus.ERROR
            self.error_message = str(e)
            print(self.error_message)
            self.traceback = traceback.format_exc()
        finally:
            self.finished_at = datetime.now()
            if on_status_change:
                on_status_change(self)

    def execute_sync(self, on_status_change: Optional[Callable[['Task'], None]] = None):
        """同步执行任务（兼容旧版本，新增状态回调）"""
        try:
            self.status = TaskStatus.PROCESSING
            self.started_at = datetime.now()
            if on_status_change:
                on_status_change(self)

            if self.is_async:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    self.result = loop.run_until_complete(self.function(**self.kwargs))
                finally:
                    loop.close()
            else:
                self.result = self.function(**self.kwargs)

            self.status = TaskStatus.FINISHED
        except Exception as e:
            self.status = TaskStatus.ERROR
            self.error_message = str(e)
            self.traceback = traceback.format_exc()
        finally:
            self.finished_at = datetime.now()
            if on_status_change:
                on_status_change(self)


class TaskDatabase:
    """微型数据库类，用于存储任务结果"""

    def __init__(self, db_path: str = "task_queue.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_database()

    def _init_database(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                         CREATE TABLE IF NOT EXISTS task_results
                         (
                             task_id
                             TEXT
                             PRIMARY
                             KEY,
                             function_name
                             TEXT,
                             kwargs
                             TEXT,
                             status
                             TEXT,
                             is_async
                             BOOLEAN,
                             created_at
                             TEXT,
                             started_at
                             TEXT,
                             finished_at
                             TEXT,
                             result
                             TEXT,
                             error_message
                             TEXT,
                             traceback
                             TEXT,
                             created_timestamp
                             REAL
                         )
                         ''')

            conn.execute('''
                         CREATE INDEX IF NOT EXISTS idx_status ON task_results(status)
                         ''')
            conn.execute('''
                         CREATE INDEX IF NOT EXISTS idx_created_timestamp ON task_results(created_timestamp)
                         ''')
            conn.commit()

    def save_task_result(self, task: Task):
        """保存任务结果到数据库"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO task_results 
                        (task_id, function_name, kwargs, status, is_async, created_at, started_at, 
                         finished_at, result, error_message, traceback, created_timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        task.task_id,
                        task.function.__name__ if hasattr(task.function, '__name__') else str(task.function),
                        json.dumps(task.kwargs, ensure_ascii=False),
                        task.status.value,
                        task.is_async,
                        task.created_at.isoformat(),
                        task.started_at.isoformat() if task.started_at else None,
                        task.finished_at.isoformat() if task.finished_at else None,
                        json.dumps(task.result, ensure_ascii=False) if task.result is not None else None,
                        task.error_message,
                        task.traceback,
                        task.created_at.timestamp()
                    ))
                    conn.commit()
            except Exception as e:
                logger.info(f"保存任务结果到数据库失败: {e}")

    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """根据任务ID获取任务结果"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        'SELECT * FROM task_results WHERE task_id = ?',
                        (task_id,)
                    )
                    row = cursor.fetchone()
                    if row:
                        return self._row_to_dict(row)
                    return None
            except Exception as e:
                logger.info(f"从数据库获取任务结果失败: {e}")
                return None

    def query_tasks(self, status: Optional[str] = None,
                    limit: Optional[int] = None,
                    offset: int = 0,
                    order_by: str = "created_timestamp DESC") -> List[Dict[str, Any]]:
        """查询任务结果"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row

                    query = "SELECT * FROM task_results"
                    params = []

                    if status:
                        query += " WHERE status = ?"
                        params.append(status)

                    query += f" ORDER BY {order_by}"

                    if limit:
                        query += " LIMIT ? OFFSET ?"
                        params.extend([limit, offset])

                    cursor = conn.execute(query, params)
                    rows = cursor.fetchall()
                    return [self._row_to_dict(row) for row in rows]
            except Exception as e:
                logger.info(f"查询任务结果失败: {e}")
                return []

    def get_statistics(self) -> Dict[str, Any]:
        """获取任务统计信息"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # 总任务数
                    total = conn.execute("SELECT COUNT(*) FROM task_results").fetchone()[0]

                    # 各状态任务数
                    status_counts = {}
                    for status in ['pending', 'processing', 'finished', 'error']:
                        count = conn.execute(
                            "SELECT COUNT(*) FROM task_results WHERE status = ?",
                            (status,)
                        ).fetchone()[0]
                        status_counts[status] = count

                    # 统计异步/同步任务
                    async_count = conn.execute(
                        "SELECT COUNT(*) FROM task_results WHERE is_async = 1"
                    ).fetchone()[0]
                    sync_count = total - async_count

                    # 最近任务 - 修复这里的问题
                    conn.row_factory = sqlite3.Row
                    recent_tasks_rows = conn.execute('''
                                                     SELECT task_id, status, is_async, created_at, finished_at
                                                     FROM task_results
                                                     ORDER BY created_timestamp DESC LIMIT 5
                                                     ''').fetchall()

                    # 使用安全的方法转换行
                    recent_tasks = []
                    for row in recent_tasks_rows:
                        try:
                            task_dict = {
                                'task_id': row['task_id'],
                                'status': row['status'],
                                'is_async': bool(row['is_async']),
                                'created_at': row['created_at'],
                                'finished_at': row['finished_at']
                            }
                            recent_tasks.append(task_dict)
                        except Exception as e:
                            logger.info(f"转换任务行时出错: {e}")
                            continue

                    return {
                        "total_tasks": total,
                        "status_counts": status_counts,
                        "async_tasks": async_count,
                        "sync_tasks": sync_count,
                        "recent_tasks": recent_tasks,
                        "database_size": os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                    }
            except Exception as e:
                logger.info(f"获取统计信息失败: {e}")
                # 返回默认值以避免崩溃
                return {
                    "total_tasks": 0,
                    "status_counts": {"pending": 0, "processing": 0, "finished": 0, "error": 0},
                    "async_tasks": 0,
                    "sync_tasks": 0,
                    "recent_tasks": [],
                    "database_size": 0,
                    "error": str(e)
                }

    def delete_task(self, task_id: str) -> bool:
        """删除任务记录"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM task_results WHERE task_id = ?",
                        (task_id,)
                    )
                    conn.commit()
                    return cursor.rowcount > 0
            except Exception as e:
                logger.info(f"删除任务记录失败: {e}")
                return False

    def clear_old_tasks(self, days: int = 30) -> int:
        """清理指定天数前的旧任务"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cutoff_time = (datetime.now() - timedelta(days=days)).timestamp()
                    cursor = conn.execute(
                        "DELETE FROM task_results WHERE created_timestamp < ?",
                        (cutoff_time,)
                    )
                    conn.commit()
                    deleted_count = cursor.rowcount
                    logger.info(f"清理了 {deleted_count} 个旧任务记录")
                    return deleted_count
            except Exception as e:
                logger.info(f"清理旧任务失败: {e}")
                return 0

    def clear_all_tasks(self) -> int:
        """清空所有任务记录"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("DELETE FROM task_results")
                    conn.commit()
                    deleted_count = cursor.rowcount
                    logger.info(f"清空了 {deleted_count} 个任务记录")
                    return deleted_count
            except Exception as e:
                logger.info(f"清空任务记录失败: {e}")
                return 0

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """安全地将数据库行转换为字典"""
        try:
            # 使用键值对的方式安全转换
            result = {}
            for key in row.keys():
                result[key] = row[key]

            # 解析JSON字段
            if result.get('kwargs'):
                try:
                    result['kwargs'] = json.loads(result['kwargs'])
                except json.JSONDecodeError:
                    result['kwargs'] = result['kwargs']

            if result.get('result'):
                try:
                    result['result'] = json.loads(result['result'])
                except json.JSONDecodeError:
                    result['result'] = result['result']

            return result
        except Exception as e:
            logger.info(f"转换数据库行时出错: {e}")
            # 返回基本的字典结构
            return {
                'task_id': str(row[0]) if len(row) > 0 else 'unknown',
                'error': 'conversion_failed'
            }


class AsyncTaskQueue:
    """支持异步任务的任务队列"""

    def __init__(self, db_path: str = "task_queue.db", check_interval: float = 1.0):
        """
        初始化任务队列

        Args:
            db_path: 数据库文件路径
            check_interval: 检查队列的间隔时间（秒）
        """
        self.tasks: Dict[str, Task] = {}
        self.task_queue = queue.Queue()
        self.current_task: Optional[Task] = None
        self.check_interval = check_interval
        self._running = False
        self._worker_thread = None
        self._loop = None
        self._lock = threading.RLock()
        self.database = TaskDatabase(db_path)

    def _on_task_status_change(self, task: Task) -> None:
        """任务状态变化时的统一 upsert"""
        try:
            self.database.save_task_result(task)
        except Exception as e:
            logger.info(f"状态变更写库失败: {e}")

    def _task_to_db_like_dict(self, task: Task) -> Dict[str, Any]:
        """把内存 Task 映射成与 DB 返回尽可能一致的字典，便于合并"""
        return {
            "task_id": task.task_id,
            "function_name": task.function.__name__ if hasattr(task.function, '__name__') else str(task.function),
            "kwargs": task.kwargs,  # 直接给对象（DB 里 query_tasks 已解析 JSON）
            "status": task.status.value,
            "is_async": bool(task.is_async),
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "finished_at": task.finished_at.isoformat() if task.finished_at else None,
            "result": task.result,
            "error_message": task.error_message,
            "traceback": task.traceback,
            "created_timestamp": task.created_at.timestamp(),
        }

    def start(self):
        """启动任务队列处理器"""
        with self._lock:
            if not self._running:
                self._running = True
                self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
                self._worker_thread.start()
                logger.info("异步任务队列已启动")

    def stop(self):
        """停止任务队列处理器"""
        with self._lock:
            if self._running:
                self._running = False
                if self._worker_thread:
                    self._worker_thread.join(timeout=5)
                logger.info("异步任务队列已停止")

    def _worker_loop(self):
        """工作线程主循环"""
        # 创建新的事件循环
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            while self._running:
                try:
                    # 尝试从队列获取任务
                    try:
                        task = self.task_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    # 执行任务
                    with self._lock:
                        self.current_task = task

                    logger.info(f"开始执行任务: {task.task_id} ({'异步' if task.is_async else '同步'})")

                    # ✅ 把状态回调传入，processing / finished / error 都会即时 upsert
                    if task.is_async or True:
                        self._loop.run_until_complete(
                            task.execute_async(on_status_change=self._on_task_status_change))
                    # （如强制区分同步执行，也可调用 task.execute_sync(on_status_change=self._on_task_status_change)）

                    logger.info(f"任务执行完成: {task.task_id}, 状态: {task.status.value}")
                    # print(self.error_message)

                    # （已在回调中 upsert，这里可不再重复；保守起见保留一次最终 upsert）
                    self.database.save_task_result(task)

                    with self._lock:
                        self.current_task = None

                except Exception as e:
                    logger.info(f"工作线程异常: {e}")
                    traceback.print_exc()
        finally:
            self._loop.close()

    def add_task(self, function: Callable, kwargs: Dict[str, Any] = None, task_id: str = None) -> str:
        """
        添加任务到队列（支持同步和异步函数）

        Args:
            function: 要执行的函数（可以是同步或异步）
            kwargs: 函数参数
            task_id: 可选的任务ID

        Returns:
            任务ID
        """
        task = Task(function, kwargs, task_id)

        with self._lock:
            if task.task_id in self.tasks:
                raise ValueError(f"任务ID {task.task_id} 已存在")

            self.tasks[task.task_id] = task
            self.task_queue.put(task)

            # ✅ 入队即落库一份 pending 快照
            try:
                self.database.save_task_result(task)
            except Exception as e:
                logger.info(f"新增任务写库失败: {e}")

        logger.info(f"任务已添加到队列: {task.task_id} ({'异步' if task.is_async else '同步'})")
        return task.task_id

    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        with self._lock:
            pending_count = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)
            processing_count = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PROCESSING)
            finished_count = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FINISHED)
            error_count = sum(1 for t in self.tasks.values() if t.status == TaskStatus.ERROR)
            async_count = sum(1 for t in self.tasks.values() if t.is_async)

            return {
                "total_tasks": len(self.tasks),
                "pending_count": pending_count,
                "processing_count": processing_count,
                "finished_count": finished_count,
                "error_count": error_count,
                "async_tasks": async_count,
                "sync_tasks": len(self.tasks) - async_count,
                "current_task": self.current_task.task_id if self.current_task else None,
                "is_running": self._running
            }

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        根据任务ID获取任务状态。
        优先返回内存中的实时状态；若内存不存在，则从数据库拉取最新快照。
        返回值与 query_tasks() 对齐，采用 DB 风格的字段命名。
        """
        # 1) 先看内存里的 Task
        with self._lock:
            task = self.tasks.get(task_id)
        if task:
            # 返回与数据库记录尽可能一致的字典（蛇形命名）
            return self._task_to_db_like_dict(task)

        # 2) 不在内存，则回落到数据库
        db_row = self.database.get_task_result(task_id)
        if db_row:
            # database.get_task_result 已做过 JSON 解析并返回 DB 风格字段
            return db_row

        # 3) 内存和数据库都没有
        return None

    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """从数据库获取任务结果"""
        return self.database.get_task_result(task_id)

    def query_tasks(
        self,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        order_by: str = "created_timestamp DESC",
        include_memory: bool = True
    ) -> List[Dict[str, Any]]:
        """
        查询任务（默认返回实时视图 = DB ∪ 内存），以“内存状态”为准去重。
        传 include_memory=False 则仅查询数据库（与旧行为一致）。
        """
        if not include_memory:
            # 旧行为：只查 DB
            return self.database.query_tasks(status=status, limit=limit, offset=offset, order_by=order_by)

        # 1) 拿到 DB 全量（为了正确合并，这里不分页）
        db_rows = self.database.query_tasks(status=status, limit=None, offset=0, order_by=order_by)
        merged: Dict[str, Dict[str, Any]] = {r["task_id"]: r for r in db_rows}

        # 2) 合并内存任务（以内存为准覆盖）
        with self._lock:
            for t in self.tasks.values():
                if status and t.status.value != status:
                    continue
                merged[t.task_id] = self._task_to_db_like_dict(t)

        # 3) 排序（只支持按 created_timestamp / created_at）
        reverse = "DESC" in order_by.upper()
        key_name = "created_timestamp" if "created_timestamp" in order_by else "created_at"

        def _key(rec):
            if key_name == "created_timestamp":
                return rec.get("created_timestamp") or 0.0
            # created_at 兜底
            try:
                return datetime.fromisoformat(rec.get("created_at") or "1970-01-01T00:00:00").timestamp()
            except Exception:
                return 0.0

        results = sorted(merged.values(), key=_key, reverse=reverse)

        # 4) 分页
        if offset:
            results = results[offset:]
        if limit is not None:
            results = results[:limit]
        return results

    def get_statistics(self, live: bool = True) -> Dict[str, Any]:
        """
        获取任务统计信息。
        默认 live=True：基于“实时”合并视图统计（包含等待/处理中）。
        传 live=False：仅返回数据库统计（保持原实现）。
        """
        if not live:
            return self.database.get_statistics()

        # 用合并视图重算统计
        all_tasks = self.query_tasks(status=None, limit=None, offset=0, include_memory=True)
        total = len(all_tasks)

        def _bool(x):
            return bool(x) if not isinstance(x, bool) else x

        status_counts = {"pending": 0, "processing": 0, "finished": 0, "error": 0}
        async_count = 0
        for r in all_tasks:
            st = r.get("status")
            if st in status_counts:
                status_counts[st] += 1
            if _bool(r.get("is_async")):
                async_count += 1

        sync_count = total - async_count

        # 最近任务（按创建时间）
        recent_tasks = []
        for r in sorted(all_tasks, key=lambda x: x.get("created_timestamp", 0.0), reverse=True)[:5]:
            recent_tasks.append({
                "task_id": r.get("task_id"),
                "status": r.get("status"),
                "is_async": _bool(r.get("is_async")),
                "created_at": r.get("created_at"),
                "finished_at": r.get("finished_at"),
            })

        return {
            "total_tasks": total,
            "status_counts": status_counts,
            "async_tasks": async_count,
            "sync_tasks": sync_count,
            "recent_tasks": recent_tasks,
            "database_size": os.path.getsize(self.database.db_path) if os.path.exists(self.database.db_path) else 0,
            "is_running": self._running,
            "current_task": self.current_task.task_id if self.current_task else None,
        }

    def remove_task(self, task_id: str) -> bool:
        """删除任务（不包括进行中的任务）"""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return False

            if task.status == TaskStatus.PROCESSING:
                print(f"无法删除正在执行的任务: {task_id}")
                return False

            # 从队列中移除
            try:
                temp_queue = queue.Queue()
                while not self.task_queue.empty():
                    try:
                        t = self.task_queue.get_nowait()
                        if t.task_id != task_id:
                            temp_queue.put(t)
                    except queue.Empty:
                        break

                while not temp_queue.empty():
                    self.task_queue.put(temp_queue.get())

            except Exception as e:
                logger.info(f"从队列中移除任务时出错: {e}")

            del self.tasks[task_id]
            logger.info(f"任务已删除: {task_id}")
            return True

    def delete_task_record(self, task_id: str) -> bool:
        """从数据库删除任务记录"""
        return self.database.delete_task(task_id)

    def clear_tasks(self, keep_processing: bool = True) -> int:
        """清空内存中的任务"""
        with self._lock:
            tasks_to_remove = []

            for task_id, task in self.tasks.items():
                if keep_processing and task.status == TaskStatus.PROCESSING:
                    continue
                tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del self.tasks[task_id]

            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except queue.Empty:
                    break

            logger.info(f"已清空 {len(tasks_to_remove)} 个内存任务")
            return len(tasks_to_remove)

    def clear_database(self) -> int:
        """清空数据库中的所有任务记录"""
        return self.database.clear_all_tasks()


# 示例使用
if __name__ == "__main__":
    import aiohttp
    import json


    # 同步任务函数
    def sync_task(name: str, duration: int = 2):
        """同步任务函数"""
        print(f"同步任务 {name} 开始执行，将耗时 {duration} 秒")
        time.sleep(duration)
        return {"message": f"同步任务 {name} 执行完成", "type": "sync"}


    # 异步任务函数
    async def async_task(name: str, duration: int = 2):
        """异步任务函数"""
        print(f"异步任务 {name} 开始执行，将耗时 {duration} 秒")
        await asyncio.sleep(duration)
        return {"message": f"异步任务 {name} 执行完成", "type": "async"}


    # 异步HTTP请求任务
    async def fetch_url_task(url: str):
        """异步获取URL内容"""
        print(f"开始获取URL: {url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                content = await response.text()
                return {
                    "url": url,
                    "status": response.status,
                    "content_length": len(content),
                    "type": "async_http"
                }


    # 异步文件操作任务
    async def async_file_task(filename: str, content: str):
        """异步文件写入任务"""
        print(f"异步写入文件: {filename}")
        await asyncio.sleep(0.5)  # 模拟IO操作
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"filename": filename, "bytes_written": len(content.encode()), "type": "async_file"}


    # 混合任务（异步函数调用同步函数）
    async def mixed_task(name: str):
        """混合任务：异步函数中调用同步操作"""
        print(f"混合任务 {name} 开始")
        # 模拟一些异步操作
        await asyncio.sleep(1)
        # 执行同步操作
        result = sync_task(f"{name}_inner", 0.5)
        # 继续异步操作
        await asyncio.sleep(0.5)
        return {"message": f"混合任务 {name} 完成", "inner_result": result, "type": "mixed"}


    # 创建异步任务队列
    task_queue = AsyncTaskQueue(db_path="async_tasks.db", check_interval=0.5)

    try:
        # 启动队列
        task_queue.start()

        # 添加各种类型的任务
        task_id1 = task_queue.add_task(sync_task, {"name": "同步任务1", "duration": 1})
        task_id2 = task_queue.add_task(async_task, {"name": "异步任务1", "duration": 1.5})
        task_id3 = task_queue.add_task(mixed_task, {"name": "混合任务1"})
        task_id4 = task_queue.add_task(async_file_task, {"filename": "test_async.txt", "content": "这是异步写入的内容"})
        task_id5 = task_queue.add_task(sync_task, {"name": "同步任务2", "duration": 0.5})
        task_id6 = task_queue.add_task(async_task, {"name": "异步任务2", "duration": 1})


        # 添加一个会失败的任务
        async def failing_task():
            await asyncio.sleep(0.5)
            raise ValueError("这是一个异步任务异常")


        task_id7 = task_queue.add_task(failing_task)

        print("\n=== 队列状态 ===")
        print(json.dumps(task_queue.get_queue_status(), indent=2, ensure_ascii=False))

        # 等待任务执行
        time.sleep(10)

        print("\n=== 所有任务结果 ===")
        all_tasks = task_queue.query_tasks()
        for task in all_tasks:
            print(f"\n任务 {task['task_id']}:")
            print(f"  类型: {'异步' if task['is_async'] else '同步'}")
            print(f"  状态: {task['status']}")
            if task['result']:
                print(f"  结果: {task['result']}")
            if task['error_message']:
                print(f"  错误: {task['error_message']}")

        print("\n=== 统计信息 ===")
        stats = task_queue.get_statistics()
        print(json.dumps(stats, indent=2, ensure_ascii=False))

        # 查询特定类型的任务
        print("\n=== 异步任务 ===")
        async_tasks = task_queue.query_tasks()
        async_tasks = [t for t in async_tasks if t['is_async']]
        for task in async_tasks:
            print(f"异步任务 {task['task_id']}: {task['status']}")

        print("\n=== 同步任务 ===")
        sync_tasks = task_queue.query_tasks()
        sync_tasks = [t for t in sync_tasks if not t['is_async']]
        for task in sync_tasks:
            print(f"同步任务 {task['task_id']}: {task['status']}")

    finally:
        # 停止队列
        task_queue.stop()
        print("\n异步任务队列已停止")

        # 清理测试文件
        if os.path.exists("test_async.txt"):
            os.remove("test_async.txt")