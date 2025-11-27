# api/main.py
from __future__ import annotations

from typing import List, Optional, Any, Dict
import asyncio
import os, uuid, shutil

import uvicorn
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form, status
from pydantic import BaseModel, Field

from utils.logger import get_logger
from utils.vector_manager import VectorManager
from utils.reranker import QAPairRerankerA
from utils.taskQueue import AsyncTaskQueue

# ----------------- init -----------------
logger = get_logger("api")

# 环境变量可覆盖默认路径
STORE_DIR = os.getenv("VECTOR_STORE_DIR", "./vector_store")
MODEL_PATH = os.getenv("EMBED_MODEL_PATH", None)  # 为 None 时，VectorManager 会走默认路径

app = FastAPI(title="RAG Search/Rerank API", version="0.1.0")

taskq = AsyncTaskQueue(db_path=os.path.join(STORE_DIR, "rag_tasks.db"))


def _ensure_task_queue_db() -> None:
    """
    Ensure the async task queue database exists.
    Recreates rag_tasks.db if destroy removed the vector store folder.
    """
    db = getattr(taskq, "database", None)
    db_path = getattr(db, "db_path", None)
    if not db_path:
        return

    db_dir = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(db_dir, exist_ok=True)

    if not os.path.exists(db_path):
        # TaskDatabase exposes _init_database which safely recreates the schema.
        db._init_database()
        logger.info("Reinitialized task queue database at %s", db_path)

# ----------------- tool func ----------------

def _save_upload_to_store_dir(file: UploadFile) -> str:
    """
    将上传的文件保存到向量库目录下的 uploads 子目录，返回保存后的绝对路径。
    """
    os.makedirs(os.path.join(STORE_DIR, "uploads"), exist_ok=True)
    ori_name = os.path.basename(file.filename or "")
    if ori_name:
        safe_name = f"{uuid.uuid4().hex}_{ori_name}"
    else:
        safe_name = f"{uuid.uuid4().hex}"
    dst = os.path.abspath(os.path.join(STORE_DIR, "uploads", safe_name))
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return dst

def _task_build_vector(doc_path: str, reinit: bool = False):
    """
    用于任务队列构建向量库
    """
    assert _vm is not None
    # 防误覆盖：这里与同步接口保持一致
    if _vm.db is not None and not reinit:
        return {"ok": False, "error": "已有向量库，需覆盖请设置 reinit=true"}

    _vm.build_vector(doc_path)
    items = _vm.show_index(limit=50)
    return {"ok": True, "built_from": doc_path, "count": len(items), "preview": items}

def _task_add_documents(doc_path: str):
    """
    用于任务队列添加文件到向量库
    """
    assert _vm is not None
    ingested = _vm.ingest_file(doc_path)
    return {
        "ok": True,
        "added_from": doc_path,
        "ingested": int(ingested),
    }


# 全局单例（进程内）
_vm: Optional[VectorManager] = None
_reranker: Optional[QAPairRerankerA] = None

@app.on_event("startup")
def _startup():
    global _vm, _reranker
    logger.info("API startup: loading VectorManager & Reranker ...")
    _vm = VectorManager(model_path=MODEL_PATH, store_dir=STORE_DIR)
    try:
        _vm.load_vector()
    except FileNotFoundError:
        logger.warning("首次启动？——请先建立向量库")
    _reranker = QAPairRerankerA(_vm, topk=8, pool_size=60, lambda_mmr=0.7)
    logger.info("API startup done.")

@app.on_event("startup")
def _startup_taskq():
    taskq.start()

@app.on_event("shutdown")
def _shutdown_taskq():
    taskq.stop()

# ----------------- Schemas -----------------

class RerankBody(BaseModel):
    query: List[str] = Field(..., description="批量查询语句")


class SearchResponse(BaseModel):
    original_text: List[str] = Field(default_factory=list)
    translated_text: List[str] = Field(default_factory=list)

class DeleteBody(BaseModel):
    indexes: Optional[List[int]] = Field(
        default=None, description="按 seq 删除"
    )
    targets: Optional[List[str]] = Field(
        default=None, description="需要匹配的目标值，单个字符串或字符串列表"
    )
    keywords: str = Field(default="source",description="metadata 中的字段名，'source'、'english'、'chinese'、'type'、'row' 。")
    fuzzy: bool = Field(default=False, description="是否使用子串模糊匹配")
    ignore_case: bool = Field(default=False, description="删除是否忽略大小写")

class StatisticItem(BaseModel):
    source: str = Field(..., description="文档来源的绝对路径")
    display_name: str = Field(..., description="用于展示的文件名")
    start_seq: int = Field(..., description="该文件第一条数据的 seq")
    end_seq: int = Field(..., description="该文件最后一条数据的 seq")
    count: int = Field(..., description="该文件对应的条目数量")

class HardRebuildBody(BaseModel):
    confirm: bool = Field(..., description="必须为 true 才会执行硬重建")

# ----------------- Endpoints -----------------

@app.get("/health")
def health():
    """
    连通性测试。
    """
    return {"ok": True, "store_dir": STORE_DIR}

@app.post("/rerank", response_model=SearchResponse)
async def rerank(
    body: RerankBody,
    topk: int = Query(1, ge=1, le=50, description="最终返回条数"),
    # 这里不能写 ge=topk，改成常量；再在函数体里校验
    pool_size: int = Query(5, ge=1, le=500, description="重排池大小（最大召回数量）"),
    lambda_mmr: float = Query(0.7, ge=0.0, le=1.0, description="MMR 多样性权重"),
    max_score: float = Query(100, description="召回的最大距离阈值"),
):

    if pool_size < topk:
        raise HTTPException(status_code=422, detail="pool_size 必须 >= topk")

    assert _vm is not None and _reranker is not None
    loop = asyncio.get_running_loop()
    queries = body.query

    def _do_rerank(query: str):
        ranked = _reranker.search_and_rerank(
            query,
            max_score=max_score,
            topk=topk,
            pool_size=pool_size,
            lambda_mmr=lambda_mmr,
        )

        english_text = ""
        chinese_text = ""
        if ranked:
            top_doc = ranked[0].get("doc")
            metadata = (getattr(top_doc, "metadata", {}) or {}) if top_doc else {}
            english_text = metadata.get("english") or ""
            chinese_text = metadata.get("chinese") or ""
        return english_text, chinese_text

    tasks = [loop.run_in_executor(None, _do_rerank, q) for q in queries]
    try:
        results = await asyncio.gather(*tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")

    original_texts: List[str] = []
    translated_texts: List[str] = []
    for english_text, chinese_text in results:
        original_texts.append(english_text)
        translated_texts.append(chinese_text)

    return SearchResponse(original_text=original_texts, translated_text=translated_texts)

@app.get("/index")
async def index_preview(
    start_seq: int = Query(1, ge=1, description="起始 seq（包含）"),
    end_seq: int = Query(10, ge=1, description="结束 seq（包含）"),
    metadata: bool = True,
):
    """
    根据 seq 范围返回索引快照。
    """
    if end_seq < start_seq:
        raise HTTPException(status_code=422, detail="end_seq 必须 >= start_seq")

    assert _vm is not None
    loop = asyncio.get_running_loop()

    def _slice_entries() -> Dict[str, Any]:
        entries = _vm.get_index_snapshot()
        ranged: List[Dict[str, Any]] = []
        for entry in entries:
            seq = entry.get("seq")
            if not isinstance(seq, int):
                continue
            if seq < start_seq:
                continue
            if seq > end_seq:
                break
            item = dict(entry)
            if not metadata:
                item.pop("metadata", None)
            ranged.append(item)
        return {"count": len(ranged), "items": ranged}

    try:
        return await loop.run_in_executor(None, _slice_entries)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="尚未构建向量库或索引缓存")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"index 缓存读取失败: {e}" )

@app.get("/statistic", response_model=List[StatisticItem])
async def statistic():
    """
    返回向量库内各源文件的 seq 范围与数量。
    """
    assert _vm is not None
    loop = asyncio.get_running_loop()

    def _collect_statistics() -> List[Dict[str, Any]]:
        try:
            return _vm.get_source_statistics()
        except FileNotFoundError:
            return []

    try:
        return await loop.run_in_executor(None, _collect_statistics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"show_index 失败: {e}")

@app.post("/destroy")
async def destroy(remove_dir: bool = False):
    """
    删除现有向量库（FAISS + _id_map.json），并清空内存中的 db。
    Args:
        remove_dir: 为 True 时，尝试直接删除整个 store_dir 目录；
                    为 False 时，仅清空目录内容并保留目录本身。默认为 False。

    """
    if _vm is None:
        raise HTTPException(status_code=500, detail="VectorManager 未初始化")

    deleted = _vm.drop_vector_store(remove_dir=remove_dir)
    return {
        "ok": True,
        "remove_dir": remove_dir,
        "deleted_entries": int(deleted),
        "store_dir": _vm.store_dir,
    }

@app.post("/delete")
async def delete(body: DeleteBody):
    """seq 删除与 metadata 删除的统一入口"""
    assert _vm is not None
    if not body.indexes and not body.targets:
        raise HTTPException(status_code=400, detail="indexes 和 docs_name 不能同时为空")

    loop = asyncio.get_running_loop()
    response: Dict[str, Any] = {"ok": True}

    if body.indexes:
        requested_seqs = sorted(set(body.indexes))

        def _delete_seq():
            try:
                index_view = _vm.show_index()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"show_index 失败: {e}")

            seq2pos = {}
            for item in index_view:
                seq = item.get("seq")
                pos = item.get("pos")
                if isinstance(seq, int) and isinstance(pos, int):
                    seq2pos[seq] = pos

            resolved_pos: List[int] = []
            not_found: List[int] = []
            for s in requested_seqs:
                p = seq2pos.get(s)
                if p is None:
                    not_found.append(s)
                else:
                    resolved_pos.append(p)

            if not resolved_pos:
                raise HTTPException(
                    status_code=404,
                    detail={"msg": "没有可删除的项（seq 未匹配到 pos）", "not_found_seqs": not_found}
                )

            if not hasattr(_vm, "delete_by_index"):
                raise HTTPException(status_code=501, detail="VectorManager 未实现 delete_by_index")

            try:
                pos_list = sorted(set(resolved_pos))
                deleted = _vm.delete_by_index(pos_list)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"delete_by_index 失败: {e}")

            return {
                "requested_seqs": requested_seqs,
                "resolved_pos": pos_list,
                "not_found_seqs": not_found,
                "deleted": int(deleted),
            }

        response["seq"] = await loop.run_in_executor(None, _delete_seq)

    if body.targets:
        def _delete_meta():
            deleted = _vm.delete_by_metadata(
                targets=body.targets,
                fuzzy=body.fuzzy,
                ignore_case=body.ignore_case,
                key=body.keywords,
            )
            return {"deleted_docs_name": body.targets, "deleted": int(deleted)}

        response["metadata"] = await loop.run_in_executor(None, _delete_meta)

    return response


#----任务队列版本----

@app.post("/tasks/build", status_code=status.HTTP_202_ACCEPTED)
def enqueue_build(
    reinit: bool = Form(False),
    doc_path: Optional[str] = Form(None),
    file: UploadFile | None = File(None),
):
    # 解析输入：file 优先，其次 doc_path
    if file is not None:
        doc_path_resolved = _save_upload_to_store_dir(file)
    elif doc_path:
        doc_path_resolved = doc_path
    else:
        return {"ok": False, "error": "请提供 doc_path 或上传 file"}

    _ensure_task_queue_db()
    tid = taskq.add_task(_task_build_vector, {"doc_path": doc_path_resolved, "reinit": reinit})
    return {"ok": True, "task_id": tid, "status_url": f"/tasks/{tid}"}

@app.post("/tasks/add", status_code=status.HTTP_202_ACCEPTED)
def enqueue_add(
    doc_path: Optional[str] = Form(None),
    file: UploadFile | None = File(None),
):
    if file is not None:
        doc_path_resolved = _save_upload_to_store_dir(file)
    elif doc_path:
        doc_path_resolved = doc_path
    else:
        return {"ok": False, "error": "请提供 doc_path 或上传 file"}

    _ensure_task_queue_db()
    tid = taskq.add_task(_task_add_documents, {"doc_path": doc_path_resolved})
    return {"ok": True, "task_id": tid, "status_url": f"/tasks/id/{tid}"}


@app.get("/tasks/id/{task_id}")
def get_task(task_id: str):
    st = taskq.get_task_status(task_id)
    if not st:
        return {"ok": False, "error": "task_id 不存在"}
    return st

@app.get("/tasks/task_list")
def get_task_list():
#    st = taskq.get_queue_status()
    st = taskq.get_statistics()
    if not st:
        return {"ok": False, "error": "task_list 为空"}
    return st

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8031, reload=True)
