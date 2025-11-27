# utils/vector_manager.py
import csv
import os
import json
import shutil
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Set

from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredWordDocumentLoader,
    PyPDFLoader, CSVLoader
)
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from utils.logger import get_logger

logger = get_logger("vector_manager")

class VectorManager:
    """
    统一管理 FAISS 向量库的构建、加载、更新、检索与软删除（单一 _id_map.json）

    _id_map.json 结构:
    {
      "version": 2,
      "items": [
        { "pos": int, "id": str, "deleted": bool, "seq": int|null }
      ]
    }

    说明：
    - seq 为展示/交互的连续序号；底层以 pos/id 为准；search/show 会过滤 deleted。
    """

    def __init__(self, model_path: Optional[str] = None, store_dir: str = "vector_store"):
        """
        Args:
            model_path: HuggingFaceEmbeddings 模型路径；为 None 时使用默认路径。
            store_dir: 向量库及元数据（FAISS 与 _id_map.json）的保存目录。
        """
        self.store_dir = store_dir
        if model_path is None:
            model_path = self.get_default_model_path()
        logger.info(f"加载 embedding 模型：{model_path}")
        self.embeddings = HuggingFaceEmbeddings(model_name=model_path,model_kwargs={"device":"cuda"})
        self.db: Optional[FAISS] = None

        os.makedirs(store_dir, exist_ok=True)
        self._id_map_path = os.path.join(self.store_dir, "_id_map.json")
        self._index_stats_path = os.path.join(self.store_dir, "_index_stats.json")
        self._stats_path = os.path.join(self.store_dir, "_source_stats.json")
        self._hard_delete_threshold = 50
        self._translation_doc_type = "translation_pair"
        self._csv_batch_size = 256
        self._csv_log_every = 2000

    # ======================= 基础构建/加载 =======================
    def build_vector(self, doc_path: str) -> None:
        """
        首次构建向量库，并初始化 _id_map.json。

        Args:
            doc_path: 待构建的原始文档路径（.txt / .pdf / .docx/ .csv）。
        """
        logger.info(f"开始构建向量库，使用模型：{self.embeddings.model_name}")
        ext = os.path.splitext(doc_path)[1]

        if ext == ".csv":
            docs = self._load_csv_as_documents(doc_path)
        else:
            loader = self.load_document(doc_path,ext=ext)
            documents = loader.load()
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = splitter.split_documents(documents)
            logger.info(f"文档加载完成，共 {len(docs)} 个分块")

        self.db = FAISS.from_documents(docs, self.embeddings)
        self.db.save_local(self.store_dir)
        logger.info(f"向量库已构建并保存到：{self.store_dir}")

        # 初始化 id_map（全部未删除），并写入 seq
        self._rebuild_id_map_from_db(assign_seq=True)
        self.refresh_index_cache(quiet=True)

    def load_vector(self) -> None:
        """
        加载已有向量库；若不存在则抛出 FileNotFoundError。
        同步/纠正 _id_map.json 与 DB 的一致性。
        """
        logger.info("加载已保存的向量库...")

        if not os.path.exists(self.store_dir):
            raise FileNotFoundError(f"向量库目录不存在：{self.store_dir}")

        index_path = os.path.join(self.store_dir, f"{self.index_name}.faiss") \
            if hasattr(self, "index_name") else os.path.join(self.store_dir, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"向量库索引文件不存在：{index_path}")

        try:
            self.db = FAISS.load_local(
                self.store_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except RuntimeError as e:
            msg = str(e)
            if "could not open" in msg or "No such file or directory" in msg:
                raise FileNotFoundError(
                    f"向量库索引文件无法打开（可能已删除）：{index_path}"
                ) from e
            raise

        logger.info("向量库加载完成。")
        self._sync_id_map_with_db()

    def search(self, query: str, k: int = 3, max_score: float = 1000) -> List[Tuple[Any, float]]:
        """
        相似度检索（过滤软删除）。

        Args:
            query: 查询文本。
            k: 期望返回的结果数量（过滤后仍可能 < k）。
            max_score: 最大接受的距离分数（数值越小越相近；不同度量取值含义由底层 FAISS/向量空间决定）。

        Returns:
            [(Document, score)] 的列表，按 score 递增排序。
        """
        if not self.db:
            self.load_vector()

        logger.info(f"执行检索（过滤软删除）：{query}")
        deleted_ids = self._load_deleted_id_set()

        # 向量化并扩大候选，避免过滤后数量不足
        vec = np.array([self.embeddings.embed_query(query)], dtype="float32")
        topn = max(k * 5, k)
        D, I = self.db.index.search(vec, topn)  # D: 距离/相似度, I: pos
        mapping = self._alive_mapping_from_db(self.db)
        store = self.db.docstore

        raw: List[Tuple[Any, float]] = []
        for dist, pos in zip(D[0], I[0]):
            if pos == -1:
                continue
            doc_id = mapping.get(pos)
            if not doc_id or doc_id in deleted_ids:
                continue
            d = self._get_doc(store, doc_id)
            if not d:
                continue
            raw.append((d, float(dist)))

        logger.info(f"初始检索（过滤软删后）共有 {len(raw)} 条候选。")

        # 过滤 + 排序 + 截断
        results = [(doc, score) for (doc, score) in raw if score <= max_score]
        results.sort(key=lambda x: x[1])
        results = results[:k]

        logger.info(f"筛选后保留 {len(results)} 条（距离分数 <= {max_score}）。")
        for i, (doc, score) in enumerate(results, start=1):
            content = (getattr(doc, "page_content", "") or "")[:100].replace("\n", " ")
            logger.info(f"[Top {i}] 距离分数：{score:.4f} | 内容片段：{content}...")
        return results

    def import_translation_pairs(
        self,
        csv_path: str,
        encoding: str = "utf-8",
        delimiter: str = ",",
        batch_size: Optional[int] = None,
    ) -> int:
        """流式导入翻译对 CSV，批量写入避免一次性占满内存"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"文件路径不存在：{csv_path}")
        try:
            self._ensure_db_loaded()
        except FileNotFoundError as exc:
            raise FileNotFoundError("未检测到已存在的向量库，请先调用 build_vector 再导入 CSV") from exc

        abs_path = os.path.abspath(csv_path)
        filename = os.path.basename(abs_path) # 形如{uuid}_{ori_name}
        parts = filename.split("_",1)
        if len(parts) == 2:
            ori_name = parts[1]
        else:
            ori_name = filename #分割失败兜底

        batch_cap = batch_size or self._csv_batch_size
        if batch_cap <= 0:
            batch_cap = self._csv_batch_size
        ingest_total = 0
        skipped = 0
        batch: List[Document] = []
        started = time.time()

        def flush_batch() -> None:
            nonlocal batch, ingest_total
            if not batch:
                return
            start_idx = ingest_total + 1
            end_idx = ingest_total + len(batch)
            logger.info(f"CSV 导入：写入第 {start_idx}-{end_idx} 行")
            if not self.db:
                raise RuntimeError("向量库未加载")
            self.db.add_documents(batch)
            ingest_total = end_idx
            batch = []
            if ingest_total % max(self._csv_log_every, batch_cap) == 0:
                elapsed = time.time() - started
                logger.info(f"CSV 导入进度：累计 {ingest_total} 条，耗时 {elapsed:.1f}s")

        with open(csv_path, "r", encoding=encoding, newline="") as fp:
            reader = csv.reader(fp, delimiter=delimiter)
            for idx, row in enumerate(reader, start=1):
                if len(row) < 2:
                    skipped += 1
                    continue
                english = (row[0] or "").strip()
                chinese = (row[1] or "").strip()
                if not english or not chinese:
                    skipped += 1
                    continue
                metadata = {
                    "type": self._translation_doc_type,
                    "english": english,
                    "chinese": chinese,
                    "source": abs_path,
                    "filename": ori_name,
                    "row": idx,
                }
                batch.append(Document(page_content=english, metadata=metadata))
                if len(batch) >= batch_cap:
                    flush_batch()

        flush_batch()

        if ingest_total == 0:
            raise ValueError(f"{csv_path} 未读取到有效的翻译对，请确认至少两列且包含英文/中文内容")

        elapsed_total = time.time() - started
        logger.info(f"CSV 导入完成：成功 {ingest_total} 条，跳过 {skipped} 条，用时 {elapsed_total:.1f}s；开始持久化")
        if not self.db:
            raise RuntimeError("向量库未加载")
        self.db.save_local(self.store_dir)
        self._sync_id_map_with_db()
        logger.info("CSV 导入完成：向量库及 _id_map.json 已同步")
        self.refresh_index_cache(quiet=True)
        return ingest_total

    def add_documents(self, new_doc_path: str) -> int:
        """
        增量添加 txt/pdf/docx 文档（追加向量），并扩充 _id_map.json。

        Args:
            new_doc_path: 新增文档路径。

        Returns:
            int: 新增的分段数量。
        """
        if not self.db:
            self.load_vector()

        loader = self.load_document(new_doc_path)
        new_docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(new_docs)
        added_chunks = len(chunks)
        if added_chunks == 0:
            logger.warning(f"未从 {new_doc_path} 中解析出有效文本，不执行追加。")
            return 0

        self.db.add_documents(chunks)
        self.db.save_local(self.store_dir)
        logger.info(f"新增文档已加入向量库，共新增 {added_chunks} 段。")

        # 扩充 id_map
        self._sync_id_map_with_db()
        self.refresh_index_cache(quiet=True)
        return added_chunks

    def ingest_file(self, file_path: str, encoding: str = "utf-8", delimiter: str = ",") -> int:
        """
        统一入口：根据文件类型路由到 CSV 翻译对导入或普通文档追加。

        Args:
            file_path: 输入文件路径。
            encoding: 读取 CSV 时使用的编码。
            delimiter: CSV 分隔符。

        Returns:
            int: 成功写入向量库的条目/分段数量。

        Raises:
            ValueError: 文件扩展名不受支持。
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            return self.import_translation_pairs(file_path, encoding=encoding, delimiter=delimiter)
        elif ext in {".txt", ".pdf", ".docx"}:
            return self.add_documents(file_path)
        else:
            raise ValueError(f"暂不支持的文件类型：{ext or '[无扩展名]'}，仅支持 csv/txt/pdf/docx。")

    def _ensure_db_loaded(self) -> None:
        """确保内存中的向量库已加载，如未加载则尝试从磁盘读取"""
        if self.db is not None:
            return
        self.load_vector()

    def _rebuild_id_map_from_db(self, assign_seq: bool = True) -> None:
        """
        根据当前 DB 完整重建 _id_map.json（全部 deleted=False；可同时分配连续 seq）。

        Args:
            assign_seq: 是否为未删除项重排连续 seq。
        """
        if not self.db:
            self.load_vector()

        mapping = self._alive_mapping_from_db(self.db)  # pos -> doc_id
        items = [{"pos": pos, "id": doc_id, "deleted": False, "seq": None}
                 for pos, doc_id in sorted(mapping.items(), key=lambda kv: kv[0])]

        if assign_seq:
            self._assign_seq_inplace(items)

        data = {"version": 2, "items": items}
        self._write_id_map(self._id_map_path, data)

    def _sync_id_map_with_db(self) -> None:
        """
        将 _id_map.json 与 DB 对齐：
        - 新增 pos：追加 {deleted=False, seq=None}
        - 缺失/过期 pos：移除（通常硬重建后才会发生）
        - 不调整 deleted；不重排 seq（交给 show_index 重排）
        """
        if not self.db:
            self.load_vector()

        data = self._read_id_map(self._id_map_path)
        items = data.get("items", [])

        # DB 现状
        mapping = self._alive_mapping_from_db(self.db)     # pos -> doc_id
        alive_pos = set(mapping.keys())
        pos_to_item = {it["pos"]: it for it in items}

        # 移除无效 pos
        items = [it for it in items if it["pos"] in alive_pos]

        # 追加新 pos
        for pos, doc_id in mapping.items():
            if pos not in pos_to_item:
                items.append({"pos": pos, "id": doc_id, "deleted": False, "seq": None})

        self._normalize_items_inplace(items)
        data["items"] = sorted(items, key=lambda x: x["pos"])
        self._write_id_map(self._id_map_path, data)

    def _load_deleted_id_set(self) -> Set[str]:
        """
        从 _id_map.json 读取已软删的 doc_id 集合。

        Returns:
            已被标记 deleted=True 的文档 ID 集合。
        """
        data = self._read_id_map(self._id_map_path)
        return self._deleted_id_set_from_items(data.get("items", []))



    def _maybe_hard_delete(self, deleted_total: int) -> None:
        """       检查是否达到硬删除阈值        """
        if deleted_total <= self._hard_delete_threshold:
            return

        logger.info(
            f"软删除 {deleted_total} 条，达到硬删除阈值 {self._hard_delete_threshold} "
        )
        self._hard_delete()



    def _hard_delete(self) -> None:
        if not self.db:
            self.load_vector()

        data = self._read_id_map(self._id_map_path)
        items = data.get("items", [])
        removed_ids = [it["id"] for it in items if it.get("deleted", False)]

        if not removed_ids:
            logger.info("[硬删除]removed_ids为空")
            return

        if hasattr(self.db, "delete"):
            self.db.delete(ids=removed_ids)

        else:
            logger.warning("[硬删除] 检测到FAISS实例无delete方法，将通过重建索引方式删除已标记ID")
            keeper_docs = []
            store = self.db.docstore

            for it in items:
                if it.get("deleted", False):
                    continue
                doc = self._get_doc(store, it["id"])
                if doc:
                    keeper_docs.append(doc)

            if keeper_docs:
                self.db = FAISS.from_documents(keeper_docs, self.embeddings)

            else:
                self.db = None

        if self.db:
            self.db.save_local(self.store_dir)
            self._rebuild_id_map_from_db(assign_seq=True)
            logger.info(
                "[硬删除] 硬删除操作完成，共删除%s个已标记的ID，已重建id_map并更新seq序号",
                len(removed_ids)
            )
            self.refresh_index_cache(quiet=True)

        else:
            logger.warning(
                "[硬删除] 硬删除后向量库为空，清空id_map文件并失效统计数据，_id_map路径：%s，_source_stats路径：%s",
                self._id_map_path,
                self._stats_path
            )
            self._write_id_map(self._id_map_path, {"version": 2, "items": []})
            self.invalidate_index_cache()


    # ======================= 展示（按 seq 顺序；重排 seq） =======================

    def show_index(self, limit: int = 20, preview_chars: int = 120, show_metadata: bool = False, quiet: bool = False) -> List[Dict[str, Any]]:
        """
        以“当前可见快照”形式展示向量库（仅显示未删除项），并将可见项的 seq 重排为 1..N 写回 _id_map.json。

        Args:
            limit: 预览条数上限（为 0 时返回全部可见项）。
            preview_chars: 每条预览截取的字符数。
            show_metadata: 是否在返回项中包含 metadata 字段。

        Returns:
            按 seq 顺序排列的预览列表：[{pos, id, preview, (metadata)}]。
        """
        if not self.db:
            self.load_vector()

        # 确保映射覆盖最新 DB
        self._sync_id_map_with_db()
        data = self._read_id_map(self._id_map_path)
        items = data.get("items", [])

        store = self.db.docstore

        # 仅对未删除项重排 seq
        self._assign_seq_inplace(items)
        data["items"] = items
        self._write_id_map(self._id_map_path, data)

        # 生成可见项（按 seq 升序）
        visible = [it for it in items if not it.get("deleted", False)]
        visible.sort(key=lambda x: x.get("seq", 10**12))

        entries: List[Dict[str, Any]] = []
        for it in visible:
            doc = self._get_doc(store, it["id"])
            if not doc:
                continue
            item = {
                "seq": it["seq"],
                "pos": it["pos"],
                "id": it["id"],
                "preview": (getattr(doc, "page_content", "") or "")[:preview_chars],
            }
            if show_metadata:
                item["metadata"] = getattr(doc, "metadata", None)
            entries.append(item)
        if limit == 0:
            shown = entries
        else:
            shown = entries[:limit]
        if not quiet:
            logger.info(f"向量库预览：当前（过滤软删）共有 {len(entries)} 条，按 seq 顺序显示前 {len(shown)} 条。")
            for i, e in enumerate(shown, 1):
                snip = e["preview"].replace("\n", " ")
                logger.info(f"[{i}] id={e['id']} | pos={e['pos']} | {snip}...")
        return shown

    def refresh_index_cache(self, quiet: bool = False) -> List[Dict[str, Any]]:
        """
        重建索引快照与 source 统计缓存。
        """
        entries = self.show_index(limit=0, preview_chars=120, show_metadata=True, quiet=quiet)
        self._write_index_snapshot(entries)
        stats = self._aggregate_source_statistics(entries)
        self._write_source_statistics(stats)
        return entries

    def get_index_snapshot(self) -> List[Dict[str, Any]]:
        """
        读取缓存的索引快照；如无可用缓存则重建。
        """
        cached = self._read_index_snapshot()
        if cached is not None:
            return cached
        return self.refresh_index_cache(quiet=True)

    def get_source_statistics(self) -> List[Dict[str, Any]]:
        """
        获取缓存的 source 统计信息；若不存在则重建缓存。
        """
        cached = self._read_source_statistics()
        if cached is not None:
            return cached
        self.refresh_index_cache(quiet=True)
        cached = self._read_source_statistics()
        return cached or []

    def invalidate_index_cache(self) -> None:
        """
        删除索引/统计缓存文件。
        """
        for path in (self._index_stats_path, self._stats_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning(f"删除索引缓存失败：{exc}")

    def _read_index_snapshot(self) -> Optional[List[Dict[str, Any]]]:
        if not os.path.exists(self._index_stats_path):
            return None
        try:
            with open(self._index_stats_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return None
        except Exception as exc:
            logger.warning(f"读取索引缓存失败：{exc}")
            return None

    def _read_source_statistics(self) -> Optional[List[Dict[str, Any]]]:
        if not os.path.exists(self._stats_path):
            return None
        try:
            with open(self._stats_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return None
        except Exception as exc:
            logger.warning(f"读取统计缓存失败：{exc}")
            return None

    def _write_index_snapshot(self, entries: List[Dict[str, Any]]) -> None:
        try:
            with open(self._index_stats_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning(f"写入索引缓存失败：{exc}")

    def _write_source_statistics(self, stats: List[Dict[str, Any]]) -> None:
        try:
            with open(self._stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning(f"写入统计缓存失败：{exc}")



    # ======================= 软删除（按 id / contains / regex） =======================

    def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        contains: Optional[str] = None,
        regex: Optional[str] = None
    ) -> int:
        """
        软删除（仅修改 _id_map.json，不重建索引）。

        Args:
            ids: 需要删除的 doc_id 列表。
            contains: 文本或元数据包含该子串即加入删除集合。
            regex: 正则匹配文本或元数据即加入删除集合。

        Returns:
            本次“新增软删”的条数（若匹配项此前已删除，不重复计数）。
        """
        import re as _re

        if not self.db:
            self.load_vector()

        data = self._read_id_map(self._id_map_path)
        items = data.get("items", [])
        id_to_item = {it["id"]: it for it in items}

        store = self.db.docstore
        mapping = self._alive_mapping_from_db(self.db)
        all_ids = list(set(mapping.values()))

        candidates: Set[str] = set()

        # 1) 指定ID
        if ids:
            asked = set(ids)
            real = list(asked & set(all_ids))
            if not real:
                logger.warning("delete_documents：提供的 ids 未命中任何文档。")
            candidates.update(real)

        # 2) 子串匹配
        if contains:
            for doc_id in all_ids:
                d = self._get_doc(store, doc_id)
                if not d:
                    continue
                text_hit = contains in (getattr(d, "page_content", "") or "")
                meta_hit = any(contains in str(v) for v in (getattr(d, "metadata", {}) or {}).values())
                if text_hit or meta_hit:
                    candidates.add(doc_id)

        # 3) 正则匹配
        if regex:
            pat = _re.compile(regex)
            for doc_id in all_ids:
                d = self._get_doc(store, doc_id)
                if not d:
                    continue
                text_hit = bool(pat.search(getattr(d, "page_content", "") or ""))
                meta_hit = any(pat.search(str(v)) for v in (getattr(d, "metadata", {}) or {}).values())
                if text_hit or meta_hit:
                    candidates.add(doc_id)

        if not candidates:
            logger.info("delete_documents：未匹配到需要删除的文档，索引保持不变。")
            return 0

        before = sum(1 for it in items if it.get("deleted", False))
        for doc_id in candidates:
            it = id_to_item.get(doc_id)
            if it:
                it["deleted"] = True

        # 落盘（seq 重排交给 show_index）
        data["items"] = items
        self._write_id_map(self._id_map_path, data)

        after = sum(1 for it in items if it.get("deleted", False))
        added = after - before
        logger.info(f"软删除完成：本次新增软删 {added} 条（累计 {after} 条）。")
        self._maybe_hard_delete(after)
        self.refresh_index_cache(quiet=True)
        return added

    # =======================  删除 =======================

    def drop_vector_store(self, remove_dir: bool = False) -> int:
        """
        删除当前向量库：包括 FAISS 索引文件和 _id_map.json，并清空内存中的 db。

        Args:
            remove_dir: 为 True 时，尝试直接删除整个 store_dir 目录；
                        为 False 时，仅清空目录内容并保留目录本身。

        Returns:
            实际删除的文件/子目录数量（仅作参考）。
        """
        deleted_count = 0

        if self.db is not None:
            self.db = None

        if not os.path.exists(self.store_dir):
            logger.info(f"向量库目录不存在，无需删除：{self.store_dir}")
            return 0

        if remove_dir:
            shutil.rmtree(self.store_dir, ignore_errors=True)
            logger.info(f"已删除向量库目录：{self.store_dir}")
            self.invalidate_index_cache()
            return 1

        for name in os.listdir(self.store_dir):
            path = os.path.join(self.store_dir, name)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.remove(path)
                    deleted_count += 1
                elif os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                    deleted_count += 1
            except Exception as e:
                logger.warning(f"删除 {path} 失败：{e}")

        os.makedirs(self.store_dir, exist_ok=True)
        self._write_id_map(self._id_map_path, {"version": 2, "items": []})
        logger.info(f"向量库已清空，目录保留：{self.store_dir}")
        self.invalidate_index_cache()
        return deleted_count

    def delete_by_index(self, indexes: Union[int, List[int]], by: str = "pos") -> int:
        """
        按“序号”删除（软删）。

        Args:
            indexes: 单个或多个序号；当 by='pos' 时表示 FAISS 内部位置；by='seq' 时表示当前可见连续序号。
            by: 'pos'（默认，更稳定）或 'seq'（对 UI 友好；以最近一次 show_index 的快照为准）。

        Returns:
            本次“新增软删”的条数。
        """
        if not self.db:
            self.load_vector()

        data = self._read_id_map(self._id_map_path)
        items = data.get("items", [])

        if isinstance(indexes, int):
            indexes = [indexes]

        ids: List[str] = []
        bad: List[int] = []

        if by == "pos":
            pos_to_id = {it["pos"]: it["id"] for it in items}
            for pos in indexes:
                doc_id = pos_to_id.get(pos)
                (ids.append(doc_id) if doc_id else bad.append(pos))
        elif by == "seq":
            seq_to_id = {it["seq"]: it["id"] for it in items if not it.get("deleted", False)}
            for seq in indexes:
                doc_id = seq_to_id.get(seq)
                (ids.append(doc_id) if doc_id else bad.append(seq))
        else:
            logger.warning(f"未知 by 参数：{by}，应为 'pos' 或 'seq'。")
            return 0

        if bad:
            logger.warning(f"以下 {by} 未找到（或已被软删过滤）：{bad}")
        if not ids:
            logger.info("未匹配到任何文档，索引不变。")
            return 0

        deleted = self.delete_documents(ids=ids)
        logger.info(f"按 {by} 删除完成：本次新增软删 {deleted} 条。")
        return deleted

    def delete_by_metadata(
            self,
            targets: Union[str, List[str]],
            key: str = "source",
            fuzzy: bool = False,
            ignore_case: bool = False,
    ) -> int:
        """
        根据 metadata 中指定字段的值批量删除文档（软删）。

        Args:
            targets: 需要匹配的目标值，可以是：
                     - 单个字符串
                     - 字符串列表
            key:     metadata 中的字段名， 'source'、'english'、'chinese'、'type'、'row' 。
            fuzzy:   True 表示模糊匹配（子串包含），False 表示完全匹配。
            ignore_case: 是否忽略大小写。

        Returns:
            int: 实际删除的文档数量（按 doc_id 计数）。
        """
        if not key:
            raise ValueError("delete_by_metadata: key 不能为空")

        # 规范化 targets 形态
        if isinstance(targets, str):
            raw_values = [targets]
        elif isinstance(targets, list):
            raw_values = targets
        else:
            raise TypeError("delete_by_metadata: targets 必须是 str 或 List[str]")

        key_lower = key.lower()
        # 这些字段视为“路径类”，会做绝对路径标准化
        path_like_keys = {"source", "path", "file", "file_path", "doc_path", "origin_path"}

        def _normalize(val: Any) -> Optional[str]:
            """把各种值统一成可比较的字符串形式。"""
            if val is None:
                return None
            text = str(val).strip()
            if not text:
                return None
            # 路径：转绝对路径
            if key_lower in path_like_keys:
                text = os.path.abspath(text)
            # 统一大小写
            if ignore_case:
                text = text.lower()
            return text

        # 规范化所有目标值
        targets_norm: List[str] = []
        for v in raw_values:
            norm = _normalize(v)
            if norm:
                targets_norm.append(norm)

        if not targets_norm:
            logger.info("delete_by_metadata: 传入的 targets 为空或全是空白，跳过删除。")
            return 0

        # 确保 DB 已加载
        try:
            self._ensure_db_loaded()
        except FileNotFoundError as exc:
            # 这里直接抛出去，让上层 API 决定怎么提示用户
            raise FileNotFoundError("未检测到已存在的向量库，无法按 metadata 删除") from exc

        store = self.db.docstore
        # pos -> doc_id 的映射
        mapping = self._alive_mapping_from_db(self.db)

        target_set = set(targets_norm)
        candidate_ids: Set[str] = set()

        for doc_id in set(mapping.values()):
            doc = self._get_doc(store, doc_id)
            if not doc:
                continue
            meta = getattr(doc, "metadata", {}) or {}
            norm_meta = _normalize(meta.get(key))
            if not norm_meta:
                continue

            if fuzzy:
                matched = any(t in norm_meta for t in target_set)
            else:
                matched = norm_meta in target_set

            if matched:
                candidate_ids.add(doc_id)

        if not candidate_ids:
            logger.info(f"delete_by_metadata: 未找到匹配的文档 (key={key}, targets={targets})")
            return 0

        deleted = self.delete_documents(ids=list(candidate_ids))
        logger.info(f"delete_by_metadata: 按 key={key} 删除了 {deleted} 条文档")
        return deleted

    # ======================= 工具函数 =======================

    def _load_csv_as_documents(self, csv_path: str, encoding: str = "utf-8", delimiter: str = ",") -> List[Document]:
        """
        复用 CSV 翻译对处理逻辑，将 CSV 解析为 Document 列表（供首次构建使用）
        与 import_translation_pairs 保持一致的解析规则
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 文件不存在：{csv_path}")

        abs_path = os.path.abspath(csv_path)
        filename = os.path.basename(abs_path)
        parts = filename.split("_",1)
        if len(parts) == 2:
            ori_name = parts[1]
        else:
            ori_name = filename

        docs: List[Document] = []
        skipped = 0
        started = time.time()

        with open(csv_path, "r", encoding=encoding, newline="") as fp:
            reader = csv.reader(fp, delimiter=delimiter)
            for idx, row in enumerate(reader, start=1):
                if len(row) < 2:
                    skipped += 1
                    continue
                english = (row[0] or "").strip()
                chinese = (row[1] or "").strip()
                if not english or not chinese:
                    skipped += 1
                    continue
                # 保持与 import_translation_pairs 一致的 metadata 结构
                metadata = {
                    "type": self._translation_doc_type,
                    "english": english,
                    "chinese": chinese,
                    "source": abs_path,
                    "filename": ori_name,
                    "row": idx,
                }
                docs.append(Document(page_content=english, metadata=metadata))

                # 日志输出（复用 _csv_log_every 配置）
                if len(docs) % self._csv_log_every == 0:
                    elapsed = time.time() - started
                    logger.info(f"CSV 解析进度：已处理 {len(docs)} 条，跳过 {skipped} 条，耗时 {elapsed:.1f}s")

        elapsed_total = time.time() - started
        logger.info(f"CSV 解析完成：成功解析 {len(docs)} 条，跳过 {skipped} 条，用时 {elapsed_total:.1f}s")

        if len(docs) == 0:
            raise ValueError(f"{csv_path} 未读取到有效的翻译对，请确认至少两列且包含英文/中文内容")

        return docs

    # ======================= 静态工具函数（整理为 static） =======================

    @staticmethod
    def _read_id_map(path: str) -> Dict[str, Any]:
        """读取 _id_map.json；不存在时返回空结构。"""
        if not os.path.exists(path):
            return {"version": 2, "items": []}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("version", 2)
            data.setdefault("items", [])
            return data
        except Exception as e:
            logger.warning(f"读取 {path} 失败，将重建：{e}")
            return {"version": 2, "items": []}

    @staticmethod
    def _write_id_map(path: str, data: Dict[str, Any]) -> None:
        """写回 _id_map.json。"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"写入 {path} 失败：{e}")

    @staticmethod
    def _normalize_items_inplace(items: List[Dict[str, Any]]) -> None:
        """确保 id_map items 字段齐全：deleted/seq 存在。"""
        for it in items:
            it.setdefault("deleted", False)
            it.setdefault("seq", None)

    @staticmethod
    def _assign_seq_inplace(items: List[Dict[str, Any]]) -> None:
        """按 pos 升序，为未删除项分配连续 seq；删除项 seq 置为 None。"""
        seq = 1
        for it in sorted(items, key=lambda x: x["pos"]):
            if not it.get("deleted", False):
                it["seq"] = seq
                seq += 1
            else:
                it["seq"] = None

    @staticmethod
    def _deleted_id_set_from_items(items: List[Dict[str, Any]]) -> Set[str]:
        """从 items 中提取 deleted=True 的 id 集合。"""
        return {it["id"] for it in items if it.get("deleted", False)}

    @staticmethod
    def _alive_mapping_from_db(db: FAISS) -> Dict[int, str]:
        """从 FAISS 向量库读取 pos->doc_id 映射。"""
        return getattr(db, "index_to_docstore_id", {})

    @staticmethod
    def _get_doc(store: Any, doc_id: str) -> Any:
        """兼容性读取 docstore 中的文档对象。"""
        if hasattr(store, "_dict"):
            return store._dict.get(doc_id)
        if hasattr(store, "search"):
            return store.search(doc_id)
        return None

    @staticmethod
    def get_default_model_path() -> str:
        """获取默认的 embedding 模型路径。"""
        base_dir = os.path.dirname(__file__)
        return os.path.abspath(os.path.join(base_dir, "..", "model", "embedding", "m3e-base"))

    @staticmethod
    def load_document(doc_path: str, ext: str):
        """
        根据文件类型自动选择合适的 Loader。

        Args:
            doc_path: 输入文件路径（.txt/.pdf/.docx）。
            ext: 文件后缀

        Returns:
            适配的 LangChain Loader 实例。

        Raises:
            ValueError: 不支持的文件后缀名。
        """
        if ext == ".txt":
            return TextLoader(doc_path, encoding="utf-8")
        elif ext == ".pdf":
            return PyPDFLoader(doc_path)
        elif ext == ".docx":
            return UnstructuredWordDocumentLoader(doc_path)
        else:
            raise ValueError(f"暂不支持的文件类型：{ext}")

    @staticmethod
    def _aggregate_source_statistics(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for entry in entries:
            seq = entry.get("seq")
            if not isinstance(seq, int):
                continue
            metadata = entry.get("metadata") or {}
            source_path = metadata.get("source") or metadata.get("doc_path") or metadata.get("path") or ""
            display_name = metadata.get("filename") or metadata.get("doc_name") or ""

            if not source_path:
                source_path = display_name or "unknown"
            if not display_name:
                display_name = os.path.basename(source_path) if source_path and source_path != "unknown" else "unknown"

            key = (str(source_path), str(display_name))
            current = grouped.get(key)

            if not current:
                grouped[key] = {
                    "source": str(source_path),
                    "display_name": str(display_name),
                    "start_seq": seq,
                    "end_seq": seq,
                    "count": 1,
                }
            else:
                current["count"] += 1
                if seq < current["start_seq"]:
                    current["start_seq"] = seq
                if seq > current["end_seq"]:
                    current["end_seq"] = seq

        stats = list(grouped.values())
        stats.sort(key=lambda item: item["start_seq"])
        return stats