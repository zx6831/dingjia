# utils/reranker.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import math
import re
import numpy as np

# ====== 强依赖项目内 logger（无兜底）======
from utils.logger import get_logger
_LOG_DEFAULT = get_logger("reranker")

# --------- 轻量工具 ----------
_NUM_PAT = re.compile(r"\d+(?:\.\d+)?")
_DATE_PAT = re.compile(r"\b(?:\d{4}[-/年]\d{1,2}(?:[-/月]\d{1,2}日?)?)\b")
_TOKEN_SPLIT = re.compile(r"[^a-zA-Z0-9_\u4e00-\u9fa5]+")

def _norm_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float32")
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def _norm_rows(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype="float32")
    if mat.ndim != 2:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_norm_vec(a), _norm_vec(b)))

def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in _TOKEN_SPLIT.split(text.lower()) if t]

def _entity_hits(query: str, text: str) -> bool:
    if not text:
        return False
    # 数字 / 日期 / 纯 token 重合，命中其一即视为命中
    if set(_NUM_PAT.findall(query)) & set(_NUM_PAT.findall(text)):
        return True
    if set(_DATE_PAT.findall(query)) & set(_DATE_PAT.findall(text)):
        return True
    q_tokens = set(_tokenize(query))
    t_tokens = set(_tokenize(text))
    strong = {w for w in q_tokens if len(w) >= 3}
    return bool(strong & t_tokens)

# --------- BM25-lite（无三方依赖） ----------
class _BM25Lite:
    def __init__(self, docs: List[List[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.N, 1)
        df: Dict[str, int] = {}
        for d in docs:
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log((self.N - c + 0.5) / (c + 0.5) + 1.0) for w, c in df.items()}

    def score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        if not doc_tokens:
            return 0.0
        dl = len(doc_tokens)
        tf: Dict[str, int] = {}
        for w in doc_tokens:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for w in query_tokens:
            if w not in tf:
                continue
            idf = self.idf.get(w, 0.0)
            freq = tf[w]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            s += idf * (freq * (self.k1 + 1)) / max(denom, 1e-9)
        return s

# --------- MMR 选样 ----------
def mmr_select(
    q_vec: np.ndarray,
    cand_q_vecs: np.ndarray,
    k: int,
    lambda_mult: float = 0.7,
) -> List[int]:
    """
    返回被挑选出来的下标序列（基于 question 向量做多样性）
    """
    n = len(cand_q_vecs)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    selected: List[int] = []
    # cand_q_vecs 与 q_vec 均视为已归一化
    sim_to_q = np.asarray(cand_q_vecs @ q_vec, dtype="float32")
    sim_matrix = np.asarray(cand_q_vecs @ cand_q_vecs.T, dtype="float32")
    available = np.ones(n, dtype=bool)

    while len(selected) < k:
        if not available.any():
            break
        if selected:
            max_sim = sim_matrix[:, selected].max(axis=1)
            scores = lambda_mult * sim_to_q - (1 - lambda_mult) * max_sim
        else:
            scores = sim_to_q.copy()
        scores[~available] = -np.inf
        next_idx = int(np.argmax(scores))
        if not available[next_idx]:
            break
        selected.append(next_idx)
        available[next_idx] = False
    return selected

# --------- Q/A 重排器 ----------
class QAPairRerankerA:
    """
    A 档：零外部依赖
    融合分数 = 0.55*cos(q, Q) + 0.25*cos(q, A) + 0.15*BM25(q, Q) + 0.05*overlap(Q|A)
    再配合 MMR（按 Question 向量）做多样性
    """
    def __init__(
        self,
        vm,                             # VectorManager 实例（用于拿 embeddings）
        lambda_mmr: float = 0.7,
        topk: int = 2,
        pool_size: int = 60,            # 召回候选池大小：vm.search(query, k=pool_size)
        w_qQ: float = 0.55,
        w_qA: float = 0.25,
        w_bm25: float = 0.15,
        w_overlap: float = 0.05,
        answer_len_penalty: float = 0.02,   # 过长答案惩罚（可为 0 关闭）
        answer_len_p95: int = 800,          # 超过此长度视为过长
        logger=None,                        # 可显式传入自定义 logger；不传用默认
    ):
        self.vm = vm
        self.emb = vm.embeddings
        self.lambda_mmr = lambda_mmr
        self.topk = topk
        self.pool_size = pool_size
        self.w_qQ, self.w_qA, self.w_bm25, self.w_overlap = w_qQ, w_qA, w_bm25, w_overlap
        self.answer_len_penalty = answer_len_penalty
        self.answer_len_p95 = answer_len_p95
        self.logger = logger or _LOG_DEFAULT  # 无兜底；若 utils.logger 出问题，这里在 import 时已抛错

        self.logger.info(
            "QAPairRerankerA:init lambda_mmr=%.2f topk=%d pool=%d w=[%.2f,%.2f,%.2f,%.2f] len_penalty=%.3f p95=%d",
            self.lambda_mmr, self.topk, self.pool_size, self.w_qQ, self.w_qA, self.w_bm25, self.w_overlap,
            self.answer_len_penalty, self.answer_len_p95
        )

    def search_and_rerank(
        self,
        query: str,
        max_score: float = 1e12,
        *,
        topk: Optional[int] = None,
        pool_size: Optional[int] = None,
        lambda_mmr: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        用 vm.search 取候选池，再做 A 档重排，返回前 topk
        返回项字段：{doc, base_score, fuse_score, q_text, a_text}
        """
        self.logger.info("Rerank:start query=%s", query)

        topk = topk if topk is not None else self.topk
        pool_size = pool_size if pool_size is not None else self.pool_size
        lambda_mmr = lambda_mmr if lambda_mmr is not None else self.lambda_mmr

        raw: List[Tuple[Any, float]] = self.vm.search(query, k=pool_size, max_score=max_score)
        self.logger.debug("Rerank:raw_candidates count=%d", len(raw))
        if not raw:
            self.logger.warning("Rerank:empty")
            return []

        items: List[Dict[str, Any]] = []
        for d, base_score in raw:
            q_text, a_text = self._extract_qa_from_doc(d)
            items.append({"doc": d, "base_score": float(base_score), "q_text": q_text, "a_text": a_text})
        self.logger.debug("Rerank:qa_extracted with_Q=%d", sum(1 for x in items if x['q_text']))

        q_vec = _norm_vec(np.asarray(self.emb.embed_query(query), dtype="float32"))
        q_texts = [it["q_text"] or "" for it in items]
        a_texts = [it["a_text"] or "" for it in items]
        q_tokens_all = [_tokenize(text) for text in q_texts]

        q_vecs = np.asarray(self.emb.embed_documents(q_texts), dtype="float32")
        a_vecs = np.asarray(self.emb.embed_documents(a_texts), dtype="float32")
        q_vecs = _norm_rows(q_vecs)
        a_vecs = _norm_rows(a_vecs)

        mmr_ratio = 0.4  # 从候选池按比例先挑一截
        mmr_min = max(8, topk * 2)  # 至少保留若干条（与 topk 成比例）
        mmr_max = min(pool_size, topk * 6, 128)  # 给定合理上限，防止爆炸

        mmr_k = min(
            pool_size,
            max(mmr_min, int(pool_size * mmr_ratio)),
            mmr_max,
        )

        sel_idx = mmr_select(q_vec, q_vecs, k=mmr_k, lambda_mult=lambda_mmr)
        self.logger.debug("Rerank:mmr_select mmr_k=%d selected=%d", mmr_k, len(sel_idx))
        pool = [items[i] for i in sel_idx]
        pool_q_vecs = q_vecs[sel_idx]
        pool_a_vecs = a_vecs[sel_idx]
        pool_q_tokens = [q_tokens_all[i] for i in sel_idx]

        bm25 = _BM25Lite(pool_q_tokens)

        q_tokens = _tokenize(query)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for i, it in enumerate(pool):
            cos_qQ = float(np.dot(q_vec, pool_q_vecs[i]))
            cos_qA = float(np.dot(q_vec, pool_a_vecs[i]))
            bm25_qQ = bm25.score(q_tokens, pool_q_tokens[i])
            overlap = 1.0 if (_entity_hits(query, it["q_text"]) or _entity_hits(query, it["a_text"])) else 0.0

            s = (self.w_qQ * cos_qQ +
                 self.w_qA * cos_qA +
                 self.w_bm25 * bm25_qQ +
                 self.w_overlap * overlap)

            if self.answer_len_penalty > 0 and len(it["a_text"] or "") > self.answer_len_p95:
                s -= self.answer_len_penalty

            it2 = dict(it); it2["fuse_score"] = float(s)
            scored.append((s, it2))

            if i < 5:
                self.logger.debug(
                    "Rerank:feat i=%d cos_qQ=%.4f cos_qA=%.4f bm25=%.4f overlap=%.1f fuse=%.4f q='%s'",
                    i, cos_qQ, cos_qA, bm25_qQ, overlap, s, (it['q_text'][:60] if it['q_text'] else "")
                )

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [it for _, it in scored[: topk]]
        self.logger.info("Rerank:done topk=%d pool=%d", len(top), len(pool))
        return top

    def _extract_qa_from_doc(self, doc) -> Tuple[str, str]:
        """
        兼容几种常见存储形式：
        1) doc.metadata 里有 'question' / 'answer'
        2) page_content 里按 'Q:' / 'A:' 或 '问题：' / '答案：' 等格式
        3) 兜底：整段当 Answer，Question 为空
        """
        meta = getattr(doc, "metadata", {}) or {}
        if isinstance(meta, dict):
            q = meta.get("question") or meta.get("q") or ""
            a = meta.get("answer") or meta.get("a") or ""
            if q or a:
                return str(q), str(a)

        text = (getattr(doc, "page_content", "") or "").strip()
        if not text:
            return "", ""
        m = re.search(r"(?:^|\n)\s*(?:Q:|问题[:：])\s*(.+?)(?:\n+\s*(?:A:|答案[:：])\s*(.+))?$", text, re.S | re.I)
        if m:
            q = (m.group(1) or "").strip()
            a = (m.group(2) or "").strip()
            return q, a
        m = re.search(r"(?:^|\n)\s*(?:Question[:：])\s*(.+?)(?:\n+\s*(?:Answer[:：])\s*(.+))?$", text, re.S | re.I)
        if m:
            q = (m.group(1) or "").strip()
            a = (m.group(2) or "").strip()
            return q, a
        return "", text
