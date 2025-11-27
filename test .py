# -*- coding: utf-8 -*-
"""
test.py - 功能验证脚本

使用：
    python test.py
"""
import os
import time
import json
from typing import Dict, Any, List, Optional

from utils.vector_manager import VectorManager


# ===================== 可调参数 =====================

MODEL_PATH: Optional[str] = None     # 为空：VectorManager 默认模型路径
STORE_DIR: str = f"./test/_vm_test_rich_{int(time.time())}"

# 让每个文档尽量产生 >=3 个分块（VectorManager 中 chunk_size=1000, chunk_overlap=100）
CHUNK_TARGET_CHARS = 2400      # 单文档目标总字数
PER_MARKER_TARGET = 900        # 每个 marker 段目标字数
FILL_SENTENCE = (
    "Filler text to expand the document length for chunking. "
    "This sentence helps us exceed the splitter threshold. "
)


# ===================== 工具函数 =====================

def print_header(title: str) -> None:
    bar = "=" * 20
    print(f"\n{bar} {title} {bar}\n")


def read_id_map(vm: "VectorManager") -> Dict[str, Any]:
    """读取 vm._id_map_path 的 JSON（兼容旧/新实现）。"""
    path = getattr(vm, "_id_map_path", None)
    if not path or not os.path.exists(path):
        return {"version": 2, "items": []}
    try:
        if hasattr(VectorManager, "_read_id_map"):
            return VectorManager._read_id_map(path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"读取 _id_map.json 失败: {e}")
        return {"version": 2, "items": []}


def visible_items(vm: "VectorManager") -> List[Dict[str, Any]]:
    """返回未删除的 items（按 seq 升序）。"""
    m = read_id_map(vm)
    items = [it for it in m.get("items", []) if not it.get("deleted")]
    items.sort(key=lambda x: x.get("seq", 10**12))
    return items


def assert_seq_continuous(vm: "VectorManager") -> None:
    """断言可见项的 seq 连续（1..N）。"""
    vis = visible_items(vm)
    seqs = [it.get("seq") for it in vis]
    if not seqs:
        return
    expected = list(range(1, len(seqs) + 1))
    assert seqs == expected, f"seq 非连续: got={seqs}, expected={expected}"


def ensure_min_visible(vm: "VectorManager", min_count: int) -> None:
    """确保可见条数不少于 min_count。"""
    vis = visible_items(vm)
    assert len(vis) >= min_count, f"可见条目不足：{len(vis)} < {min_count}"


def make_long_text(head: str, markers: List[str]) -> str:
    """
    生成一篇长文本：为每个 marker 生成一个段落（到达 PER_MARKER_TARGET），
    最后整体补齐到 CHUNK_TARGET_CHARS。使用列表累积避免 O(n^2) 拼接与无界增长。
    """
    parts: List[str] = []
    total_len = 0
    for mk in markers:
        seg: List[str] = [
            f"{head} | marker: {mk}\n",
            (f"{head} is discussed here with context around '{mk}'. "
             f"This paragraph intentionally repeats some tokens to encourage chunking. ")
        ]
        seg_len = sum(len(s) for s in seg)
        # 补齐到每 marker 目标长度
        unit_len = len(FILL_SENTENCE)
        need = max(0, PER_MARKER_TARGET - seg_len)
        if need > 0:
            repeats = (need + unit_len - 1) // unit_len
            seg.extend([FILL_SENTENCE] * repeats)
            seg_len += repeats * unit_len
        seg.append("\n\n")
        seg_len += 2
        parts.extend(seg)
        total_len += seg_len

    # 文档级别再补齐到总目标
    unit_len = len(FILL_SENTENCE)
    need_doc = max(0, CHUNK_TARGET_CHARS - total_len)
    if need_doc > 0:
        repeats = (need_doc + unit_len - 1) // unit_len
        parts.extend([FILL_SENTENCE] * repeats)
        total_len += repeats * unit_len

    return "".join(parts)


def write_demo_docs(root: str) -> Dict[str, str]:
    """
    创建 4 篇较长的文本，保证多分块：
    - a.txt: apples + PATTERN-111 + oranges
    - b.txt: neural networks + remove_me（删除标记）
    - c.txt: grapes + PATTERN-999 + apples
    - d.txt: vector databases（无删除标记，作为保底剩余）
    """
    os.makedirs(root, exist_ok=True)
    paths = {
        "a": os.path.join(root, "a.txt"),
        "b": os.path.join(root, "b.txt"),
        "c": os.path.join(root, "c.txt"),
        "d": os.path.join(root, "d.txt"),
    }

    with open(paths["a"], "w", encoding="utf-8") as f:
        f.write(make_long_text("Doc A about apples/oranges", ["apples", "PATTERN-111", "oranges"]))

    with open(paths["b"], "w", encoding="utf-8") as f:
        f.write(make_long_text("Doc B about ML", ["neural networks", "remove_me", "remove_me again"]))

    with open(paths["c"], "w", encoding="utf-8") as f:
        f.write(make_long_text("Doc C about grapes/apples", ["grapes", "PATTERN-999", "apples"]))

    with open(paths["d"], "w", encoding="utf-8") as f:
        f.write(make_long_text("Doc D about vector databases", ["faiss index", "ANN search", "HNSW graph"]))

    return paths


# ===================== 主流程 =====================

def main():
    os.makedirs("./test", exist_ok=True)
    print_header("Prepare long demo docs")
    docs_dir = os.path.join(STORE_DIR, "demo_docs")
    paths = write_demo_docs(docs_dir)
    for k, v in paths.items():
        print(f"  {k}: {v}")

    print_header("Init VectorManager & Build")
    vm = VectorManager(model_path=MODEL_PATH, store_dir=STORE_DIR)
    # 先用 a.txt 构建
    if hasattr(vm, "build_vector"):
        vm.build_vector(paths["a"])
    else:
        vm.build_index(paths["a"])
    # 再追加 b/c/d
    vm.add_documents(paths["b"])
    vm.add_documents(paths["c"])
    vm.add_documents(paths["d"])
    # 刷新视图并做基本断言
    vm.show_index(limit=0)
    assert_seq_continuous(vm)
    ensure_min_visible(vm, min_count=8)   # 预期至少有 8 个可见分块

    # ---- 检索（删除前）
    print_header("Search before deletions: 'apples' | 'neural networks'")
    for q in ["apples", "neural networks"]:
        res = vm.search(q, k=5, max_score=1e9)
        for i, (doc, score) in enumerate(res, 1):
            snip = (getattr(doc, "page_content", "") or "")[:80].replace("\n", " ")
            print(f"[{q}] Top {i} | score={score:.4f} | {snip}...")

    # ---- contains 删除（B 的多个分块）
    print_header("delete_documents(contains='remove_me')")
    n1 = vm.delete_documents(contains="remove_me")
    print(f"contains 删除新增软删：{n1}")
    vm.show_index(limit=0)
    assert_seq_continuous(vm)
    ensure_min_visible(vm, min_count=5)

    # ---- regex 删除（A/C 的 PATTERN-111 / PATTERN-999，跨文档多分块）
    print_header(r"delete_documents(regex='PATTERN-\d{3}')")
    n2 = vm.delete_documents(regex=r"PATTERN-\d{3}")
    print(f"regex 删除新增软删：{n2}")
    vm.show_index(limit=0)
    assert_seq_continuous(vm)
    ensure_min_visible(vm, min_count=3)

    # ---- 按 seq 删除（删除当前可见列表的第 1、3 条）
    print_header("delete_by_index([1, 3], by='seq')")
    # 兼容旧签名
    try:
        n3 = vm.delete_by_index([1, 3], by="seq")
    except TypeError:
        # 旧实现不支持 by 参数时，回退按 id 删除
        vis = visible_items(vm)
        ids = [vis[i]["id"] for i in (0, 2) if i < len(vis)]
        n3 = vm.delete_documents(ids=ids) if ids else 0
    print(f"seq 删除新增软删：{n3}")
    vm.show_index(limit=0)
    assert_seq_continuous(vm)
    # 至少保留 2 个可见，便于后续测试
    ensure_min_visible(vm, min_count=2)

    # ---- 按 pos 删除（删除“当前可见中 pos 最大”的一条）
    print_header("delete_by_index(max_visible_pos, by='pos')")
    vis = visible_items(vm)
    max_pos = max(it["pos"] for it in vis)
    try:
        n4 = vm.delete_by_index(max_pos, by="pos")
    except TypeError:
        # 回退按 id
        target_id = next((it["id"] for it in vis if it["pos"] == max_pos), None)
        n4 = vm.delete_documents(ids=[target_id]) if target_id else 0
    print(f"pos 删除新增软删：{n4}")
    vm.show_index(limit=0)
    assert_seq_continuous(vm)
    ensure_min_visible(vm, min_count=1)

    # ---- 按 id 删除（若仍有可见项>1，删除前几条但保底留 1 条）
    print_header("delete_documents(ids=[...]) (leave one visible)")
    vis = visible_items(vm)
    if len(vis) > 1:
        target_ids = [it["id"] for it in vis[:max(1, len(vis)-1)]]  # 留最后 1 条不删
        n5 = vm.delete_documents(ids=target_ids)
        print(f"id 删除新增软删：{n5}")
        vm.show_index(limit=0)
        assert_seq_continuous(vm)
    else:
        print("只剩 1 条可见，跳过按 id 删除。")

    # ---- 检索（删除后）
    print_header("Search after deletions: 'apples' | 'neural networks' | 'vector databases'")
    for q in ["apples", "neural networks", "vector databases"]:
        res = vm.search(q, k=5, max_score=1e9)
        if not res:
            print(f"[{q}] 无结果（可能已被软删过滤）。")
        for i, (doc, score) in enumerate(res, 1):
            snip = (getattr(doc, "page_content", "") or "")[:80].replace("\n", " ")
            print(f"[{q}] Top {i} | score={score:.4f} | {snip}...")

    # ---- 总结
    print_header("Summary")
    m = read_id_map(vm)
    total = len(m.get("items", []))
    deleted = sum(1 for it in m.get("items", []) if it.get("deleted"))
    visible = total - deleted
    print(f"总条目：{total} | 已软删：{deleted} | 可见：{visible}")
    print(f"测试存档于: {STORE_DIR}")
    print(f"向量库状态: {os.path.join(STORE_DIR, '_id_map.json')}")


if __name__ == "__main__":
    main()
