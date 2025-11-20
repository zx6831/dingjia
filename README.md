# VectorManager（FAISS）——软删除 + 多文档检索

_wzx，2025.11.4_

基于 **FAISS** 的轻量向量库管理器，封装了**构建/加载/增量/检索/展示/软删除**等常用能力。

---

## TODO
* 异步调用/任务队列
* FastAPI

---
## 目录结构

与项目截图一致，核心文件如下（省略无关项）：

```
dingjia/
├─ model/
│  └─ embedding/
│     └─ m3e-base/                # 默认 embedding 模型
├─ test/
│  └─ _vm_test_rich_*/            # 运行 test.py 产生的示例向量库
│     ├─ demo_docs/               # 生成的示例文本
│     ├─ _id_map.json             # 可见性与序号映射
│     ├─ index.faiss              # FAISS 索引
│     └─ index.pkl
├─ utils/
│  ├─ __init__.py
│  ├─ logger.py
│  └─ vector_manager.py           # 核心实现
├─ vector_store/                  # 你自己的向量库
├─ environment.yaml
├─ README.md
└─ test.py                        # 一键测试
```

---

## 功能特性

* **多文件加载**：`txt / pdf / docx` 自动选择合适 Loader。
* **向量库管理**：`build_vector`（首次构建）、`load_vector`（加载）、`add_documents`（增量）。
* **检索**：`search(query, k, max_score)`，自动过滤软删除项。
* **展示**：`show_index(limit, show_metadata)`，同时**重排 `seq=1..N`** 并写入 `_id_map.json`。
* **软删除**：

  * 元数据子串：`delete_documents(contains=...)`
  * 正则：`delete_documents(regex=...)`
  * 按 ID：`delete_documents(ids=[...])`
  * 按序号：`delete_by_index([...], by='seq'|'pos')`
  * 按 metadata（如 source 路径）：`delete_by_metadata('/abs/path/to/file')`
* **数据状态文件** `_id_map.json`：维护 `pos/id/deleted/seq`，结构如下：

```json
{
  "version": 2,
  "items": [
    { "pos": 0, "id": "55b38a...", "deleted": false, "seq": 1 },
    { "pos": 1, "id": "0e2e16...", "deleted": true,  "seq": null },
    { "pos": 2, "id": "990037...", "deleted": false, "seq": 2 }
  ]
}
```

* version：版本信息，没什么用。
* items：存储每一个文本块的信息。
  * `pos`：在json文件中的位置信息。
  * `id`：在向量库的UUID，文本块的根本依据。
  * `deleted`：是否软删除布尔值（每次硬删除都需要重构向量库，开销大且会打乱顺序，故先软删除，到达一定数量再统一删除）。
  * `seq`：滤过`deleted`为`true`的文本块后的顺序，前端依据这个顺序来展示向量库。
* 因此：

  * **对外交互**可优先用 `seq`（用户易理解）；
  * **程序内部**用 `pos` 或 `id`；
  * 软删后不会改变 `pos`，但 `seq` 会在下次 `show_index()` 重新计算。
---

## 快速开始

### 1) 环境

```bash
# 建议
conda env create -f environment.yaml
conda activate dingjia
```

### 2) 模型

* **默认**：`vector_manager.py` 会从 `model/embedding/m3e-base` 读取本地模型（HuggingFace 格式）。
* **自定义**：初始化时传入你的本地路径。

### 3) 一键自测

```bash
python test.py
```

脚本会自动生成 4 篇示例长文（产生多分块）、构建索引、做一系列检索与删除，并在 `./test/_vm_test_rich_*` 下生成：

* `index.faiss / index.pkl`：FAISS 索引；
* `_id_map.json`：可见性与序号映射（见下文）。

---

## 代码用法（最小示例）

```python
from utils.vector_manager import VectorManager

# 1) 初始化（默认 embedding 模型为 m3e）
vm = VectorManager(model_path=None, store_dir="./vector_store")

# 2) 首次构建（支持 .txt/.pdf/.docx）
vm.build_vector("./docs/a.txt")

# 3) 增量加入
vm.add_documents("./docs/b.txt")

# 4) 展示
preview = vm.show_index(limit=10)

# 5) 检索
hits = vm.search("apples", k=5, max_score=1e9)

# 6) 软删除
vm.delete_documents(contains="remove_me")               #按子段删除
vm.delete_documents(regex=r"PATTERN-\d+")               #按正则删除
vm.delete_documents(ids=["<doc_id_1>", "<doc_id_2>"])   #按 doc_id 删除，即_id_map.json中的 id 字段
vm.delete_by_index([1,3], by="seq")                     # 按当前可见列表的第 1、3 条删除，即_id_map.json中的 seq 字段
vm.delete_by_index(42, by="pos")                        # 按底层 FAISS 位置删除，即_id_map.json中的 pos 字段

# 7) 再次展示
vm.show_index(limit=20)
```
---
## 日志

* 默认通过 `utils/logger.py` 输出到控制台并存储在`logs`文件夹。
* 关键功能（构建、增量、检索、展示、删除）均有 INFO 日志。

---

## 常见问题（FAQ）

* **检索没结果？**
  看看 `_id_map.json` 是否把相关分块都标记成 `deleted=true`；软删会被自动过滤。
* **模型加载失败？**
  检查 `model/embedding/m3e-base` 是否存在；或在初始化时传入可用的 `model_path`（HuggingFace 模型名或本地路径）。
* **Windows 路径问题？**
  建议使用原始字符串 `r"C:\path\to\file.txt"` 或 `Path`。
* **测试脚本太慢？**
  打开 `test.py`，调小 `CHUNK_TARGET_CHARS / PER_MARKER_TARGET` 即可。

---


## 统一文件导入

* `/tasks/add` 会统一处理 txt/pdf/docx/csv，内部调用 `VectorManager.ingest_file()` 自动分流。
* 上传后接口立即返回 task_id，可通过 `/tasks/id/{task_id}` 轮询结果；同步 `/kb/add` 已下线。
