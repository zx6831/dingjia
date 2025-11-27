# Dingjia RAG Toolkit

一个围绕 **FAISS + FastAPI** 构建的检索增强（RAG）工具箱，目标是让“知识库构建 → 检索重排 → API 暴露”这一整条链路保持简单、可观察、易于扩展。核心逻辑集中在 `utils/vector_manager.py`、`utils/reranker.py` 与 `main.py`，默认加载本地 `model/embedding/m3e-base/` 作为嵌入模型，并把运行期索引保存在 `vector_store/` 下。

---

## 功能亮点
- **一体化向量库管理**：`VectorManager` 负责文档加载、切块、建库、增量导入、软删/硬删、缓存 `_id_map.json`、`_index_stats.json`、`_source_stats.json`，并暴露 `search/show_index/get_index_snapshot/get_source_statistics` 等调试能力。
- **可调权重的 QA 重排**：`QAPairRerankerA` 同时考虑 query-问题/答案余弦相似度、BM25、实体重合与 MMR 抽样，可按请求调整 `topk/pool_size/lambda_mmr/max_score`，默认直接复用 `VectorManager` 的向量检索结果。
- **API + 任务队列**：`main.py` 提供 `/health`、`/rerank`、`/index`、`/statistic`、`/delete`、`/destroy`、`/tasks/*` 等接口，并通过 `AsyncTaskQueue` 把构建/增量导入落到 `rag_tasks.db`，Task 队列支持状态查询、统计面板与上传文件自动落盘到 `vector_store/uploads/`。
- **内置调试脚本**：`python "test .py"` 一键生成演示语料、校验 `seq` 连续性；`python search.py` 可直接跑一次重排示例。所有测试语料放在 `docs/` 或 `test/` 下，便于清理。

---

## 目录结构（核心）
```
dingjia/
├─ main.py                     # FastAPI 入口，汇聚 VectorManager、Reranker、任务队列
├─ utils/
│  ├─ vector_manager.py        # 向量库管理（建库/增量/软删/缓存/统计）
│  ├─ reranker.py              # Q/A 重排器（MMR + BM25 + 语义特征融合）
│  ├─ taskQueue.py             # sqlite 落库的异步任务队列
│  └─ logger.py                # 统一日志初始化
├─ docs/                       # 示例或测试语料
├─ model/embedding/m3e-base/   # 默认 HuggingFace embedding 模型
├─ vector_store/               # 运行期索引、缓存、uploads/、rag_tasks.db（请勿提交）
├─ search.py                   # CLI 版重排 demo
├─ test .py                    # 烟雾测试
├─ test/                       # 临时语料
├─ api_docs.md                 # REST 接口文档（与 README 同步维护）
├─ environment.yaml            # Conda 环境
├─ requirements.txt            # pip 依赖
└─ README.md                   # 当前说明文档
```
> `vector_store/` 会生成 `_id_map.json`、`_index_stats.json`、`_source_stats.json`、`index.faiss`、`index.pkl`、`rag_tasks.db`、`uploads/*.ext` 等文件，全部视为运行期敏感数据，不要提交到版本库。

---

## 快速上手
1. **准备环境**
   ```bash
   conda env create -f environment.yaml
   conda activate dingjia
   # 或按需使用 requirements.txt
   pip install -r requirements.txt
   ```
2. **配置模型与存储**
   - 默认向量库目录：`VECTOR_STORE_DIR=./vector_store`
   - 默认嵌入模型：`EMBED_MODEL_PATH` 为空时读取 `model/embedding/m3e-base`
   - 根据租户/环境设置上述两个环境变量即可隔离索引与模型
3. **构建或导入语料**
   - CLI：运行 `python test.py`会在 `test/_vm_test_rich_<timestamp>/` 下生成演示语料并执行全流程校验
   - API：参考 `/tasks/build`、`/tasks/add` 上传或引用文件，上传的原始文件会保存到 `vector_store/uploads/<uuid>_<filename>`
   - 如果只需要 CLI 测试，可直接 `VectorManager.ingest_file` / `add_documents`
4. **启动 API**
   ```bash
   python -m uvicorn main:app --reload --host 0.0.0.0 --port 8031
   curl http://localhost:8031/health
   ```
5. **快速验证**
   - `python search.py`：复用当前向量库执行一次重排，输出 top-k 结果
   - `python test.py`：确保 `assert_seq_continuous`、`ensure_min_visible` 等断言均通过
   - `python search.py` 或 `curl /rerank` 前请确认 `vector_store/` 已存在索引

---

## 核心模块
### VectorManager (`utils/vector_manager.py`)
- 支持 `.txt/.pdf/.docx/.csv`（含批量 CSV loader）并使用 `CharacterTextSplitter` 自动切块，默认 chunk_size=1000、overlap=100。
- 默认加载 GPU 版 `HuggingFaceEmbeddings`，也可通过 `EMBED_MODEL_PATH` 指定任意 checkpoint。
- `build_vector` / `load_vector` 处理首次建库与磁盘加载，自动生成 `_id_map.json` 并维持 `seq` 连续。
- `ingest_file` / `add_documents` 追加语料，`delete_by_index` / `delete_by_metadata` / `delete_documents` 提供按 `seq/pos/id/metadata` 的软删能力。
- `refresh_index_cache`、`get_index_snapshot`、`get_source_statistics` 会把索引快照和来源统计写入 JSON，供 `/index`、`/statistic` 零秒响应。
- `drop_vector_store(remove_dir=False)` 负责安全清理索引与元数据；当软删超过 `_hard_delete_threshold` 时可触发硬重建。

### QAPairRerankerA (`utils/reranker.py`)
- 复合得分：`0.55*cos(q,Q) + 0.25*cos(q,A) + 0.15*BM25 + 0.05*实体重合 - penalty(|A|)`。
- 内建 MMR 抽样（`lambda_mmr` 默认 0.7）避免重复回答，可一键调整 `pool_size/topk/max_score`。
- `search_and_rerank` 批量向量化，避免对同一 query 重复编码；在 API/CLI 中可直接调用。

### AsyncTaskQueue (`utils/taskQueue.py`)
- SQLite (`rag_tasks.db`) + 内存队列双写，进程重启后仍可查询历史任务。
- `add_task` 自动识别协程/同步函数并输出 `task_id`；`get_task_status`、`get_statistics`、`get_queue_status` 用于前端面板或排错。
- `/tasks/build`、`/tasks/add` 均走队列，上传文件统一落地 `vector_store/uploads/`，返回的 `status_url` 可直接用于轮询。

### FastAPI Surface (`main.py`)
- 统一暴露健康检查、检索重排、索引快照、来源统计、软删/硬删、任务队列管理等接口。
- `/index` 现在按 `start_seq/end_seq` 截取缓存，`/statistic` 会返回每个来源文件的 `seq` 范围和条目数。
- 所有阻塞操作均封装在 `asyncio.to_thread` 或任务队列中，确保 API 不被长耗时操作拖垮。
- 详细请求/响应体请查阅 `api_docs.md`。

---

## 调试与测试
- `python test.py` 会在隔离目录构造 4 篇长文档，依次执行建库、`add_documents`、多种删除策略、检索前后对比，并触发 `assert_seq_continuous` / `ensure_min_visible`。
- 当你修改检索、删除或 `_id_map` 相关逻辑时，务必新增/更新测试断言，确保烟雾测试仍能发现回归。
- 临时语料放在 `docs/` 或 `test/<timestamp>` 下，实验结束后删除（尤其是 `test/_vm_test_rich_*` 与 `vector_store/uploads`），避免把 FAISS 索引或原始文档提交到 Git。
- `search.py` 提供最小化重排示例，可在调优 Reranker 或检查向量库是否可用时直接运行。

---

## API 快速参考
`api_docs.md` 记录了完整的入参、出参与 curl 示例，以下列出关键接口：
- `GET /health`：探活，返回当前 `store_dir`。
- `POST /rerank`：批量问题检索 + 重排，支持 `topk/pool_size/lambda_mmr/max_score` 调参。
- `GET /index`：通过 `start_seq`、`end_seq`、`metadata` 快速获取缓存快照。
- `GET /statistic`：输出每个源文件的 `seq` 范围与条数，便于前端展示。
- `POST /delete`：支持按 `seq` 或元数据条件软删；`POST /destroy` 可清空索引（可选连同目录）。
- `POST /tasks/build` / `POST /tasks/add`：异步构建/增量导入，`status_url` 用于轮询 `GET /tasks/id/{task_id}`。
- `GET /tasks/task_list`：透出 `total_tasks/status_counts/async_tasks/sync_tasks/recent_tasks/database_size/is_running/current_task` 等统计信息。

---

## 配置与安全
- 环境变量 `VECTOR_STORE_DIR`、`EMBED_MODEL_PATH` 是推荐的多租户/多环境隔离手段；不同环境应拥有独立的 `vector_store/`。
- `vector_store/`、`logs/`、`rag_tasks.db`、`vector_store/uploads/` 属于运行期敏感数据，切勿提交到 Git，也不要泄露上传的原始文件。
- 上传语料时统一落地 `vector_store/uploads/<uuid>_<filename>`，若为临时测试，请在验证完成后清理对应文件与生成的索引。
- 当 `_id_map.json` 大面积标记 `deleted` 时，可调用 `/destroy?remove_dir=true` 或重新运行构建任务来重建索引。
- Windows 环境建议统一使用 `Path`/`os.path` 组合路径（所有 loader 与 API 也是这样处理路径的）。

---

## FAQ / Roadmap
- **没有召回？** 检查 `_id_map.json` 是否全部被软删，或确认 `/rerank` 的 `max_score` 是否设置过小。
- **模型加载失败？** 确保 `model/embedding/m3e-base` 已下载；若改用其他 checkpoint，请设置 `EMBED_MODEL_PATH=/path/to/model`。
- **任务队列报数据库不存在？** 调用 `/tasks/*` 前确保 `vector_store/` 可写，本项目会在任务入队时自动执行 `_ensure_task_queue_db()` 重建 `rag_tasks.db`。
- **路径乱码？** API 和 CLI 均通过 `Path` 处理 Windows 路径；若自行传参请避免手写反斜杠。

Roadmap（节选）：
- [ ] 自适应的 MMR 抽样策略（根据 `pool_size` 自动调节）。
- [ ] 对 question/answer embedding 做缓存，进一步降低重排延迟。
- [ ] API 增加鉴权与按 `VECTOR_STORE_DIR`、`EMBED_MODEL_PATH` 的租户隔离策略。
