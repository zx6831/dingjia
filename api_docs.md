# RAG API Documentation

所有接口默认运行在 `http://localhost:8031`，若经由内网穿透或其他域名暴露，只需替换 Base URL。当前版本未做鉴权，请确保部署环境受控或在网关层补充认证。

## 通用说明
- 默认请求头：`Content-Type: application/json`。`/tasks/build`、`/tasks/add` 由于需要上传文件，请使用 `multipart/form-data`。
- FastAPI 抛出的异常会返回 `{"detail": "...错误信息..."}`，可结合 HTTP 状态码提示用户。
- 所有接口均假设向量库已经构建；若 `vector_store/` 不存在会返回 404/500。
- 样例中的 `\` 用于换行，实际命令行可根据需要合并为一行。

---

## 1. GET `/health`
- **用途**：探活，确认服务可达且向量库目录可访问。
- **请求参数**：无
- **返回字段**
  - `ok`: `true`
  - `store_dir`: 当前实例加载的向量库目录
- **示例**
  ```bash
  curl -X GET http://localhost:8031/health
  ```

---

## 2. POST `/rerank`
- **用途**：对问题集合执行检索+重排，返回英文原文与中文译文（默认取每个问题的 top1）。
- **Query 参数**
  - `topk` *(int, 默认 1, 1-50)*：最终返回的条数。
  - `pool_size` *(int, 默认 5, 1-500)*：初始召回的候选数量，必须 `>= topk`。
  - `lambda_mmr` *(float, 默认 0.7, 0-1)*：MMR 去重权重，越大越强调多样性。
  - `max_score` *(float, 默认 100)*：召回的最大距离阈值，可用于过滤低置信度结果。
- **Body**
  ```json
  {
    "query": ["车辆质保期多长？", "如何联系客服？"]
  }
  ```
- **返回字段 (`SearchResponse`)**
  - `original_text`: 与 query 顺序一致的英文原文数组（来自 metadata.english）
  - `translated_text`: 对应的中文译文数组（来自 metadata.chinese）
- **示例**
  ```bash
  curl -X POST 'http://localhost:8031/rerank?topk=1&pool_size=5&lambda_mmr=0.7&max_score=100' \
    -H 'Content-Type: application/json' \
    -d '{"query": ["车辆质保期多长？","如何联系客服？"]}'
  ```

---

## 3. GET `/index`
- **用途**：基于缓存的 `_index_stats.json` 预览向量库快照，供删除面板或运营后台使用。
- **Query 参数**
  - `start_seq` *(int, 默认 1)*：起始序号（包含）。
  - `end_seq` *(int, 默认 10)*：结束序号（包含），必须 `>= start_seq`。传较大值可以一次拉取多条。
  - `metadata` *(bool, 默认 true)*：是否在每条记录中附带 metadata。
- **返回字段**
  - `count`: 本次返回的条数。
  - `items`: `seq/pos/id/preview/metadata` 组成的数组，顺序按 `seq` 升序。
- **示例**
  ```bash
  curl -X GET 'http://localhost:8031/index?start_seq=1&end_seq=50&metadata=true'
  ```

---

## 4. GET `/statistic`
- **用途**：查看每个源文件在向量库中的 `seq` 范围与条目数量，便于构造展示面板。
- **返回数组 (`StatisticItem`)**
  - `source`: 绝对路径或上传文件保存路径。
  - `display_name`: 供 UI 展示的文件名。
  - `start_seq` 与 `end_seq`: 该文件覆盖的 `seq` 范围。
  - `count`: 条目数量。
- **示例**
  ```bash
  curl -X GET http://localhost:8031/statistic
  ```

---

## 5. POST `/destroy`
- **用途**：清空当前向量库（FAISS + `_id_map.json`）；可选地删除整个目录。
- **Query 参数**
  - `remove_dir` *(bool, 默认 false)*：`true` 时直接删除整个 `store_dir`；否则仅清空文件。
- **返回字段**
  - `ok`: 恒为 `true`
  - `remove_dir`: 是否同时删除目录
  - `deleted_entries`: 本次清空的条目数
  - `store_dir`: 被处理的目录
- **示例**
  ```bash
  curl -X POST 'http://localhost:8031/destroy?remove_dir=false'
  ```

---

## 6. POST `/delete`
- **用途**：基于 `seq` 或 metadata 执行软删，最终变更会体现在 `_id_map.json`。
- **Body (`DeleteBody`)**
  - `indexes` *(int[] 可选)*：按 `seq` 删除，`seq` 为当前可见列表的 1-based 序号。
  - `targets` *(string[] 可选)*：和 `keywords` 组合，按 metadata 字段值匹配。
  - `keywords` *(string, 默认 `"source"`)*：需要匹配的 metadata 字段，可取 `source/english/chinese/type/row` 等。
  - `fuzzy` *(bool, 默认 false)*：是否使用子串模糊匹配。
  - `ignore_case` *(bool, 默认 false)*：是否忽略大小写。
  > `indexes` 与 `targets` 至少提供一个；若两者同时存在会先执行按 `seq` 的删除。
- **返回字段**
  - `ok`: `true`
  - `seq`: （可选）按 `seq` 删除时的详细结果，包含 `requested_seqs/resolved_pos/not_found_seqs/deleted`
  - `metadata`: （可选）按 metadata 删除时返回 `deleted_docs_name/deleted`
- **示例（按 seq 删除 1、3 并按 source 模糊删除）**
  ```bash
  curl -X POST http://localhost:8031/delete \
    -H 'Content-Type: application/json' \
    -d '{
          "indexes": [1,3],
          "targets": ["nejm_train_en2zh.csv"],
          "keywords": "source",
          "fuzzy": true,
          "ignore_case": true
        }'
  ```

---

## 7. POST `/tasks/build`
- **用途**：将构建向量库的操作放入任务队列，避免阻塞 HTTP 请求。
- **请求方式**：`multipart/form-data`
- **Form 字段**
  - `reinit` *(bool, 默认 false)*：若已有向量库，需要显式传 `true` 才允许覆盖。
  - `file` *(可选)*：直接上传语料（优先使用）。服务器会保存到 `vector_store/uploads/<uuid>_<filename>`。
  - `doc_path` *(可选)*：服务器已存在的文件路径。`file` 与 `doc_path` 至少提供一个。
- **返回字段**
  - `ok`: `true` 表示任务成功入队
  - `task_id`: 后续查询所需的 ID
  - `status_url`: 当前实现返回 `/tasks/{task_id}`；请在前端拼接 Base URL，或直接改用 `/tasks/id/{task_id}`
- **示例**
  ```bash
  curl -X POST http://localhost:8031/tasks/build \
    -H 'accept: application/json' \
    -F reinit=true \
    -F file=@docs/nejm_train_en2zh.csv
  ```

---

## 8. POST `/tasks/add`
- **用途**：异步将新增文件追加到现有向量库，字段与 `/tasks/build` 基本一致（无 `reinit`）。
- **请求方式**：`multipart/form-data`
- **Form 字段**
  - `file` *(可选)*
  - `doc_path` *(可选)*
- **返回字段**
  - `ok`, `task_id`
  - `status_url`: `/tasks/id/{task_id}`
- **示例**
  ```bash
  curl -X POST http://localhost:8031/tasks/add \
    -F doc_path="C:\path\to\docs\nejm_test_en2zh.csv"
  ```

---

## 9. GET `/tasks/id/{task_id}`
- **用途**：查询某个异步任务的实时状态/结果。
- **返回字段**（节选）
  - `task_id`
  - `function_name`: `_task_build_vector` / `_task_add_documents`
  - `kwargs`: 队列执行时的参数
  - `status`: `pending/processing/finished/error`
  - `is_async`: 是否异步函数
  - `created_at/started_at/finished_at`
  - `result`: 由任务函数返回，如 `{ "ok": true, "built_from": "...", "count": 123 }`
  - `error_message`、`traceback`：仅在失败时返回
- **示例**
  ```bash
  curl -X GET http://localhost:8031/tasks/id/246b1470-b67b-4c71-af5d-eb783d9eb90e
  ```

---

## 10. GET `/tasks/task_list`
- **用途**：获取任务队列总体状态，便于展示“后台正在处理 N 个任务”。
- **返回字段**
  - `total_tasks`
  - `status_counts`: `{"pending":0,"processing":0,"finished":0,"error":0}`
  - `async_tasks` / `sync_tasks`
  - `recent_tasks`: 最近 5 条任务的 `task_id/status/is_async/created_at/finished_at`
  - `database_size`: `rag_tasks.db` 文件大小（字节）
  - `is_running`: 队列线程是否正在运行
  - `current_task`: 目前正在执行的任务 ID（无则为 `null`）
- **示例**
  ```bash
  curl -X GET http://localhost:8031/tasks/task_list
  ```

---

## 常用场景与建议
1. **批量检索**：`/rerank` 支持一次提交多个问题，返回数组与输入顺序一一对应；前端可直接 zip 成对话列表。
2. **删除前先拉取快照**：调用 `/index`（可配合 `metadata=false` 提速）确认 `seq`，再执行 `/delete`，根据响应中的 `deleted` 数量告知用户。
3. **异步任务轮询**：`/tasks/build` 与 `/tasks/add` 返回 `task_id` 后，建议 2~5 秒轮询 `/tasks/id/{task_id}`；任务完成后如需刷新索引视图，可以再请求 `/index` 或 `/statistic`。
4. **大批量索引预览**：`/index` 走缓存，推荐按 200~500 条一次分页拉取；如需全量导出，请分批请求，避免一次性拉取导致响应超时。
5. **上传文件命名**：后端会生成 `<uuid>_<filename>` 以避免冲突，前端展示时可截取 `_` 之后的原始文件名。
