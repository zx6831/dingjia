# Repository Guidelines

## Project Structure & Module Organization
Core service code lives in `main.py`, the FastAPI surface wiring `VectorManager`, `QAPairRerankerA`, and `AsyncTaskQueue`. `utils/vector_manager.py` handles ingestion, chunking, and FAISS persistence, while `utils/reranker.py`, `utils/taskQueue.py`, and `utils/logger.py` supply supporting logic. Source docs stay in `docs/`, the default `m3e-base` checkpoint in `model/embedding/`, and runtime indexes plus `_id_map.json` in `vector_store/`. `test.py` and `test/` host disposable fixtures.

## Build, Test & Development Commands
Create the runtime env with `conda env create -f emvironment.yaml && conda activate dingjia`. Run `python test.py` to generate long-form docs, build/add/search/delete vectors, and verify `seq` continuity. Start the API locally via `python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000` (it auto-loads `vector_store/`). Use `python search.py` for quick rerank sanity checks against the current store.

## Coding Style & Naming Conventions
Adhere to the existing Python 3.11 style: 4-space indents, `snake_case` functions, `PascalCase` classes, constants in `UPPER_SNAKE`. Keep type hints and one-line docstrings that describe side effects (see `utils/vector_manager.py`). Instantiate loggers with `utils.logger.get_logger()` and favor `os.path` helpers or `Path` over manual string concatenation.

## Testing Guidelines
When changing retrieval or deletion flows, extend the assertions already in `test.py` (`assert_seq_continuous`, `ensure_min_visible`) or add nearby helpers so regressions show up in the smoke run. Place any synthetic corpora under `docs/` or timestamped folders inside `test/` and clean them afterward. For API work, exercise `/health`, `/search_raw`, `/rerank`, and `/kb/index` and capture the curl commands plus outputs.

## Commit & Pull Request Guidelines
With no visible history to mirror, default to Conventional Commit prefixes (`feat`, `fix`, `refactor`, `test`, `chore`). Keep commits focused, and mention affected modules plus log/test evidence in the body. Pull requests should describe the problem, the new behavior, migration or config steps (e.g., new env vars, store dirs), and include manual test commands or screenshots of API responses.

## Security & Configuration Tips
Treat everything inside `vector_store/`, `logs/`, and `rag_tasks.db` as sensitive: never commit raw uploads or task payloads. Honor `VECTOR_STORE_DIR` and `EMBED_MODEL_PATH` env vars to isolate tenants, and purge `test/_vm_test_rich_*` folders after experiments so stale FAISS indexes are not accidentally shipped.
