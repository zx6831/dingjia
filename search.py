from utils.vector_manager import VectorManager
from utils.reranker import QAPairRerankerA

vm = VectorManager(model_path=None, store_dir="./vector_store")
# 1) 直接让重排器内部调用 vm.search(query, k=pool_size) 取候选池
reranker = QAPairRerankerA(vm, topk=8, pool_size=60, lambda_mmr=0.7)
top_ctx = reranker.search_and_rerank("项目的有什么优势，解决什么行业痛点")
print(top_ctx)
# top_ctx[i] -> {"doc": <Document>, "base_score": float, "fuse_score": float, "q_text": str, "a_text": str}






