from utils.vector_manager import VectorManager
from utils.logger import get_logger

logger = get_logger("search_agent")

if __name__ == "__main__":
    vm = VectorManager(model_name="moka-ai/m3e-base")

    query = "行业痛点"
    results = vm.search(query, k=3)

    logger.info("查询结果：")
    for i, doc in enumerate(results, 1):
        logger.info(f"[{i}] {doc.page_content[:200]}...")






