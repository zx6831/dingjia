import time
from utils.logger import get_logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

logger = get_logger("translation_agent")

def main():
    try:
        logger.info("程序启动 —— LangChain + FAISS + m3e 翻译智能体")

        # 1️⃣ 文档加载
        start_time = time.time()
        loader = TextLoader("./docs/test.txt", encoding="utf-8")
        documents = loader.load()
        logger.info(f"文档加载成功，共 {len(documents)} 条记录")

        # 2️⃣ 文本切分
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)
        logger.info(f"文档切分完成，共 {len(docs)} 个分块")

        # 3️⃣ 向量化
        embeddings = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
        db = FAISS.from_documents(docs, embeddings)
        elapsed = time.time() - start_time
        logger.info(f"向量库构建完成，耗时 {elapsed:.2f} 秒")

        # 4️⃣ 查询示例
        query = "行业痛点"
        logger.info(f"开始查询：{query}")
        start_query = time.time()
        results = db.similarity_search(query, k=3)
        query_time = time.time() - start_query
        logger.info(f"检索完成，耗时 {query_time:.2f} 秒，共返回 {len(results)} 条结果")

        for i, doc in enumerate(results, 1):
            snippet = doc.page_content[:150].replace("\n", " ")
            logger.info(f"[Top {i}] {snippet}...")

        logger.info("✅ 程序执行完毕")

    except Exception as e:
        logger.error(f"❌ 运行异常：{e}", exc_info=True)

if __name__ == "__main__":
    main()
