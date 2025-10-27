import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from utils.logger import get_logger

logger = get_logger("vector_manager")

class VectorManager:
    """统一管理 FAISS 向量库的构建、加载、更新与检索"""

    def __init__(self, model_name="moka-ai/m3e-base", store_dir="vector_store"):
        self.store_dir = store_dir
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.db = None

        os.makedirs(store_dir, exist_ok=True)

    def build_index(self, doc_path):
        """首次构建向量库"""
        logger.info(f"开始构建向量库，使用模型：{self.model_name}")

        loader = TextLoader(doc_path, encoding="utf-8")
        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)
        logger.info(f"文档加载完成，共 {len(docs)} 个分块")

        self.db = FAISS.from_documents(docs, self.embeddings)
        self.db.save_local(self.store_dir)
        logger.info(f"向量库已构建并保存到：{self.store_dir}")

    def load_index(self):
        """加载已有向量库"""
        logger.info("加载已保存的向量库...")
        self.db = FAISS.load_local(
            self.store_dir, self.embeddings, allow_dangerous_deserialization=True
        )
        logger.info("向量库加载完成。")

    def add_documents(self, new_doc_path):
        """增量添加新文档"""
        if not self.db:
            self.load_index()

        loader = TextLoader(new_doc_path, encoding="utf-8")
        new_docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(new_docs)
        self.db.add_documents(chunks)
        self.db.save_local(self.store_dir)
        logger.info(f"新增文档已加入向量库，共新增 {len(chunks)} 段。")

    def search(self, query, k=3):
        """查询相似内容"""
        if not self.db:
            self.load_index()

        logger.info(f"执行检索：{query}")
        results = self.db.similarity_search(query, k=k)
        logger.info(f"检索完成，返回 {len(results)} 条结果。")
        return results
