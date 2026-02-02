from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.logger import logger

# ================= CONFIG =================
# Use absolute path to work from any directory
INDEX_PATH = str(Path(__file__).parent.parent.parent / "data" / "index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# ==========================================

class FAISSRetriever:
    """
    Handles vector-based retrieval from FAISS index.
    """

    def __init__(self, k: int = 8):
        self.k = k
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        self.vectorstore = self._load_index()

    def _load_index(self):
        """
        Load FAISS index from disk.
        """
        logger.info("Loading FAISS index...")
        vectorstore = FAISS.load_local(
            INDEX_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS index loaded successfully")
        return vectorstore

    def retrieve(self, query: str):
        """
        Retrieve top-k relevant documents for a query.
        """
        logger.info(f"Retrieving documents for query: {query}")

        docs = self.vectorstore.similarity_search(
            query,
            k=self.k
        )

        return docs
