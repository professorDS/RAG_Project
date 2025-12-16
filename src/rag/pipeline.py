from src.rag.retriever import FAISSRetriever
from src.rag.generator import AnswerGenerator
from src.utils.logger import logger


class RAGPipeline:
    """
    Orchestrates the full RAG flow:
    Query -> Retrieval -> Generation
    """

    def __init__(self, top_k: int = 5):
        logger.info("Initializing RAG Pipeline")

        self.retriever = FAISSRetriever(k=top_k)
        self.generator = AnswerGenerator()

        logger.info("RAG Pipeline initialized successfully")

    def run(self, query: str):
        """
        Run full RAG pipeline for a query.
        """
        logger.info(f"Running RAG pipeline for query: {query}")

        # Step 1: Retrieve relevant documents
        documents = self.retriever.retrieve(query)

        # Step 2: Generate answer using retrieved docs
        answer = self.generator.generate_answer(query, documents)

        return {
            "query": query,
            "answer": answer,
            "sources": [doc.metadata.get("source") for doc in documents]
        }
