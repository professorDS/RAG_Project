from typing import List
from click import prompt
from langchain_core.documents import Document
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from src.utils.logger import logger


# ================= CONFIG =================
MODEL_NAME = "google/flan-t5-base"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
# =========================================


class AnswerGenerator:
    """
    Generates answers using an open-source LLM.
    """

    def __init__(self):
        logger.info("Loading open-source LLM...")

        text2text_pipeline = pipeline(
            task="text2text-generation",
            model=MODEL_NAME,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE
        )

        self.llm = HuggingFacePipeline(pipeline=text2text_pipeline)

        logger.info("Open-source LLM loaded successfully")

    def _build_prompt(self, query: str, documents) -> str:
        context = "\n\n".join(
            f"- {doc.page_content.strip()}"
            for doc in documents
            if doc.page_content.strip()
        )

        prompt = f"""
        You are a domain-aware assistant analyzing company documents.

        Use ONLY the information from the context below.
        If the answer is not explicitly present, say "The information is not available in the documents."

        Context:
        {context}

        Task:
        Answer the question clearly and concisely.
        If possible, structure the answer using bullet points.

        Question:
        {query}

        Answer:
        """
        return prompt.strip()


    def generate_answer(self, query: str, documents: List[Document]):
        """
        Generate answer from query and retrieved documents.
        """
        logger.info("Generating answer using open-source LLM")

        prompt = self._build_prompt(query, documents)

        response = self.llm.invoke(prompt)

        return response
