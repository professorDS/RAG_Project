from src.rag.pipeline import RAGPipeline
from src.utils.logger import logger


def main():
    logger.info("Starting RAG Query Interface")

    rag = RAGPipeline(top_k=5)

    print("\nRAG system is ready.")
    print("Type your question below (type 'exit' to quit):\n")

    while True:
        query = input("give query")

        if query.lower() in ["exit", "quit"]:
            print("Exiting RAG system.")
            break

        if not query.strip():
            print("Please enter a valid question.")
            continue

        result = rag.run(query)

        print("\nAnswer:\n")
        print(result["answer"])

        print("\nSources:\n")
        for src in set(result["sources"]):
            print(f"- {src}")

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
