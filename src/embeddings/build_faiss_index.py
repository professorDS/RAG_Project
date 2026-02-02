from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.logger import logger

# Paths - Use absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEXT_FOLDER = PROJECT_ROOT / "data" / "processed_text"
INDEX_FOLDER = PROJECT_ROOT / "data" / "index"
INDEX_FOLDER.mkdir(parents=True, exist_ok=True)

# Embedding model (fast & good quality)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_text_files():
    documents = []

    for txt_file in TEXT_FOLDER.glob("*.txt"):
        try:
            text = txt_file.read_text(encoding="utf-8")

            documents.append({
                "text": text,
                "source": txt_file.name
            })

        except Exception as e:
            logger.error(f"Failed to read {txt_file.name}: {e}")

    logger.info(f"Loaded {len(documents)} text files")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    texts = []
    metadatas = []

    for doc in documents:
        chunks = splitter.split_text(doc["text"])

        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({"source": doc["source"]})

    logger.info(f"Created {len(texts)} text chunks")
    return texts, metadatas


def build_faiss_index():
    logger.info("Starting FAISS index build")

    docs = load_text_files()
    texts, metadatas = split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    vectorstore.save_local(str(INDEX_FOLDER))

    logger.info("FAISS index created and saved successfully")


if __name__ == "__main__":
    build_faiss_index()
