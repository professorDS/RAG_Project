# RAG Project – End-to-End Retrieval Augmented Generation

A comprehensive **Retrieval Augmented Generation (RAG)** system designed to process documents and provide intelligent retrieval-based responses. This project implements a production-ready pipeline for document ingestion, processing, embedding, and semantic search.

## 📋 Overview

This RAG system extracts information from various document formats (PDFs, images, announcements), processes them through OCR and NLP pipelines, and enables semantic search using vector embeddings and FAISS indexing.

**Key Features:**
- 📄 Multi-format document support (PDF, images, CSV)
- 🔍 OCR-powered text extraction
- 🧩 Intelligent text chunking and preprocessing
- 🤖 Sentence transformer embeddings
- ⚡ FAISS vector database for fast retrieval
- 💬 LangChain-powered RAG queries
- 🎯 Interactive web interface with Streamlit

## 🏗️ Project Structure

```
RAG_project/
├── data/
│   ├── announcment_data.csv       # Announcement data source
│   ├── pdfs/                       # Raw PDF documents
│   ├── processed_text/             # OCR-extracted text files
│   └── index/
│       └── index.faiss             # FAISS vector index
├── src/
│   ├── config/                     # Configuration management
│   ├── data_loader/                # Document loading utilities
│   ├── embeddings/                 # Embedding generation & FAISS indexing
│   ├── preprocessing/              # Text processing & chunking
│   ├── rag/                        # RAG retrieval logic
│   └── utils/                      # Helper utilities
├── scripts/
│   ├── app.py                      # Main Streamlit application
│   └── run_query.py                # Query execution script
├── tests/                          # Unit tests
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── ARCHITECTURE.md                 # Detailed architecture documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda package manager
- rag_env virtual environment

### Installation

1. **Activate the virtual environment:**
   ```bash
   # Windows
   rag_env\Scripts\activate
   
   # Linux/Mac
   source rag_env/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the project:**
   - Edit `src/config/config.ini` with your settings
   - Place PDF documents in `data/pdfs/`
   - Place CSV data in `data/` directory

### Building the FAISS Index

Process documents and build the vector index:

```bash
python -m src.embeddings.build_faiss_index
```

This will:
- Extract text from PDFs using OCR
- Chunk documents into manageable segments
- Generate embeddings using sentence-transformers
- Build and save the FAISS index

### Running Queries

#### Option 1: Web Interface (Streamlit)
```bash
streamlit run scripts/app.py
```

#### Option 2: Command Line
```bash
python scripts/run_query.py "Your query here"
```

## 🛠️ Core Components

### 1. Data Loading (`src/data_loader/`)
Handles document ingestion from multiple sources:
- PDF extraction with PyMuPDF and EasyOCR
- CSV data parsing
- Text preprocessing and normalization

### 2. Preprocessing (`src/preprocessing/`)
Transforms raw text into processable chunks:
- Text cleaning and tokenization
- Semantic chunk segmentation
- Metadata preservation

### 3. Embeddings (`src/embeddings/`)
Generates vector representations:
- Sentence-transformers for semantic embeddings
- Batch processing for efficiency
- FAISS index creation and management

### 4. RAG Module (`src/rag/`)
Core retrieval and generation logic:
- Vector similarity search
- Document retrieval ranking
- LangChain integration for prompt engineering

### 5. Utilities (`src/utils/`)
Helper functions for logging, caching, and data handling

## 📦 Dependencies

Key libraries used:
- **LangChain**: RAG framework
- **Sentence-Transformers**: Embedding generation
- **FAISS**: Vector similarity search
- **EasyOCR**: Optical character recognition
- **Streamlit**: Web interface
- **Transformers**: NLP models
- **PyTorch**: Deep learning backend

See `requirements.txt` for complete list.

## 🧪 Testing

Run unit tests to verify components:

```bash
python -m pytest tests/ -v
```

Available tests:
- `test_downloader.py` - Document loading functionality
- `test_embeddings.py` - Embedding generation and indexing

## 📖 Usage Examples

### Query with Web Interface
1. Start the Streamlit app: `streamlit run scripts/app.py`
2. Enter your query in the text field
3. View retrieved documents and AI-generated responses

### Programmatic Usage
```python
from src.rag.retriever import RAGRetriever

# Initialize retriever
retriever = RAGRetriever(index_path="data/index/index.faiss")

# Retrieve documents
results = retriever.retrieve("Your question here", top_k=5)

# Process results
for doc in results:
    print(doc.content)
```

## ⚙️ Configuration

Edit `src/config/config.ini` to customize:
- Embedding model selection
- Chunk size and overlap
- FAISS index parameters
- LLM settings (if using external models)

## 🔄 Workflow

```
Documents → OCR/Text Extraction → Preprocessing → Chunking 
     ↓
Embeddings Generation → FAISS Indexing
     ↓
Query → Embedding → Vector Search → Retrieval → Response Generation
```

## 📝 Data Format

### Processed Text Files
Location: `data/processed_text/`
Format: Plain text with metadata in filename
Example: `TICKER_TIMESTAMP_DESCRIPTION_HASH.txt`

### Index
Location: `data/index/`
Format: FAISS binary index file
- Loaded automatically by retriever
- Can be rebuilt without losing source documents

## 🚨 Troubleshooting

**Issue: FAISS index not found**
- Run `python -m src.embeddings.build_faiss_index` to generate index

**Issue: OCR extraction issues**
- Ensure PDFs are not corrupted
- Check EasyOCR language settings in config
- Verify document image quality

**Issue: Low retrieval quality**
- Check embedding model compatibility
- Verify chunk size settings
- Review query formulation

## 📚 Additional Resources

- See `ARCHITECTURE.md` for detailed technical documentation
- Check `tests/` for usage examples
- Review individual module docstrings for API details

## 👥 Contributors

Built with LangChain, FAISS, and Sentence Transformers ecosystems.

## 📄 License

[Add your license information here]

---

**Last Updated:** February 2026
**Status:** Active Development

