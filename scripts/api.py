from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from functools import lru_cache
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.pipeline import RAGPipeline
from src.utils.logger import logger


# Request/Response models
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


# Initialize FastAPI app
app = FastAPI(
    title="RAG API",
    description="Query your indexed documents using RAG (Retrieval-Augmented Generation)",
    version="1.0.0"
)

# Enable CORS for browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Cache RAG pipeline to avoid reloading on each request
@lru_cache(maxsize=1)
def get_rag() -> RAGPipeline:
    """Initialize and cache RAG pipeline"""
    logger.info("Initializing RAG Pipeline")
    return RAGPipeline(top_k=5)


@app.get("/")
def root():
    """Root endpoint with API info"""
    return {
        "message": "RAG API is running",
        "docs": "/docs",
        "query_endpoint": "/query"
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    Query the RAG pipeline with a question
    
    Args:
        request: QueryRequest with 'query' field
    
    Returns:
        QueryResponse with 'answer' and 'sources'
    """
    try:
        if not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        logger.info(f"Received query: {request.query}")
        
        rag = get_rag()
        result = rag.run(request.query)
        
        return QueryResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", [])
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)