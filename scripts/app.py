import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.rag.pipeline import RAGPipeline

# Page config
st.set_page_config(
    page_title="RAG Document Assistant",
    layout="wide"
)

st.title("📄 RAG Document Assistant")
st.write("Ask questions from your indexed PDF documents")

# Initialize RAG pipeline once
@st.cache_resource
def load_rag():
    return RAGPipeline(top_k=5)

rag = load_rag()

# User input
query = st.text_input("Enter your question:")

# Button
if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Retrieving answer..."):
            result = rag.run(query)

        # Show answer
        st.subheader("✅ Answer")
        st.write(result["answer"])

        # Show sources
        st.subheader("📚 Sources")
        for src in set(result["sources"]):
            st.write(f"- {src}")
