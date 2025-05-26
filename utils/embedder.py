#embedder.py
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedder():
    # REPLACED: return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Using a financial-specific embedding model for better domain understanding
    return HuggingFaceEmbeddings(model_name="FinLang/finance-embeddings-investopedia")