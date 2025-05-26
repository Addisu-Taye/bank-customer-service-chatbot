import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from utils.doc_loader import load_documents
import logging # Import logging for better messages

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_vector_db(source_folder, save_path):
    documents = load_documents(source_folder)
    logging.info(f"Loaded {len(documents)} documents from {source_folder}")

    # REPLACED: embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Using a financial-specific embedding model for better domain understanding
    embedder = HuggingFaceEmbeddings(model_name="FinLang/finance-embeddings-investopedia")
    logging.info("Using FinLang/finance-embeddings-investopedia for creating embeddings.")

    # Create the vector store from documents and the financial embedder
    vectorstore = FAISS.from_documents(documents, embedder)
    logging.info("FAISS vector store created from documents.")

    # Create the directory if it doesn't exist and save the vector DB
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        logging.info(f"Created directory: {save_path}")

    vectorstore.save_local(save_path)
    logging.info(f"✅ Vector DB saved to {save_path}")