import os
from langchain_community.vectorstores import FAISS
# Note: HuggingFaceEmbeddings is now passed directly from generate_vectors.py or initialized there.
# from langchain_huggingface import HuggingFaceEmbeddings
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# This function now directly accepts the loaded documents and the embedder instance.
def save_vector_db(documents, save_path, embedder):
    """
    Creates and saves a FAISS vector database from a list of documents.

    Args:
        documents (list): A list of Langchain Document objects to embed.
        save_path (str): The directory path where the FAISS vector store will be saved.
        embedder: An initialized embedding model (e.g., HuggingFaceEmbeddings instance).
    """
    if not documents:
        logging.warning("No documents provided to create the vector database. Skipping save operation.")
        return

    logging.info(f"Creating FAISS vector store for {len(documents)} documents.")

    try:
        # Create the vector store from documents and the provided embedder
        vectorstore = FAISS.from_documents(documents, embedder)
        logging.info("FAISS vector store created successfully.")

        # Create the directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            logging.info(f"Created directory: {save_path}")

        # Save the vector DB locally
        vectorstore.save_local(save_path)
        logging.info(f"✅ Vector DB saved to {save_path}")

    except Exception as e:
        logging.error(f"Error saving vector database: {e}")

# This function is now used by doc_loader.py to load documents.
# It is kept here as a helper for clarity, but the primary loading logic is in doc_loader.py
# (It was originally here in the prompt's utils/vector_builder.py, so keeping it
# might be confusing. Let's remove it from here and ensure doc_loader.py handles it.)
# Rationale: The original vector_builder.py had 'load_documents(source_folder)'.
# My doc_loader.py now has 'load_documents_from_folder'.
# To avoid redundancy and circular imports, doc_loader.py will be responsible for loading.
# So, no need for load_documents here.
