import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from utils.doc_loader import load_documents

def save_vector_db(source_folder, save_path):
    documents = load_documents(source_folder)
    print(f"Loaded {len(documents)} documents")

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embedder)

    # Create the directory if it doesn't exist and save the vector DB
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    vectorstore.save_local(save_path)
    print(f"✅ Vector DB saved to {save_path}")
