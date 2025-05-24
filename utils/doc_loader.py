import os
from langchain_unstructured import UnstructuredLoader  # ✅ updated import

def load_documents(source_folder):
    documents = []

    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        try:
            print(f"📄 Loading {filename}")
            loader = UnstructuredLoader(file_path)  # ✅ updated loader
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"⚠️ Failed to load {filename}: {e}")

    return documents
