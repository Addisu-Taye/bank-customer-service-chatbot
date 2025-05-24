from utils.doc_loader import load_documents_from_folder, save_vectors_to_disk
from utils.chat_engine import ChatEngine

DOCUMENT_FOLDER = "./data"  # your docs folder
VECTOR_PATH = "vectors/bank_data.pkl"  # updated path


class Embedder:
    def embed_documents(self, docs):
        # dummy example, replace with real embedding logic
        return [f"vector_of_{doc}" for doc in docs]

def main():
    print("🔍 Loading documents from folder...")
    documents = load_documents_from_folder(DOCUMENT_FOLDER)
    print(f"✅ Loaded {len(documents)} documents. Embedding...")

    embedder = Embedder()
    save_vectors_to_disk(documents, embedder, VECTOR_PATH)

    # Initialize ChatEngine with vector path and embedder
    engine = ChatEngine(VECTOR_PATH, embedder)
    print("✅ Vectors saved and ChatEngine initialized.")

if __name__ == "__main__":
    main()
