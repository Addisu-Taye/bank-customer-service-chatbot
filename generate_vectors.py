import os
from utils.doc_loader import load_documents_from_folder
from utils.vector_builder import save_vector_db
from utils.chat_engine import ChatEngine
from langchain_huggingface import HuggingFaceEmbeddings # Import HuggingFaceEmbeddings for the ChatEngine

# Define the folder containing bank documents and where vectors will be saved
DOCUMENT_FOLDER = "./bank_docs"
VECTOR_PATH = "./vectors"

def main():
    # Create the document folder if it doesn't exist
    if not os.path.exists(DOCUMENT_FOLDER):
        os.makedirs(DOCUMENT_FOLDER)
        print(f"Created document folder: {DOCUMENT_FOLDER}. Please place your bank documents (.docx) here.")
        print("Exiting as no documents are available for embedding.")
        return

    print("🔍 Loading documents from folder...")
    # Load documents from the specified folder
    documents = load_documents_from_folder(DOCUMENT_FOLDER)

    if not documents:
        print("No documents found in the folder. Please add .docx files to './bank_docs'.")
        return

    print(f"✅ Loaded {len(documents)} documents. Embedding and saving vectors...")

    # Initialize the actual embedder model that ChatEngine will use for queries
    # Using a financial-specific embedding model for better domain understanding
    embedder = HuggingFaceEmbeddings(model_name="FinLang/finance-embeddings-investopedia")
    print("Using 'FinLang/finance-embeddings-investopedia' for creating embeddings.")

    # Save the vector database to disk using the dedicated function
    save_vector_db(documents, VECTOR_PATH, embedder)

    # Initialize ChatEngine with the path to the saved vectors and the embedder
    # The ChatEngine will load the FAISS vector store and use the embedder for queries
    engine = ChatEngine(VECTOR_PATH, embedder)
    print("✅ Vectors saved and ChatEngine initialized. You can now use the ChatEngine to query your documents.")

    # Example of how to use the chat engine (optional)
    # query = "What are the eligibility requirements for a GlobalTrust Personal Loan?"
    # response = engine.get_response(query)
    # print(f"\nQuery: {query}")
    # print(f"Response: {response}")


if __name__ == "__main__":
    main()

