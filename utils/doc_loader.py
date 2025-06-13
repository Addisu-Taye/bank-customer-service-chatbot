import os
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Import the text splitter
from langchain_core.documents import Document
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_documents_from_folder(folder_path):
    """
    Loads all .docx documents from a specified folder and splits them into chunks.

    Args:
        folder_path (str): The path to the folder containing the documents.

    Returns:
        list: A list of Langchain Document objects (chunks).
    """
    all_documents = []
    if not os.path.exists(folder_path):
        logging.warning(f"Document folder not found: {folder_path}")
        return all_documents

    logging.info(f"Scanning folder: {folder_path} for .docx documents...")
    try:
        # Use DirectoryLoader to load all .docx files
        # This will load each .docx file as one large Document object initially
        loader = DirectoryLoader(
            folder_path,
            glob="**/*.docx",
            loader_cls=UnstructuredWordDocumentLoader,
            silent_errors=True # Suppress errors for files that can't be loaded
        )
        loaded_raw_docs = loader.load()
        logging.info(f"Successfully loaded {len(loaded_raw_docs)} raw .docx documents.")

        # Initialize the RecursiveCharacterTextSplitter
        # This splitter tries to split by paragraphs, then sentences, etc.
        # It's good for maintaining context within chunks.
        # chunk_size: maximum size of each chunk in characters
        # chunk_overlap: number of characters to overlap between adjacent chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,   # Experiment with this value (e.g., 500, 1000, 1500)
            chunk_overlap=200, # Recommended overlap for context (e.g., 10-20% of chunk_size)
            length_function=len,
            is_separator_regex=False,
        )
        
        # Split each loaded raw document into smaller chunks
        for raw_doc in loaded_raw_docs:
            chunks = text_splitter.split_documents([raw_doc])
            all_documents.extend(chunks)
        
        logging.info(f"Split raw documents into {len(all_documents)} smaller chunks.")

    except Exception as e:
        logging.error(f"Error loading or splitting documents from {folder_path}: {e}")
    
    return all_documents

