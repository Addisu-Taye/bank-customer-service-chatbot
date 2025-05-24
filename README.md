# Bank Customer Service Chatbot

## Overview

This project implements a Bank Customer Service Chatbot using advanced NLP and vector search techniques. It enables efficient and intelligent responses to customer inquiries based on relevant bank document data.

Key features:
- Document loading from PDFs, Word files, and text files
- Embedding and vector indexing with HuggingFace embeddings and FAISS
- Retrieval-based question answering powered by HuggingFaceHub LLM models
- Spell correction for improved query understanding
- Offline vector database persistence and loading
- Modular code structure for extensibility

## Features

- **Document Loaders**: Supports PDF, DOCX, and TXT files with proper error handling.
- **Embeddings**: Uses sentence-transformers/all-MiniLM-L6-v2 model.
- **Vector Database**: FAISS index creation and local persistence.
- **Chat Engine**: RetrievalQA chain with Google FLAN-T5 large model.
- **Spell Correction**: Corrects user query spelling using TextBlob.
- **Offline Support**: Vector DB saved and loaded locally for offline querying.
- **Extensible**: Easy to add new document types and LLMs.

## Requirements

- Python 3.8+
- langchain
- langchain_huggingface
- langchain_community
- unstructured[local-inference] (for document loading)
- faiss-cpu
- textblob
- pdfminer.six
- python-docx

Install dependencies via:

```bash
pip install -r requirements.txt
Note: For PDF loading, run:

bash
Copy
Edit
pip install "unstructured[pdf]"
to install required PDF dependencies.

Setup & Usage
Place your documents (PDFs, DOCX, TXT) into the bank_docs folder.

Build and save the vector database:

bash
Copy
Edit
python -c "from utils.vector_builder import save_vector_db; save_vector_db('bank_docs', 'vectors')"
Run the chatbot by initializing ChatEngine with the saved vectors and calling the chat() method with your query.

Project Structure
bash
Copy
Edit
bank-customer-service-chatbot/
│
├── bank_docs/               # Folder containing source documents
├── vectors/                 # Folder to save vector database files
├── utils/
│   ├── doc_loader.py        # Document loading utilities
│   ├── vector_builder.py    # Vector database build and save utilities
│   └── chat_engine.py       # Chat engine implementation
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── ...
Troubleshooting
If you get errors about missing PDF dependencies, install with:

bash
Copy
Edit
pip install "unstructured[pdf]"
For Word documents, ensure python-docx is installed.

If FAISS raises errors, verify that faiss-cpu is installed correctly.

Address deprecation warnings by upgrading to the latest LangChain community packages.

License
MIT License © 2025 Addisu Taye Dadi