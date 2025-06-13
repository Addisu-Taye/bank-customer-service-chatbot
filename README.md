
# Bank Customer Service Chatbot

A vector-based chatbot system for bank customer service, built using LangChain and HuggingFace embeddings. This project dynamically loads its knowledge base from a GitHub Release, processes financial documents into vector embeddings, and uses a FAISS vector database for efficient semantic search and chatbot interaction.

## Features
📄 Multi-format Document Loading: Supports PDF, DOCX, and TXT files using langchain_unstructured loaders.
🧠 Financial-Domain Vector Store: Uses FinLang/finance-embeddings-investopedia for embedding generation.
🌐 Self-Contained Deployment: Streamlit app downloads and extracts the vector DB at runtime.
🗣️ Enhanced Conversational AI: Recognizes greetings and thanks with polite responses before answering queries.
⚙️ Extensible Utilities: Modular code structure for maintainability.
✅ Clean Error Handling: Skips unsupported files gracefully.
📁 Project Structure

Upload vectors.zip under Assets
## Project Structure
```
bank-customer-service-chatbot/
├── bank_docs/                # Source documents (PDF, DOCX, TXT)
├── vectors/                  # Vector database files (index.faiss, index.pkl)
├── utils/
│   ├── doc_loader.py         # Loads and processes documents
│   ├── vector_builder.py     # Builds and saves the FAISS vector DB
│   ├── chat_engine.py        # RAG pipeline + chat logic
│   └── embedder.py           # Configures the embedding model
├── .env                     # Hugging Face API token
├── requirements.txt          # All dependencies
├── app.py                   # Streamlit app entry point
└── README.md                 # This file
```

### 1. Clone the Repository

PDF Issues: Ensure pdfminer.six, pypdf, and unstructured are installed.
```bash
git clone https://github.com/Addisu-Taye/bank-customer-service-chatbot.git
cd bank-customer-service-chatbot
```
### 2. Create and Activate Virtual Environment

Friendly Chat: Responds politely to greetings and thanks.
```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

📦 Dependencies
Python 3.8+
```
### 3. Install Dependencies

langchain, langchain-community, langchain-huggingface
```bash
pip install -r requirements.txt

transformers, streamlit
🚀 Usage

unstructured[pdf], pdfminer.six, pypdf
Step 1: Prepare Your Documents
Place your bank-related documents (PDFs, Word files, etc.) in the bank_docs/ folder.

faiss-cpu or faiss-gpu
Step 2: Generate Vector Database

torch or tensorflow
```bash
python -c "from utils.vector_builder import save_vector_db; save_vector_db('bank_docs', 'vectors')"
```

textblob, python-dotenv, requests
## About Me
![Your GitHub Profile Picture](https://github.com/Addisu-Taye.png)
Hi there! I'm Addisu Taye Dadi, a passionate and dedicated software developer based in Addis Ababa, Ethiopia. With a keen interest in leveraging technology to solve real-world problems, I specialize in areas like Natural Language Processing (NLP) and building intelligent systems. I enjoy creating robust and efficient solutions, as demonstrated in projects like the Bank Customer Service Chatbot.

