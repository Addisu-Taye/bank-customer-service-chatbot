
# Bank Customer Service Chatbot

Step 2: Upload Vector DB to GitHub Release
Zip the vector DB:
A vector-based chatbot system for bank customer service, built using LangChain and HuggingFace embeddings. This project dynamically loads its knowledge base from a GitHub Release, processes financial documents into vector embeddings, and uses a FAISS vector database for efficient semantic search and chatbot interaction.

## Features

zip -r vectors.zip vectors/
Go to your repo on GitHub → Releases → “Draft a new release”
- Multi-format document loading (PDF, DOCX, TXT) using `langchain_unstructured` loaders.
- Domain-specific embeddings using `FinLang/finance-embeddings-investopedia`.
- Streamlit app dynamically downloads and extracts vector database at runtime.
- Conversational enhancements for recognizing greetings and thanks.
- Modular code structure for maintainability.
- Graceful error handling for unsupported or corrupted files.

Upload vectors.zip under Assets
## Project Structure

Publish the release and copy the direct download URL

Paste the URL into app.py:

GITHUB_RELEASES_ZIP_URL = “https://github.com/Addisu-Taye/bank-customer-service-chatbot/releases/download/.../vectors.zip”
Step 3: Set Environment Variables
Create a .env file with:

env

HUGGINGFACEHUB_API_TOKEN=your_token_here
Or configure this variable in your deployment environment (e.g. Streamlit Cloud).

Step 4: Run the Streamlit App
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

streamlit run app.py
🛠 Notes & Troubleshooting
Dynamic Download: App downloads vectors.zip and extracts if vectors/ is missing.

Embedding Model: Uses FinLang/finance-embeddings-investopedia from HuggingFace.
## Installation

LLM Model: Uses google/flan-t5-large via transformers pipeline.
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

👤 About Me
Addisu Taye Dadi
Software Engineer | NLP Enthusiast | Addis Ababa, Ethiopia 🇪🇹
I build intelligent systems to solve real-world problems — like this chatbot.
---

📄 License
MIT License © 2025 Addisu Taye
MIT License © 2025 Addisu Taye.
See the LICENSE file for more details.

See the LICENSE file for details.
🤝 Contact
For questions, issues, or collaborations, please open an issue or contact Addisu Taye.
