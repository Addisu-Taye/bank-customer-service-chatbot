🏦 Bank Customer Service Chatbot
A vector-based chatbot system for bank customer service built using LangChain and HuggingFace embeddings. This project dynamically loads its knowledge base from a GitHub Release, processes documents into vector embeddings, and uses a FAISS vector database for fast semantic search and chatbot interaction.

✨ Features
📄 Multi-format Document Loading: Supports PDF, DOCX, and TXT files using langchain_unstructured loaders.
🧠 Financial-Domain Vector Store: Uses FinLang/finance-embeddings-investopedia for embedding generation.
🌐 Self-Contained Deployment: Streamlit app downloads and extracts the vector DB at runtime.
🗣️ Enhanced Conversational AI: Recognizes greetings and thanks with polite responses before answering queries.
⚙️ Extensible Utilities: Modular code structure for maintainability.
✅ Clean Error Handling: Skips unsupported files gracefully.
📁 Project Structure
bank-customer-service-chatbot/
│
├── bank_docs/ # Source documents (PDF, DOCX, TXT)
├── vectors/ # Vector database files (index.faiss, index.pkl)
├── utils/
│ ├── doc_loader.py # Loads and processes documents
│ ├── vector_builder.py # Builds and saves the FAISS vector DB
│ ├── chat_engine.py # RAG pipeline + chat logic
│ └── embedder.py # Configures the embedding model
├── .env # Hugging Face API token
├── requirements.txt # All dependencies
├── app.py # Streamlit app entry point
└── README.md # This file

⚙️ Installation
1. Clone the Repository
git clone https://github.com/Addisu-Taye/bank-customer-service-chatbot.git
cd bank-customer-service-chatbot
Create & Activate Virtual Environment
python -m venv venv
Activate (Linux/macOS)
source venv/bin/activate
Activate (Windows)
venv\Scripts\activate
Install Dependencies
pip install -r requirements.txt
pip install torch        # or tensorflow
pip install faiss-cpu    # or faiss-gpu
pip install textblob
python -m textblob.download_corpora
Manually install any missing packages:

pip install langchain langchain-community langchain-huggingface 
transformers streamlit
pip install "unstructured[pdf]" pdfminer.six pypdf
🚀 Usage

Step 1: Prepare Documents & Generate Vectors

Put your PDF/DOCX/TXT documents inside bank_docs/.
Then build the FAISS vector DB:
python -c "from utils.vector_builder import save_vector_db; save_vector_db('bank_docs', 'vectors')"
This will:

Load files from bank_docs/

Generate embeddings with FinLang/finance-embeddings-investopedia

Save vectors/index.faiss and vectors/index.pkl

Step 2: Upload Vector DB to GitHub Release
Zip the vector DB:


zip -r vectors.zip vectors/
Go to your repo on GitHub → Releases → “Draft a new release”

Upload vectors.zip under Assets

Publish the release and copy the direct download URL

Paste the URL into app.py:

GITHUB_RELEASES_ZIP_URL = “https://github.com/your-username/your-repo/releases/download/.../vectors.zip”
Step 3: Set Environment Variables
Create a .env file with:

env

HUGGINGFACEHUB_API_TOKEN=your_token_here
Or configure this variable in your deployment environment (e.g. Streamlit Cloud).

Step 4: Run the Streamlit App

streamlit run app.py
🛠 Notes & Troubleshooting
Dynamic Download: App downloads vectors.zip and extracts if vectors/ is missing.

Embedding Model: Uses FinLang/finance-embeddings-investopedia from HuggingFace.

LLM Model: Uses google/flan-t5-large via transformers pipeline.

PDF Issues: Ensure pdfminer.six, pypdf, and unstructured are installed.

Friendly Chat: Responds politely to greetings and thanks.

📦 Dependencies
Python 3.8+

langchain, langchain-community, langchain-huggingface

transformers, streamlit

unstructured[pdf], pdfminer.six, pypdf

faiss-cpu or faiss-gpu

torch or tensorflow

textblob, python-dotenv, requests

👤 About Me
Addisu Taye Dadi
Software Engineer | NLP Enthusiast | Addis Ababa, Ethiopia 🇪🇹
I build intelligent systems to solve real-world problems — like this chatbot.

📄 License
MIT License © 2025 Addisu Taye

See the LICENSE file for details.