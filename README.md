🏦 Bank Customer Service Chatbot
A vector-based chatbot system for bank customer service built using LangChain and HuggingFace embeddings. This project now dynamically loads its knowledge base from a GitHub Release, processes documents into vector embeddings, and uses a FAISS vector database for fast semantic search and chatbot interaction.

✨ Features
📄 Multi-format Document Loading: Supports PDF, DOCX, and TXT files using langchain_unstructured loaders for building the knowledge base.
🧠 Financial-Domain Vector Store: Converts documents into embeddings using FinLang/finance-embeddings-investopedia for enhanced financial understanding. The FAISS vector database is then saved and can be dynamically downloaded by the Streamlit app.
🌐 Self-Contained Deployment: The Streamlit application can download and prepare the necessary FAISS vector database at runtime, making deployment easier.
🗣️ Enhanced Conversational AI: The chatbot now understands common greetings ("Hello," "Hi") and appreciative phrases ("Thank you," "Thanks"), providing natural and friendly responses before diving into banking-specific queries.
⚙️ Extensible Utility Functions: Modular structure for easy updates and extension of the core components.
✅ Clean Error Handling: Skips problematic files and logs loading issues during document processing, and handles runtime errors gracefully.
📁 Project Structure
Plaintext

bank-customer-service-chatbot/
│
├── bank_docs/              # Source documents (PDF, DOCX, TXT) - used for building the initial vector DB
├── vectors/                # Local directory for the downloaded/saved vector database files (e.g., index.faiss, index.pkl)
├── utils/
│   ├── doc_loader.py       # Document loading utility
│   ├── vector_builder.py   # Vector DB creation and saving logic
│   ├── chat_engine.py      # Chat interaction logic (LLM, RAG, conversational logic)
│   └── embedder.py         # Centralized embedding model configuration
├── .env                    # Environment variables (e.g., Hugging Face API Token)
├── requirements.txt        # Python dependencies
├── app.py                  # Streamlit web application
└── README.md               # Project documentation
⚙️ Installation
Clone the Repository:

Bash

git clone https://github.com/Addisu-Taye/bank-customer-service-chatbot.git
cd bank-customer-service-chatbot
Create and Activate Virtual Environment:

Bash

python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
Install Dependencies:
Install the core libraries for LangChain, HuggingFace, FAISS, Streamlit, and document parsing. It's recommended to install torch first for GPU support if available, then the others.

Bash

pip install -r requirements.txt
pip install torch # Or tensorflow if preferred. For GPU, check PyTorch/TensorFlow docs.
pip install faiss-cpu # Use faiss-gpu if you have a compatible NVIDIA GPU
pip install textblob
python -m textblob.download_corpora # Download TextBlob's necessary data
Note: Ensure your requirements.txt includes essential packages like langchain, langchain-community, langchain-huggingface, transformers, streamlit, unstructured, pdfminer.six, pypdf, etc. If not, add them or install them individually. For example:

Bash

pip install langchain langchain-community langchain-huggingface transformers streamlit
pip install "unstructured[pdf]" # For PDF support
pip install pypdf # Often required for PDF processing alongside unstructured
🚀 Usage
Step 1: Prepare Your Documents & Generate Vector Database
Place your bank-related documents (PDFs, Word files, TXT files) in the bank_docs/ folder.

Then, use the vector_builder.py script to create your FAISS vector database. This will use the FinLang/finance-embeddings-investopedia model as configured in utils/embedder.py.

Bash

python -c "from utils.vector_builder import save_vector_db; save_vector_db('bank_docs', 'vectors')"
This command will:

Load all supported documents from bank_docs/.
Generate vector embeddings using the model specified in utils/embedder.py.
Save the FAISS index files (e.g., index.faiss, index.pkl) to the vectors/ directory.
Step 2: Upload Your vectors.zip to GitHub Releases
For seamless deployment, the app.py is configured to download the vectors folder from a remote URL.

Create vectors.zip: Zip the vectors/ folder that was just generated. Ensure that index.faiss and index.pkl are at the root level of your vectors.zip archive.
Example (Linux/macOS): zip -r vectors.zip vectors/ (if index.faiss and index.pkl are directly inside vectors/)
Example (Windows): Right-click the vectors folder -> Send to -> Compressed (zipped) folder. Rename to vectors.zip.
Create a GitHub Release: Go to your GitHub repository on the web, navigate to the "Releases" section, and "Draft a new release."
Upload vectors.zip as an Asset: In the release drafting page, under the "Assets" section, upload your vectors.zip file.
Publish Release: Publish the release.
Copy Direct Download URL: After publishing, right-click on the vectors.zip asset link and select "Copy link address." This URL will be used in app.py.
Step 3: Configure app.py and Environment
Update GITHUB_RELEASES_ZIP_URL in app.py: Open app.py and replace "YOUR_DIRECT_GITHUB_RELEASES_ZIP_URL_HERE" with the URL you copied in Step 2.
Set Hugging Face API Token:
Local: Create a .env file in your project root with HUGGINGFACEHUB_API_TOKEN="your_token_here".
Deployment: Configure HUGGINGFACEHUB_API_TOKEN as a secret/environment variable in your deployment platform's settings (e.g., Streamlit Community Cloud secrets).
Step 4: Run the Chatbot Application
Start the Streamlit application to interact with your bank customer service chatbot:

Bash

streamlit run app.py
🛠 Notes and Troubleshooting
Dynamic Download: The first time app.py runs (or if the vectors/ folder is missing/incomplete), it will attempt to download and extract vectors.zip from the specified URL. This might take a moment.
PDF Errors? Ensure unstructured[pdf], pdfminer.six, and pypdf are installed.
Deprecation Warnings? This project uses recent langchain and langchain-unstructured versions, which might occasionally show warnings.
File Skipping? Invalid or unsupported files are skipped with warnings during document loading.
Model Download: The first time you run the chatbot, the specified embedding model (FinLang/finance-embeddings-investopedia) and the large language model (google/flan-t5-large) will be downloaded by Hugging Face Transformers. This may take some time depending on your internet connection.
Conversational Greetings: The chatbot is now pre-programmed to respond to common greetings and thanks before engaging the RAG pipeline.
📦 Dependencies
Python 3.8+
LangChain (langchain, langchain-community, langchain-huggingface)
HuggingFace Transformers (for pipeline and model downloads)
Embedding Model: Configured in utils/embedder.py (FinLang/finance-embeddings-investopedia)
Language Model (LLM): Configured in utils/chat_engine.py (google/flan-t5-large)
FAISS (faiss-cpu or faiss-gpu)
textblob (for spell correction)
streamlit (for the web UI)
unstructured, pdfminer.six, pypdf (for document loading)
requests (for downloading the ZIP)
python-dotenv (for local environment variables)
See requirements.txt for the full list.

About Me
Hi there! I'm Addisu Taye Dadi, a passionate and dedicated software developer based in Addis Ababa, Ethiopia. With a keen interest in leveraging technology to solve real-world problems, I specialize in areas like Natural Language Processing (NLP) and building intelligent systems. I enjoy creating robust and efficient solutions, as demonstrated in projects like the Bank Customer Service Chatbot.

📄 License
MIT License © 2025 Addisu Taye.
See the LICENSE file for more details.

🤝 Contact
For questions, issues, or collaborations, please open an issue or contact Addisu Taye.