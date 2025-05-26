# 🏦 Bank Customer Service Chatbot

A vector-based chatbot system for bank customer service built using **LangChain** and **HuggingFace embeddings**. This project dynamically loads its knowledge base from a GitHub Release, processes documents into vector embeddings, and uses a **FAISS** vector database for fast semantic search and chatbot interaction.

---

## ✨ Features

- 📄 **Multi-format Document Loading**  
  Supports PDF, DOCX, and TXT files using `langchain_unstructured` loaders for building the knowledge base.

- 🧠 **Financial-Domain Vector Store**  
  Converts documents into embeddings using `FinLang/finance-embeddings-investopedia` for enhanced financial understanding. The FAISS vector database is saved and can be dynamically downloaded by the Streamlit app at runtime.

- 🌐 **Self-Contained Deployment**  
  The Streamlit application downloads and prepares the FAISS vector database on the fly, making deployment seamless.

- 🗣️ **Enhanced Conversational AI**  
  The chatbot understands common greetings ("Hello", "Hi") and appreciative phrases ("Thank you", "Thanks"), offering friendly responses before addressing banking-specific queries.

- ⚙️ **Extensible Utility Functions**  
  Modular architecture allows easy updates and extensions.

- ✅ **Clean Error Handling**  
  Invalid or unsupported files are skipped with warnings, and runtime errors are handled gracefully.

---

## 📁 Project Structure

bank-customer-service-chatbot/
│
├── bank_docs/ # Source documents (PDF, DOCX, TXT)
├── vectors/ # FAISS vector DB files (index.faiss, index.pkl)
├── utils/
│ ├── doc_loader.py # Loads documents
│ ├── vector_builder.py # Creates and saves vector DB
│ ├── chat_engine.py # LLM, RAG, and chatbot logic
│ └── embedder.py # Embedding model configuration
├── .env # Environment variables (Hugging Face token)
├── requirements.txt # Python dependencies
├── app.py # Streamlit chatbot app
└── README.md # Documentation

yaml
Copy
Edit

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/Addisu-Taye/bank-customer-service-chatbot.git
cd bank-customer-service-chatbot
Create and Activate Virtual Environment
bash
Copy
Edit
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
pip install torch               # or tensorflow, for GPU support
pip install faiss-cpu           # or faiss-gpu if using NVIDIA GPU
pip install textblob
python -m textblob.download_corpora
If needed, install additional dependencies individually:

bash
Copy
Edit
pip install langchain langchain-community langchain-huggingface transformers streamlit
pip install "unstructured[pdf]"
pip install pypdf
🚀 Usage
Step 1: Prepare Documents & Generate Vector Database
Place your bank-related documents in the bank_docs/ folder.

Then generate the FAISS vector database:

bash
Copy
Edit
python -c "from utils.vector_builder import save_vector_db; save_vector_db('bank_docs', 'vectors')"
Step 2: Upload vectors.zip to GitHub Releases
Zip the vectors folder

macOS/Linux: zip -r vectors.zip vectors/

Windows: Right-click → Send to → Compressed folder → Rename to vectors.zip

Create a GitHub Release

Go to your repo → Releases → Draft a new release

Upload vectors.zip as an Asset

Publish the Release

Copy the direct download URL

Right-click vectors.zip → Copy link address

Step 3: Configure app.py and Environment
Update download URL
Replace YOUR_DIRECT_GITHUB_RELEASES_ZIP_URL_HERE in app.py with your copied URL.

Set Hugging Face API token
Add to .env:

env
Copy
Edit
HUGGINGFACEHUB_API_TOKEN=your_token_here
Or use your deployment platform's secrets management.

Step 4: Run the Application
bash
Copy
Edit
streamlit run app.py
🛠 Notes & Troubleshooting
First run?
The app will download and extract vectors.zip dynamically. Wait a moment for it to complete.

PDF support issues?
Ensure these are installed: unstructured[pdf], pdfminer.six, pypdf.

Deprecation warnings?
This app uses recent langchain versions that may log warnings.

File skipping?
Unsupported files are skipped with clear warnings.

Model download?
First-time use will download:

Embedding: FinLang/finance-embeddings-investopedia

LLM: google/flan-t5-large

Conversational enhancements
Friendly replies to greetings or appreciation are handled before banking queries.

📦 Dependencies
Python 3.8+

LangChain ecosystem:

langchain, langchain-community, langchain-huggingface

HuggingFace Transformers

Embedding Model: FinLang/finance-embeddings-investopedia

Language Model: google/flan-t5-large

FAISS (faiss-cpu or faiss-gpu)

textblob (for spell correction)

streamlit (UI)

unstructured, pdfminer.six, pypdf (for document loading)

requests, python-dotenv

See requirements.txt for the full list.

👨‍💻 About Me
Hi there! I'm Addisu Taye Dadi, a passionate software developer based in Addis Ababa, Ethiopia.
I specialize in NLP and intelligent systems that solve real-world problems—like this customer service chatbot.

📄 License
MIT License © 2025 Addisu Taye
See the LICENSE file for more details.

🤝 Contact
For questions, issues, or collaborations:

Open an issue

Or contact me directly via GitHub.