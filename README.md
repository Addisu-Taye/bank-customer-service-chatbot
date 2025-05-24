# 🏦 Bank Customer Service Chatbot

A vector-based chatbot system for bank customer service built using **LangChain** and **HuggingFace** embeddings.  
This project loads various document formats (PDF, Word, etc.) from local folders, processes them into vector embeddings, and builds a **FAISS** vector database for fast semantic search and chatbot interaction.

---

## ✨ Features

- 📄 **Multi-format Document Loading**  
  Supports PDF, DOCX, and TXT files using `langchain_unstructured` loaders.

- 🧠 **Vector Store Creation**  
  Converts documents into embeddings using `sentence-transformers/all-MiniLM-L6-v2` and stores them in FAISS for similarity search.

- 🌐 **Offline and Online Mode**  
  Supports persistent local storage and retrieval.

- ⚙️ **Extensible Utility Functions**  
  Modular structure for easy updates and extension.

- ✅ **Clean Error Handling**  
  Skips problematic files and logs loading issues for review.

---

## 📁 Project Structure

bank-customer-service-chatbot/
├── utils/
│ ├── doc_loader.py # Document loading utilities
│ ├── vector_builder.py # Embedding & vector DB logic
├── bank_docs/ # Input documents (PDF, DOCX, etc.)
├── vectors/ # Output FAISS vector DB
├── requirements.txt # Python dependencies
└── README.md # This documentation

yaml
Copy
Edit

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Addisu-Taye/bank-customer-service-chatbot.git
cd bank-customer-service-chatbot
2. Create and Activate Virtual Environment
bash
Copy
Edit
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
pip install langchain-unstructured
pip install "unstructured[pdf]"    # For PDF support
pip install pdfminer.six           # Required for PDF parsing
🚀 Usage
Step 1: Prepare Your Documents
Place your bank-related documents (PDFs, Word files, etc.) in the bank_docs/ folder.

Step 2: Generate Vector Database
bash
Copy
Edit
python -c "from utils.vector_builder import save_vector_db; save_vector_db('bank_docs', 'vectors/bank_data.pkl')"
This will:

Load all supported documents in bank_docs/

Generate vector embeddings

Save the FAISS index to vectors/bank_data.pkl

Step 3: Integrate with Chatbot or Search
Use the saved FAISS DB in your chatbot or search application by loading the vector file.

🛠 Notes and Troubleshooting
PDF Errors?
Ensure unstructured[pdf] and pdfminer.six are installed.

Deprecation Warnings?
This project uses the latest langchain_unstructured to avoid deprecated loaders.

File Skipping?
Invalid or unsupported files are skipped with warnings during document loading.

📦 Dependencies
Python 3.8+

LangChain

langchain-unstructured

HuggingFace Transformers

FAISS (faiss-cpu)

unstructured

pdfminer.six

See requirements.txt for the full list.

📄 License
MIT License © 2025 Addisu Taye.
See the LICENSE file for more details.

🤝 Contact
For questions, issues, or collaborations, please open an issue or contact Addisu Taye.