🏦 Bank Customer Service Chatbot
Overview
This project delivers a Bank Customer Service Chatbot designed to revolutionize how financial institutions handle customer inquiries. Leveraging cutting-edge Natural Language Processing (NLP) and vector search techniques, the chatbot provides instant, accurate, and intelligent responses by drawing information directly from your bank's existing documentation.

✨ Key Features
Intelligent Document Ingestion: Seamlessly load knowledge from diverse sources, including PDFs, Word documents, and plain text files, with robust error handling.
Advanced Embeddings: Utilizes the efficient sentence-transformers/all-MiniLM-L6-v2 model to transform documents into powerful, searchable vectors.
High-Performance Vector Search: Employs FAISS for rapid and precise similarity searches, ensuring quick retrieval of relevant information.
Sophisticated Chat Engine: Powered by a RetrievalQA chain integrated with the potent Google FLAN-T5 large LLM, providing comprehensive and contextually appropriate answers.
Smart Spell Correction: Enhances query understanding by automatically correcting user input spelling using TextBlob, minimizing frustration.
Offline Capability: Your vector database can be persisted locally and loaded for offline querying, ensuring continuous service availability.
Modular & Extensible Design: The codebase is structured for easy expansion, allowing you to integrate new document types, Large Language Models (LLMs), and functionalities with minimal effort.
🛠️ Requirements
To get started, you'll need Python 3.8+ and the following libraries:

langchain
langchain_huggingface
langchain_community
unstructured[local-inference] (for robust document loading)
faiss-cpu
textblob
pdfminer.six
python-docx
Installation
Install the primary dependencies using the provided requirements.txt file:

Bash

pip install -r requirements.txt
Note: For comprehensive PDF document loading, you'll need to install additional dependencies:

Bash

pip install "unstructured[pdf]"
🚀 Setup & Usage
Follow these simple steps to set up and run your chatbot:

Populate Your Knowledge Base: Place all your bank-related documents (PDFs, DOCX, TXT files) into the ./bank_docs folder.

Build Your Vector Database: Generate and save the vector database from your documents. This process creates the searchable knowledge base for the chatbot.

Bash

python -c "from utils.vector_builder import save_vector_db; save_vector_db('bank_docs', 'vectors')"
Engage with the Chatbot: Once the vector database is built, you can initialize the ChatEngine with the saved vectors and start querying.

Python

from utils.chat_engine import ChatEngine
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface.llms import HuggingFaceHub
import os

# Set your HuggingFace API token as an environment variable
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_HUGGINGFACE_TOKEN" # Replace with your actual token

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("vectors", embeddings, allow_dangerous_deserialization=True) # Load the saved vector DB

# Initialize the LLM (ensure HUGGINGFACEHUB_API_TOKEN is set)
llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.5, "max_length": 512})

chatbot = ChatEngine(vector_db, llm)

# Start chatting!
query = "What are the requirements for opening a new savings account?"
response = chatbot.chat(query)
print(f"Chatbot: {response}")

query = "How can I apply for a loan?"
response = chatbot.chat(query)
print(f"Chatbot: {response}")
📂 Project Structure
The project is organized for clarity and maintainability:

bank-customer-service-chatbot/
│
├── bank_docs/              # Your source documents (PDFs, DOCX, TXT)
├── vectors/                # Saved vector database files (FAISS index)
├── utils/
│   ├── doc_loader.py       # Handles document loading from various formats
│   ├── vector_builder.py   # Scripts to build and save the FAISS vector database
│   └── chat_engine.py      # Core chatbot logic: spell correction, retrieval, LLM integration
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── ...                     # Other potential files (e.g., example usage script)
⚠️ Troubleshooting
Encountering issues? Here are some common fixes:

PDF Loading Errors: If you experience problems loading PDFs, ensure you've installed the necessary unstructured PDF dependencies:
Bash

pip install "unstructured[pdf]"
Word Document Issues: Verify that python-docx is correctly installed for processing .docx files.
FAISS Errors: Double-check that faiss-cpu is installed properly.
Deprecation Warnings: Keep your langchain and langchain-community packages updated to their latest versions to address any deprecation notices.
📄 License
This project is open-sourced under the MIT License.

© 2025 Addisu Taye Dadi