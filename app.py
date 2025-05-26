# app.py
import os
import streamlit as st
import requests
from dotenv import load_dotenv
from utils.chat_engine import ChatEngine # Assuming utils/chat_engine.py exists and is correctly implemented
import zipfile
import io
import logging

# Configure logging for better visibility in Streamlit logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

# Retrieve Hugging Face API token from environment variables.
# IMPORTANT: For deployment (e.g., Streamlit Community Cloud), configure this token
# securely in your platform's secrets management (e.g., Streamlit Secrets).
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    st.error("HUGGINGFACEHUB_API_TOKEN environment variable is not set. "
             "Please configure it in your .env file locally, or in your deployment platform's secrets.")
    st.stop() # Stop the app if the token is missing

# Define paths and URLs as constants
LOCAL_VECTOR_DIR = "vectors" # This is the local directory where FAISS files will be extracted

# IMPORTANT: REPLACE THIS URL WITH THE DIRECT DOWNLOAD LINK TO YOUR 'vectors.zip' FILE.
# This ZIP file should contain 'index.faiss' and 'index.pkl' directly at its root.
# You typically get this URL by uploading 'vectors.zip' to a GitHub Release,
# and then right-clicking on the uploaded asset to "Copy link address".
GITHUB_RELEASES_ZIP_URL = "https://github.com/user-attachments/files/20436458/Vectors.zip"
# Example: "https://github.com/YourUsername/YourRepoName/releases/download/v1.0.0/vectors.zip"

# --- Helper Functions ---

@st.cache_resource
def download_and_extract_vector_database(local_dir: str, remote_zip_url: str) -> bool:
    """
    Downloads a ZIP file from a remote URL and extracts its contents
    (expected to be FAISS index files) to the specified local directory.
    Uses Streamlit's cache_resource to ensure this heavy operation runs only once.
    """
    # Check if the necessary FAISS files already exist in the target directory
    # This prevents re-downloading if the app restarts and files are persistent (e.g., in some deployment environments)
    if os.path.exists(os.path.join(local_dir, 'index.faiss')) and \
       os.path.exists(os.path.join(local_dir, 'index.pkl')):
        logging.info("Vector database already exists locally. Skipping download.")
        st.success("Bank knowledge base loaded successfully!")
        return True

    logging.info(f"Initiating download and extraction of vector database from: {remote_zip_url}")
    st.info("Downloading and preparing the bank knowledge base. This may take a moment...")

    try:
        # Send a GET request to download the ZIP file, with a timeout
        response = requests.get(remote_zip_url, stream=True, timeout=300)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        # Create an in-memory byte stream from the response content
        zip_file_bytes = io.BytesIO(response.content)

        # Open the ZIP file from the in-memory stream
        with zipfile.ZipFile(zip_file_bytes, 'r') as zip_ref:
            # Ensure the local directory exists to extract files into
            os.makedirs(local_dir, exist_ok=True)
            
            extracted_files_count = 0
            # Iterate through all members (files/folders) in the ZIP archive
            for member in zip_ref.namelist():
                if not member.endswith('/'): # Process only files, skip directories
                    # Construct the full target path for the extracted file
                    # This assumes index.faiss and index.pkl are at the root of the ZIP
                    target_filepath = os.path.join(local_dir, member) 
                    
                    # Create any necessary parent directories for the target file
                    os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
                    
                    logging.info(f"Extracting {member} to {target_filepath}")
                    # Write the content of the ZIP member to the target file
                    with open(target_filepath, "wb") as outfile:
                        outfile.write(zip_ref.read(member))
                    extracted_files_count += 1
            
            if extracted_files_count == 0:
                logging.error("No files were extracted from the ZIP. Is the ZIP empty or structured unexpectedly?")
                st.error("Error: The downloaded knowledge base ZIP seems empty or malformed.")
                return False

        st.success("Bank knowledge base downloaded and prepared successfully!")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download ZIP from {remote_zip_url}: {e}")
        st.error(f"Network Error: Could not download the bank knowledge base. "
                 f"Please check your internet connection or try again later. Details: {e}")
        return False
    except zipfile.BadZipFile:
        logging.error("Downloaded file is not a valid ZIP archive.")
        st.error("Error: Downloaded knowledge base file is corrupted. Please try again.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during extraction: {e}")
        st.error(f"An unexpected error occurred while preparing the knowledge base: {e}")
        return False

@st.cache_resource
def get_chat_engine_instance(vector_path: str):
    """
    Initializes and returns the ChatEngine.
    Uses Streamlit's cache_resource to ensure the ChatEngine (and its heavy models)
    are loaded only once per app deployment instance.
    """
    logging.info("Initializing ChatEngine...")
    try:
        engine = ChatEngine(vector_path)
        logging.info("ChatEngine initialized successfully.")
        return engine
    except Exception as e:
        logging.error(f"Failed to initialize ChatEngine: {e}", exc_info=True) # Log full traceback
        st.error(f"Failed to set up the AI assistant. Please try refreshing the page or contact support. Details: {e}")
        st.stop() # Stop the app if initialization fails critically


def clear_chat_history():
    """Clears the chat history in Streamlit's session state."""
    st.session_state.chat_history = []
    # Add a welcome message back after clearing for a fresh start
    st.session_state.chat_history.append({"role": "assistant", "content": "Hello! How can I assist you with your banking needs today?"})


# --- Streamlit Page Setup ---
st.set_page_config(page_title="GlobalTrust Bank AI Assistant", layout="wide", initial_sidebar_state="expanded")

st.title("🏦 GlobalTrust Bank AI Assistant")
st.markdown("""
Welcome! I'm here to help you with common questions about our banking services.
Feel free to ask me anything from account inquiries to loan information.
""")

# --- Sidebar ---
st.sidebar.header("Chat Options")
if st.sidebar.button("Clear Chat History", help="Click to clear all messages from the chat."):
    clear_chat_history()
    st.experimental_rerun() # Rerun the app to refresh the chat display

st.sidebar.markdown("---")
st.sidebar.info("Powered by Hugging Face models and Retrieval-Augmented Generation (RAG).")
st.sidebar.markdown("This chatbot uses a comprehensive knowledge base of bank services to provide accurate answers.")


# --- Main Chat Interface Logic ---

# Step 1: Download and extract the vector database
# This function is cached, so it runs only once per app instance.
# It will create the 'vectors' directory and populate it with FAISS files.
if download_and_extract_vector_database(LOCAL_VECTOR_DIR, GITHUB_RELEASES_ZIP_URL):
    # Step 2: Initialize the ChatEngine
    # This function is also cached, ensuring the LLM and retriever are loaded once.
    chat_engine = get_chat_engine_instance(LOCAL_VECTOR_DIR)
else:
    # If download_and_extract_vector_database returns False, it means a critical error
    # occurred during the knowledge base setup, and the app should stop.
    st.error("Critical error: Could not prepare the bank knowledge base. Please check logs for details.")
    st.stop()


# Step 3: Initialize chat history with a welcome message if it's a new session
# This ensures a clean start for new users or after clearing history.
if "chat_history" not in st.session_state or not st.session_state.chat_history:
    clear_chat_history()


# Step 4: Display past chat history from session state
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Step 5: Handle user input and generate assistant response
if prompt := st.chat_input("Ask me anything about our bank services..."):
    # Display user's message in the chat interface
    st.chat_message("user").markdown(prompt)
    # Append user's message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Display assistant's response with a spinner while processing
    with st.chat_message("assistant"):
        with st.spinner("Getting your answer..."):
            # Call the chat method of the initialized ChatEngine instance
            response = chat_engine.chat(prompt)
        st.markdown(response)
    # Append assistant's response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

