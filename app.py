import os
import streamlit as st
import requests
from dotenv import load_dotenv
from utils.chat_engine import ChatEngine # Assuming utils/chat_engine.py exists and is correctly implemented
from langchain_huggingface import HuggingFaceEmbeddings # Import for embedding model (still used by helper functions)
import zipfile
import io
import logging
import re # Import regex for advanced text matching

# Configure logging for better visibility in Streamlit logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

# Retrieve Hugging Face API token from environment variables.
# IMPORTANT: For production deployment (e.g., Streamlit Community Cloud),
# configure this token securely in your platform's secrets management (e.g., Streamlit Secrets).
# For local development, ensure it's set in your .env file.
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    st.error("HUGGINGFACEHUB_API_TOKEN environment variable is not set. "
             "Please configure it in your .env file locally, or in your deployment platform's secrets.")
    st.stop() # Stop the app if the token is missing

# Define paths as constants
LOCAL_VECTOR_DIR = "vectors" # This is the local directory where FAISS files are expected
# In a production environment, ensure this 'vectors' folder is included in your deployment
# package or that the vector database is fetched from a persistent storage location (e.g., S3).

# --- Helper Functions for Chatbot Conversation Flow ---

# Removed @st.cache_resource
def _get_chat_engine_instance_internal(vector_path: str): # Removed 'embedder' argument
    """
    Initializes and returns the ChatEngine.
    This function directly initializes the engine without Streamlit caching.
    The ChatEngine itself will load its embedding model internally.
    """
    logging.info("Initializing ChatEngine...")
    
    # Check if the vector store files exist locally
    if not os.path.exists(os.path.join(vector_path, 'index.faiss')) or \
       not os.path.exists(os.path.join(vector_path, 'index.pkl')):
        st.error(f"Vector database files not found in '{vector_path}'. "
                 "Please ensure 'index.faiss' and 'index.pkl' are present in this folder. "
                 "Run 'generate_vectors.py' first to create them.")
        st.stop() # Stop the app if files are missing

    try:
        # Pass only the vector_path to the ChatEngine constructor
        engine = ChatEngine(vector_path) 
        logging.info("ChatEngine initialized successfully.")
        return engine
    except Exception as e:
        logging.error(f"Error initializing ChatEngine: {e}", exc_info=True) # Log full traceback
        st.error(f"Failed to set up the AI assistant. Please try refreshing the page or contact support. Details: {e}")
        st.stop() # Stop the app if initialization fails critically


def clear_chat_history():
    """Clears the chat history in Streamlit's session state."""
    st.session_state.chat_history = []
    # Add a welcome message back after clearing for a fresh start
    st.session_state.chat_history.append({"role": "assistant", "content": "Hello! How can I assist you with your banking needs today?"})
    # The embedder_instance is no longer stored directly in app.py's session_state
    # as ChatEngine now handles it internally.
    if 'chat_engine' in st.session_state:
        del st.session_state['chat_engine']

def handle_conversational_input(prompt: str) -> str | None:
    """
    Handles basic greetings and common words to provide a more natural conversational flow.
    Returns a predefined response if a match is found, otherwise returns None.
    """
    lower_prompt = prompt.lower().strip()
    logging.info(f"DEBUG: Entering handle_conversational_input for prompt: '{prompt}'")

    # Basic greetings
    if any(word in lower_prompt for word in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
        logging.info("DEBUG: Matched greeting.")
        return "Hello there! How can I help you with your banking needs today?"
    
    # Common small talk words
    if any(word in lower_prompt for word in ["how are you", "what's up", "how's it going"]):
        logging.info("DEBUG: Matched small talk.")
        return "I'm a digital assistant, so I don't have feelings, but I'm ready to assist you! What can I help you with?"
    
    if "thank you" in lower_prompt or "thanks" in lower_prompt:
        logging.info("DEBUG: Matched thank you.")
        return "You're most welcome! Is there anything else I can assist you with?"
    
    if "bye" in lower_prompt or "goodbye" in lower_prompt:
        logging.info("DEBUG: Matched goodbye.")
        return "Goodbye! Have a great day."

    logging.info("DEBUG: No conversational match found.")
    # Return None if no conversational match is found
    return None

def detect_multiple_questions(prompt: str) -> bool:
    """
    Heuristically detects if the prompt contains multiple distinct questions.
    Checks for multiple question marks or common conjunctions followed by question-like phrasing.
    """
    lower_prompt = prompt.lower().strip()
    logging.info(f"DEBUG: Entering detect_multiple_questions for prompt: '{prompt}'")

    # Rule 1: More than one question mark
    if lower_prompt.count('?') > 1:
        logging.info("DEBUG: Detected multiple question marks. Assuming multiple questions.")
        return True
    
    # Rule 2: Conjunctions that often link separate questions
    # Combined with question words or verbs
    conjunctions = [" and ", " also ", " in addition to ", " furthermore ", " next ", " then ", " what's more ", " as well as "]
    question_starters = ["what", "how", "when", "where", "why", "can you", "could you", "is there", "do i", "tell me", "explain"]

    for conj in conjunctions:
        if conj in lower_prompt:
            # Check if after the conjunction, there's another question-like phrase
            # Split by regex to handle cases where conjunctions are followed by punctuation
            parts = re.split(re.escape(conj), lower_prompt, 1) 
            if len(parts) > 1:
                after_conj = parts[1].strip()
                # Check if the part after the conjunction starts with a question word or looks like a new question
                if any(after_conj.startswith(qs) for qs in question_starters) or after_conj.count('?') >= 1:
                    logging.info(f"DEBUG: Detected conjunction '{conj}' with another question. Assuming multiple questions.")
                    return True
    
    # Rule 3: Multiple distinct question phrases even if not ending with '?' but containing question words
    # Split by common sentence terminators and check each segment
    sentences = re.split(r'[.!?]\s*', lower_prompt) # Split by ., !, or ? followed by space
    question_like_sentences = [s for s in sentences if any(qs in s for qs in question_starters)]
    
    # Filter out very short segments that might be noise or part of a single question's flow
    question_like_sentences = [s for s in question_like_sentences if len(s.split()) > 2] # Require at least 3 words

    if len(question_like_sentences) > 1:
        logging.info(f"DEBUG: Detected multiple question-like sentences ({len(question_like_sentences)}). Assuming multiple questions.")
        return True

    logging.info("DEBUG: No multiple questions detected.")
    return False


def is_out_of_domain(prompt: str) -> bool:
    """
    Determines if a user's question is likely out of the chatbot's banking domain.
    This uses a simple keyword matching approach.
    """
    lower_prompt = prompt.lower()
    logging.info(f"DEBUG: Entering is_out_of_domain for prompt: '{prompt}'")
    
    # Keywords indicating general knowledge or non-banking topics
    out_of_domain_keywords = [
        "weather", "news", "recipe", "joke", "story", "poem", "history", 
        "science", "sports", "entertainment", "movie", "book", "music",
        "tell me about yourself", "who are you", "what is your name",
        "calculate", "define", "explain concept of", # if not followed by banking terms
        "stock market prediction", # specific non-banking financial
        "coding", "programming", "build an app", "football", "celebrity"
    ]

    # Banking-specific keywords that would keep it in-domain, even if some OOD words exist
    in_domain_keywords = [
        "account", "loan", "card", "transfer", "deposit", "withdraw", "fee", 
        "interest", "credit", "debit", "mortgage", "business", "foreign exchange",
        "remittance", "investment", "statement", "online banking", "mobile app",
        "branch", "atm", "swift", "iban", "eligibility", "apply", "balance",
        "secure", "fraud", "customer service"
    ]

    # Check for strong out-of_domain signals
    for keyword in out_of_domain_keywords:
        if keyword in lower_prompt:
            # If an out-of-domain keyword is found, check if it's immediately negated by a banking keyword
            is_definitely_out = True
            for in_kw in in_domain_keywords:
                if in_kw in lower_prompt:
                    is_definitely_out = False
                    break # Found an in-domain keyword, so it might not be OOD
            if is_definitely_out:
                logging.info(f"DEBUG: Matched out-of-domain keyword: '{keyword}'. Returning True.")
                return True
                
    # A catch-all for very short, non-specific queries that aren't greetings
    # Ensure it's not a greeting before marking it as OOD based on length
    if len(lower_prompt.split()) < 3 and not handle_conversational_input(prompt):
        logging.info("DEBUG: Short non-conversational query, likely out-of-domain. Returning True.")
        return True # Likely too vague to be banking-related

    logging.info("DEBUG: No strong out-of-domain signal. Returning False.")
    return False


# --- Streamlit Page Setup ---
st.set_page_config(page_title="GlobalTrust Bank AI Assistant", layout="wide", initial_sidebar_state="expanded", page_icon="🌐") # Added globe emoji as page_icon

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

# Use st.session_state to store the chat_engine instance
# The ChatEngine itself will handle loading the embedding model internally.

if 'chat_engine' not in st.session_state:
    # Initialize the ChatEngine instance only once per session, passing only the vector_path
    st.session_state.chat_engine = _get_chat_engine_instance_internal(LOCAL_VECTOR_DIR)
    st.info("Chat engine initialized.") # User feedback

# Now access the instance from session_state
chat_engine = st.session_state.chat_engine

# Initialize chat history with a welcome message if it's a new session
# This ensures a clean start for new users or after clearing history.
if "chat_history" not in st.session_state or not st.session_state.chat_history:
    clear_chat_history()


# Display past chat history from session state
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input and generate assistant response
if prompt := st.chat_input("Ask me anything about our bank services..."):
    # Display user's message in the chat interface
    st.chat_message("user").markdown(prompt)
    # Append user's message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    response = None
    logging.info(f"Processing user prompt: '{prompt}'")

    # NEW ORDER: Check for multiple questions first
    if detect_multiple_questions(prompt):
        response = "It looks like you've asked multiple questions at once! For the best assistance, please ask one question at a time. I'll be happy to help with each of them."
        logging.info("DEBUG: Detected multiple questions. Responded with guidance.")
    # Then, check for conversational greetings/small talk
    elif conversational_response := handle_conversational_input(prompt):
        response = conversational_response
        logging.info("DEBUG: Prompt handled by conversational input.")
    # Next, check if the question is out of domain
    elif is_out_of_domain(prompt):
        response = "I apologize, but my current knowledge base is focused on GlobalTrust Bank's services. I might not be able to answer questions outside this scope. Please ask me about accounts, loans, cards, or other banking-related topics!"
        logging.info("DEBUG: Prompt identified as out of domain.")
    # If not conversational and not out of domain, proceed with RAG retrieval
    else:
        logging.info("DEBUG: Prompt sent to RAG chain.")
        # Display assistant's response with a spinner while processing
        with st.spinner("Getting your answer..."):
            # Call the chat method of the initialized ChatEngine instance
            response = chat_engine.get_response(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    # Append assistant's response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

