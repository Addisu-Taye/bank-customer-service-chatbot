import os
import pickle
from textblob import TextBlob # For spelling correction
from transformers import pipeline # For local LLM inference
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline # For wrapping local pipeline as LLM
from langchain_huggingface import HuggingFaceEmbeddings # For embeddings
import logging
import streamlit as st # Ensure streamlit is imported here for @st.cache_resource

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_resource
def load_embedding_model():
    """
    Loads the HuggingFace embedding model.
    This function is cached by Streamlit to load the model only once.
    """
    logging.info("Loading HuggingFaceEmbeddings model: FinLang/finance-embeddings-investopedia...")
    return HuggingFaceEmbeddings(model_name="FinLang/finance-embeddings-investopedia")

@st.cache_resource
def load_faiss_vector_db(vector_directory_path: str, _embedding_model):
    """
    Loads the FAISS vector database from a local directory.
    The _embedding_model argument is prefixed with an underscore to tell Streamlit's
    caching mechanism not to hash this complex object.
    """
    logging.info(f"Attempting to load FAISS vector database from: {vector_directory_path}")

    # Verify that the necessary FAISS files exist before attempting to load
    if not os.path.exists(os.path.join(vector_directory_path, 'index.faiss')) or \
       not os.path.exists(os.path.join(vector_directory_path, 'index.pkl')):
        logging.error(f"FAISS index files (e.g., index.faiss, index.pkl) not found in {vector_directory_path}")
        raise FileNotFoundError(f"FAISS index files not found in {vector_directory_path}. "
                                "Please ensure the vector store is properly built and available.")

    try:
        # Load the FAISS index using the provided embeddings
        vectordb = FAISS.load_local(
            folder_path=vector_directory_path,
            embeddings=_embedding_model, # Use the _embedding_model passed as argument
            allow_dangerous_deserialization=True # Necessary for loading pickled FAISS indices
        )
        logging.info(f"FAISS vector database loaded successfully from {vector_directory_path}")
        return vectordb
    except Exception as e:
        logging.error(f"Failed to load FAISS vector database from {vector_directory_path}: {e}")
        raise RuntimeError(f"Could not load vector database: {e}")

@st.cache_resource
def load_llm_pipeline():
    """
    Loads the HuggingFace text-to-text generation pipeline (LLM) locally.
    This function is cached by Streamlit to load the model only once.
    """
    logging.info("Loading HuggingFace pipeline 'google/flan-t5-base' for local inference...")
    try:
        # Using device=-1 for CPU. Set to 0 for GPU (if available and configured)
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base", # CHANGED to 'google/flan-t5-base'
            max_length=512, # Max length of the generated response
            temperature=0.3, # Controls creativity/randomness. Lower (0.1-0.3) for factual.
            device=-1 # Use CPU
        )
        logging.info("HuggingFace pipeline 'google/flan-t5-base' initialized for local inference.")
        # Wrap the pipeline in HuggingFacePipeline for LangChain compatibility
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        logging.error(f"Failed to load HuggingFace pipeline model locally: {e}", exc_info=True)
        st.error(f"Failed to load the local language model. This might be due to a network issue when downloading the model, or insufficient RAM. Details: {e}")
        st.stop() # Stop the app if initialization fails critically


class ChatEngine:
    """
    Core Chat Engine responsible for managing the vector DB, LLM,
    and handling user queries, including pre-processing for greetings.
    """
    def __init__(self, vector_directory_path: str):
        """Initialize Chat Engine with vector DB and QA chain."""
        # Load embedding model (cached)
        self.embedding = load_embedding_model()
        
        # Load FAISS vector database (cached), passing the embedding model
        # Using the underscore prefix for the cached argument to prevent hashing issues
        self.vectordb = load_faiss_vector_db(vector_directory_path, self.embedding)
        
        # Initialize retriever from the vector database
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})
        logging.info("Retriever initialized from vector database.")

        # Load LLM pipeline (cached)
        self.llm = load_llm_pipeline()

        # Initialize the RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff", # 'stuff' combines all retrieved documents into a single prompt
            retriever=self.retriever,
            return_source_documents=False, # Set to True if you want to see which documents were used
            verbose=False # Keep verbose for debugging the chain itself
        )
        logging.info("RetrievalQA chain initialized.")

    def correct_spelling(self, query: str) -> str:
        """
        Spell-checks the input query and returns the corrected version.
        Handles potential TextBlob errors gracefully.
        """
        try:
            corrected_query = str(TextBlob(query).correct())
            if corrected_query.lower() != query.lower():
                logging.info(f"[Spelling Correction] '{query}' -> '{corrected_query}'")
            return corrected_query
        except Exception as e:
            logging.warning(f"Failed to perform spelling correction for '{query}': {e}. Returning original query.")
            return query

    def get_response(self, query: str) -> str: # Renamed from 'chat' to 'get_response' for consistency with app.py
        """
        Answers user queries by first checking for common greetings/thanks,
        then spell-checking, and finally invoking the QA chain.
        Includes robust error handling.
        """
        if not query or not query.strip():
            return "Please provide a valid question."

        try:
            # Apply spelling correction here
            corrected_query = self.correct_spelling(query) 
            
            response = self.qa_chain.invoke({"query": corrected_query})
            
            # Extract the 'result' from the response dictionary
            if isinstance(response, dict) and 'result' in response:
                response_text = response['result']
                # Remove common LLM artifacts
                response_text = response_text.replace("Based on the provided context, ", "").strip()
                response_text = response_text.replace("According to the context, ", "").strip()
                logging.info(f"Generated response for query '{corrected_query}': {response_text[:100]}...")
                return response_text
            else:
                logging.warning(f"Unexpected response format from QA chain: {response}")
                return "I apologize, but I could not retrieve a clear answer. Could you please rephrase your question?"
        except Exception as e:
            logging.error(f"Error during QA chain invocation for query '{query}': {e}", exc_info=True)
            return "I am sorry, but an error occurred while processing your request. Please try again later. This might be due to insufficient memory for the model."

