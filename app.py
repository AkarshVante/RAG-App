import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
# Import for local embeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import shutil

# --- Configuration ---
# Load environment variables from .env file for local development
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(page_title="ChatPDF with Gemini", layout="wide")

# --- UI Styling ---
# Custom CSS for a modern look and feel
st.markdown("""
<style>
    .stApp {
        background_color: #F0F2F6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        border: 1px solid #4CAF50;
    }
    .stTextInput>div>div>input {
        background-color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)


# --- Google API Configuration ---
# It is recommended to set the GOOGLE_API_KEY in Streamlit's secrets management
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set it in Streamlit secrets or as an environment variable.")
    st.stop()

# Configure the generative AI client
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Google Generative AI: {e}")
    st.stop()


# --- Constants ---
# Define the preferred model order, with fallbacks for the generative part
MODEL_ORDER = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
# Define the local model for embeddings to avoid API rate limits
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_DIR = "faiss_index"


# --- Helper Functions ---

def get_pdf_text(pdf_docs):
    """Extract text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.warning(f"Could not read file: {pdf.name}. Error: {e}")
    return text

def get_text_chunks(text):
    """Split text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save a FAISS vector store from text chunks using a local model."""
    try:
        # Use a local Sentence Transformer model for embeddings to avoid API calls
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(FAISS_DIR)
        st.session_state.faiss_ready = True
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.session_state.faiss_ready = False


def get_conversational_chain():
    """Create a conversational QA chain with a custom prompt."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, making sure to provide all the details. If the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Attempt to load models in the specified order
    for model_name in MODEL_ORDER:
        try:
            model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            st.session_state.last_model_used = model_name
            return chain
        except Exception:
            st.warning(f"Model {model_name} not available. Trying next model.")
    
    st.error("All specified models are unavailable. Please check your API key and model access.")
    return None

def user_input(user_question):
    """Handle user input, query the vector store, and display the response."""
    if not st.session_state.get("faiss_ready", False):
        st.error("Vector store not ready. Please process documents first.")
        return

    try:
        # Use the same local model for embedding the user's question
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # Load the FAISS index with support for dangerous deserialization if needed
        new_db = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
        
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()

        if chain:
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("Reply: ", response["output_text"])
            st.info(f"Model used: {st.session_state.get('last_model_used', 'N/A')}")
        else:
            st.error("Could not create a conversational chain.")
            
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")


# --- Main Application ---

def main():
    """Main function to run the Streamlit application."""
    st.header("Chat with PDF using Gemini Pro")

    # Initialize session state variables
    if "faiss_ready" not in st.session_state:
        st.session_state.faiss_ready = os.path.exists(FAISS_DIR)

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        
        if st.button("Submit & Process", key="process_button"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processing Complete")
                    else:
                        st.error("No text could be extracted from the PDFs.")
            else:
                st.warning("Please upload at least one PDF file.")

        if st.button("Clear Index", key="clear_button"):
            if os.path.exists(FAISS_DIR):
                shutil.rmtree(FAISS_DIR)
                st.session_state.faiss_ready = False
                st.success("FAISS index cleared.")
            else:
                st.info("No FAISS index to clear.")

        st.markdown("---")
        if st.session_state.faiss_ready:
            st.success("FAISS Index is ready.")
        else:
            st.warning("FAISS Index is not ready. Please process documents.")

if __name__ == "__main__":
    main()


