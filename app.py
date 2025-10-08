import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import shutil
import time

# --- Configuration ---
# Load environment variables from .env file for local development
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(page_title="ChatPDF", layout="wide", initial_sidebar_state="expanded")

# --- UI Styling (from app1.py) ---
# Custom CSS for the dark, modern UI with WhatsApp-like chat messages
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* Define animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide other default Streamlit elements */
    #MainMenu, footer, .stDeployButton {
        visibility: hidden;
    }
    
    /* Main app background */
    .stApp {
        background-color: #07101a;
    }
    
    /* --- Sidebar Styling --- */
    [data-testid="stSidebar"] {
        background-color: #07101a;
        border-right: 1px solid #13303f;
    }
    
    /* Sidebar button with glowing effect */
    [data-testid="stSidebar"] .stButton button {
        border-radius: 999px;
        border: 1px solid #2c5970;
        background-color: transparent;
        color: #add8e6;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 0 5px 0px rgba(0, 150, 255, 0.3);
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: rgba(173, 216, 230, 0.1);
        color: #fff;
        border-color: #00aaff;
        box-shadow: 0 0 10px 2px rgba(0, 150, 255, 0.6);
    }
    
    /* Status badge styling */
    .status-badge {
        display: block; padding: 8px; border-radius: 20px;
        font-weight: 600; margin: 12px auto; text-align: center;
    }
    .status-ready { background-color: rgba(25, 195, 125, 0.1); color: #19c37d; }
    .status-not-ready { background-color: rgba(255, 102, 51, 0.1); color: #ff6633; }
    
    /* --- Main Chat Interface Styling --- */
    /* Chat message styling with animation and glow */
    [data-testid="stChatMessage"] {
        animation: fadeIn 0.5s ease-in-out;
        border-radius: 10px;
        border: 1px solid #13303f;
        background-color: #0a1929;
        box-shadow: 0 0 8px 1px rgba(0, 150, 255, 0.15);
        margin: 10px 0;
    }
    
    /* Chat input box styling */
    [data-testid="stChatInput"] textarea {
        color: #FFFFFF; /* Makes the input text white and visible */
    }
</style>
""", unsafe_allow_html=True)


# --- Google API Configuration ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set it in Streamlit secrets or as an environment variable.")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Google Generative AI: {e}")
    st.stop()


# --- Constants ---
# Define the preferred model order, with fallbacks for the generative part
MODEL_ORDER = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def get_embedding_model():
    """Load the sentence transformer model, cached for performance."""
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_vector_store(text_chunks):
    """Create and save a FAISS vector store from text chunks using a local model."""
    try:
        embeddings = get_embedding_model()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(FAISS_DIR)
        st.session_state.faiss_ready = True
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.session_state.faiss_ready = False

def get_conversational_chain():
    """Create a conversational QA chain with a custom prompt and model fallback."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the context." Do not provide a wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    for model_name in MODEL_ORDER:
        try:
            model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            st.session_state.last_model_used = model_name
            return chain
        except Exception:
            st.warning(f"Model {model_name} not available. Trying next model.")
    
    st.error("All specified Gemini models are unavailable. Please check your API key and model access.")
    return None

# --- Main Application ---
def main():
    """Main function to run the Streamlit application."""

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "faiss_ready" not in st.session_state:
        st.session_state.faiss_ready = os.path.isdir(FAISS_DIR)

    # --- Sidebar for Document and Session Management ---
    with st.sidebar:
        st.header("📄 ChatPDF")
        st.markdown("Your personal document assistant.")
        
        status_text = "Ready" if st.session_state.faiss_ready else "No Documents"
        status_class = "status-ready" if st.session_state.faiss_ready else "status-not-ready"
        st.markdown(f'<div class="status-badge {status_class}">Status: {status_text}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("1. Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your PDF files here.",
            accept_multiple_files=True,
            type=['pdf'],
            label_visibility="collapsed"
        )
        
        if st.button("2. Process Documents", use_container_width=True, disabled=not uploaded_files):
            with st.spinner("Processing documents... This may take a moment."):
                raw_text = get_pdf_text(uploaded_files)
                if raw_text.strip():
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                    st.success("✅ Documents processed!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Processing failed. No readable text found in PDFs.")

        st.markdown("---")
        st.subheader("3. Manage Session")
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.session_state.faiss_ready and st.button("Delete Documents", use_container_width=True):
            shutil.rmtree(FAISS_DIR, ignore_errors=True)
            st.session_state.faiss_ready = False
            st.session_state.messages = []
            st.success("Documents and index deleted.")
            time.sleep(1)
            st.rerun()
            
    # --- Main Chat Interface ---
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="font-size: 3em; font-weight: 700; color: #FFFFFF;">
            <span style="margin-right: 15px;">📄</span>Chat With Your Documents
        </h1>
    </div>
    """, unsafe_allow_html=True)

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    prompt_placeholder = "Please process documents first..." if not st.session_state.faiss_ready else "Ask a question..."
    if prompt := st.chat_input(prompt_placeholder, disabled=not st.session_state.faiss_ready):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    embeddings = get_embedding_model()
                    vector_store = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
                    docs = vector_store.similarity_search(prompt, k=4)
                    
                    chain = get_conversational_chain()
                    if chain:
                        response = chain({"input_documents": docs, "question": prompt}, return_only_outputs=True)
                        answer = response.get("output_text", "Sorry, I couldn't generate a response.")
                        if st.session_state.get('last_model_used'):
                           answer += f"\n\n*— Generated by {st.session_state.last_model_used}*";
                    else:
                        answer = "The conversation chain could not be initialized. Please check the logs."
                        
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_message = f"An error occurred: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()




