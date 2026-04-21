# file for UI

import streamlit as st
import os
from rag_engine import InsuranceRAG

# --- Page Configuration ---
st.set_page_config(
    page_title="ShieldCare Insurance AI",
    page_icon="🛡️",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main-title {
        color: #1E3A8A;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🛡️ ShieldCare Insurance Assistant</h1>", unsafe_allow_html=True)
st.write("---")

# --- Initialize Bot (Cached to prevent redundant loading) ---
@st.cache_resource
def init_bot():
    PDF_FILE = "d:/RAG-Project/data/healthcare_policy.pdf"
    bot = InsuranceRAG(PDF_FILE)
    
    # Build vector store if it doesn't exist
    if not os.path.exists("vector_db"):
        with st.status("Building Knowledge Base...", expanded=True) as status:
            st.write("Reading Policy PDF and creating embeddings...")
            bot.create_vector_store()
            status.update(label="Knowledge Base Ready!", state="complete", expanded=False)
            
    return bot

try:
    bot = init_bot()

    # --- Chat Interface Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello Sanket! I am your ShieldCare AI Assistant. How can I help you with your health policy today?"}
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask a question about your policy..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document..."):
                try:
                    response = bot.get_response(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error while fetching response: {e}")

except Exception as e:
    st.error(f"Initialization Failed: {e}")
    st.info("Check if your .env file is present and the PDF path is correct.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Project Info")
    st.write("**Document:** healthcare_policy.pdf")
    st.write("**Engine:** LangChain + ChromaDB")
    st.write("**Embeddings:** HuggingFace MiniLM")
    
    st.write("---")
    if st.button("Clear Vector Cache"):
        if os.path.exists("vector_db"):
            import shutil
            shutil.rmtree("vector_db")
        st.cache_resource.clear()
        st.success("Cache cleared! Restart the app to rebuild.")
        st.rerun()