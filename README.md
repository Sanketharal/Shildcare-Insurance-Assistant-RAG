# Shildcare-Insurance-Assistant-RAG

ShieldCare Insurance Assistant is a Retrieval-Augmented Generation (RAG) application that answers health insurance policy questions from a PDF document.


## Project Overview

ShieldCare Insurance Assistant is an AI-powered Retrieval-Augmented Generation (RAG) application designed to answer healthcare insurance policy-related questions directly from policy documents. The system uses semantic search and Large Language Models (LLMs) to retrieve relevant information from a healthcare insurance PDF and generate accurate, context-aware responses in a conversational chat interface.

The project combines document retrieval, vector embeddings, and generative AI to help users quickly understand insurance coverage, claims, exclusions, benefits, waiting periods, and other policy details without manually reading lengthy PDFs.

- **UI:** Streamlit chat interface (`app.py`)
- **RAG Engine:** LangChain pipeline (`rag_engine.py`)
- **Vector Store:** ChromaDB (`vector_db/`)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
- **LLM:** Google Gemini (`gemini-2.5-flash`)
- **Source Document:** `Data/healthcare_policy.pdf`

## Repository Structure

- `app.py` - Streamlit application and chat UI
- `rag_engine.py` - PDF loading, chunking, embedding, retrieval, and response generation
- `Data/` - Input policy PDF
- `vector_db/` - Persisted Chroma vector database
- `requirement.txt` - Python dependencies

## Prerequisites

- Python 3.10+
- Google API key for Gemini access

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```
2. Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   ```

## Run the App

```bash
streamlit run app.py
```

## Notes

- The app loads and indexes the healthcare policy PDF, then stores embeddings in `vector_db/`.
- If needed, update the `PDF_FILE` path in `app.py` and `rag_engine.py` to match your local environment.
