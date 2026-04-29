# RAG Pipeline with Google Gemini & FAISS

A production-ready Retrieval-Augmented Generation (RAG) pipeline that indexes PDF and TXT documents into a FAISS vector store and answers questions using Google Gemini as the LLM.

## Architecture

```
Documents (PDF/TXT)
        │
        ▼
  Text Splitter (recursive, 1000 chars)
        │
        ▼
  Embeddings (Google embedding-001)
        │
        ▼
  Vector Store (FAISS)
        │
        ▼
  Retriever (top-k similarity search)
        │
        ▼
  LLM (Gemini 1.5 Flash)
        │
        ▼
  Answer + Source Documents
```

## Stack

- **LLM:** Google Gemini 1.5 Flash
- **Embeddings:** Google Generative AI Embeddings (embedding-001)
- **Vector Store:** FAISS (CPU)
- **Framework:** LangChain
- **Document Loaders:** PDF, TXT

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/rag-pipeline.git
cd rag-pipeline
```

**2. Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure your API key**

```bash
cp .env.example .env
```

Edit `.env` and add your [Google AI Studio](https://aistudio.google.com/apikey) API key.

## Usage

**Index your documents**

Place PDF or TXT files in the `docs/` folder, then run:

```bash
python rag_pipeline.py ingest
```

**Ask questions**

```bash
python rag_pipeline.py query
```

```
--- RAG Pipeline Ready ---
Type your question and press Enter. Type 'exit' to quit.

You: What are the benefits of RAG?

Answer: RAG reduces hallucinations by grounding responses in real data,
allows LLMs to access domain-specific knowledge without fine-tuning,
keeps responses up-to-date, and provides traceable sources.

Sources:
  - docs/sample.txt
```

## Project Structure

```
rag-pipeline/
├── docs/              # Place your documents here
│   └── sample.txt     # Sample document for testing
├── rag_pipeline.py    # Main pipeline code
├── requirements.txt   # Python dependencies
├── .env.example       # API key template
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## How It Works

1. **Ingestion:** Documents are loaded from `docs/`, split into chunks of 1000 characters with 200-character overlap, embedded using Google's embedding model, and stored in a FAISS index.

2. **Query:** The user's question is embedded, the 4 most similar chunks are retrieved from FAISS, and Gemini generates an answer grounded in those chunks. Source documents are returned for traceability.

## License

MIT
