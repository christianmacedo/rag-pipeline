import os
import sys
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)


def load_documents(docs_path: str):
    loaders = {
        "**/*.pdf": PyPDFLoader,
        "**/*.txt": TextLoader,
    }
    documents = []
    for glob_pattern, loader_cls in loaders.items():
        loader = DirectoryLoader(
            docs_path, glob=glob_pattern, loader_cls=loader_cls, silent_errors=True
        )
        documents.extend(loader.load())
    if not documents:
        print(f"No documents found in '{docs_path}'. Add PDF or TXT files and try again.")
        sys.exit(1)
    print(f"Loaded {len(documents)} document(s) from '{docs_path}'.")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks, persist_dir="vectorstore"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(persist_dir)
    print(f"Vector store saved to '{persist_dir}'.")
    return vectorstore


def load_vector_store(persist_dir="vectorstore"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local(
        persist_dir, embeddings, allow_dangerous_deserialization=True
    )
    print("Vector store loaded.")
    return vectorstore


def build_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
    )
    return chain


def ingest(docs_path="docs"):
    documents = load_documents(docs_path)
    chunks = split_documents(documents)
    create_vector_store(chunks)
    print("\nIngestion complete. Run 'python rag_pipeline.py query' to ask questions.")


def query_loop():
    vectorstore = load_vector_store()
    chain = build_qa_chain(vectorstore)
    print("\n--- RAG Pipeline Ready ---")
    print("Type your question and press Enter. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        result = chain.invoke({"query": question})
        print(f"\nAnswer: {result['result']}\n")
        print("Sources:")
        for doc in result["source_documents"]:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            page_info = f" (page {page})" if page != "" else ""
            print(f"  - {source}{page_info}")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python rag_pipeline.py ingest   -> Index documents from /docs folder")
        print("  python rag_pipeline.py query    -> Ask questions about your documents")
        sys.exit(1)

    command = sys.argv[1].lower()
    if command == "ingest":
        docs_path = sys.argv[2] if len(sys.argv) > 2 else "docs"
        ingest(docs_path)
    elif command == "query":
        if not Path("vectorstore").exists():
            print("No vector store found. Run 'python rag_pipeline.py ingest' first.")
            sys.exit(1)
        query_loop()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
