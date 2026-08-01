import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DOCS_PATH = "docs"
INDEX_PATH = "faiss_index"


def load_documents():
    documents = []
    if not os.path.isdir(DOCS_PATH):
        raise FileNotFoundError(
            f"'{DOCS_PATH}' folder not found. Create it and add at least one PDF."
        )
    pdf_files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{DOCS_PATH}'. Add a PDF and re-run."
        )
    for file in pdf_files:
        loader = PyPDFLoader(os.path.join(DOCS_PATH, file))
        documents.extend(loader.load())
    return documents


def main():
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} pages")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Creating embeddings (first run downloads the model, ~90MB)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Building vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"Done. Saved vector store to '{INDEX_PATH}'.")


if __name__ == "__main__":
    main()
