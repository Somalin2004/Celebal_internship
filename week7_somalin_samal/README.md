# Document Question Answering System (RAG)

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system that answers
questions based on custom documents (PDFs). Instead of relying only on a language
model's internal knowledge, the system retrieves relevant information from documents
and generates answers grounded in that information.

## Architecture
1. **Document Ingestion** – PDF loaded and converted to raw text (`PyPDFLoader`)
2. **Text Chunking** – Text split into ~800-character overlapping chunks
3. **Embedding Creation** – Each chunk embedded using `all-MiniLM-L6-v2` (local, free)
4. **Vector Database** – Embeddings stored in a FAISS index
5. **Query Processing** – User question embedded with the same model
6. **Context Retrieval** – Top-3 most similar chunks retrieved from FAISS
7. **Answer Generation** – Groq's `llama-3.3-70b-versatile` generates the final answer
   using only the retrieved context

## Tech Stack
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector store:** FAISS
- **LLM:** Groq API (`llama-3.3-70b-versatile`)
- **Framework:** LangChain
- **UI:** Streamlit

## Setup
1. Run `setup.bat` (Windows) or `bash setup.sh` (Mac/Linux)
2. Copy `.env.example` to `.env` and add your Groq API key
3. Put one or more PDFs in the `docs/` folder
4. Run `python ingest.py` to build the vector index
5. Run `streamlit run app.py` to launch the Q&A interface

## Example
**Question:** "What is the main idea of the document?"
**Process:** Retrieve relevant sections → provide as context → generate concise answer

## Possible Improvements
- Better chunking strategies (semantic chunking)
- Hybrid search (keyword + vector)
- Re-ranking retrieved chunks
- Support for multiple file types (txt, docx)
