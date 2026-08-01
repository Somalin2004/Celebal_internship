import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

INDEX_PATH = "faiss_index"


@st.cache_resource
def load_qa_chain():
    if not os.path.isdir(INDEX_PATH):
        st.error("No vector store found. Run `python ingest.py` first, then restart this app.")
        st.stop()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to your .env file.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.2,
    )

    prompt = PromptTemplate(
        template="""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain


st.title("📄 RAG Document Q&A (Groq)")
st.write("Ask questions about your uploaded documents.")

qa_chain = load_qa_chain()

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"query": query})
        st.subheader("Answer:")
        st.write(result["result"])

        with st.expander("Sources used"):
            for i, doc in enumerate(result["source_documents"]):
                st.write(f"**Chunk {i+1}:**")
                st.write(doc.page_content[:300] + "...")
