import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found in .env file")
    st.stop()

# Streamlit App UI
st.title("📄 File Question Answering App")

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload a file", type=["pdf"])

if uploaded_file:
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # Create FAISS vector store
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)

    for doc in vectorstore.docstore._dict.values():
        print("---------------------")
        print(doc.page_content)
        print("---------------------") 

    # Create Retrieval-based QA chain
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=api_key, temperature=0),
        retriever=vectorstore.as_retriever()
    )

    # --- Show history at top ---
    if st.session_state.history:
        st.subheader("📜 Chat History")
        for i, chat in enumerate(st.session_state.history, 1):
            st.markdown(f"**Q{i}:** {chat['question']}")
            st.markdown(f"**A{i}:** {chat['answer']}")
            st.markdown("---")

    # --- Input at bottom ---
    query = st.text_input("Type your question here:")

    if st.button("Ask"):
        if query:
            with st.spinner("Thinking..."):
                result = qa.run(query)

            # Save to history¬
            st.session_state.history.append(
                {"question": query, "answer": result}
            )

            # Refresh page so new history shows up immediately
            st.rerun()
