import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")

st.title("📄 PDF Chatbot using RAG")

# ---------------------------
# Session State Initialization
# ---------------------------

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ---------------------------
# Cache Embedding Model
# ---------------------------

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ---------------------------
# Cache LLM
# ---------------------------

@st.cache_resource
def load_llm():
    return OllamaLLM(model="phi3")
    # If you prefer llama3:
    # return OllamaLLM(model="llama3")


# ---------------------------
# File Upload
# ---------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

# ---------------------------
# Process PDF ONLY ONCE
# ---------------------------

if uploaded_file is not None:

    if st.session_state.pdf_name != uploaded_file.name:

        st.session_state.pdf_name = uploaded_file.name

        with st.spinner("Processing PDF..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name

            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            # Split text
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(documents)

            # Embeddings
            embedding = load_embedding_model()

            # Vector DB
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embedding
            )

            # Store retriever in session
            st.session_state.retriever = vectordb.as_retriever(
                search_kwargs={"k": 3}
            )

        st.success("PDF processed successfully!")

# ---------------------------
# Ask Questions
# ---------------------------

if st.session_state.retriever is not None:

    query = st.text_input(
        "Ask a question from the uploaded PDF"
    )

    if query:

        with st.spinner("Searching and generating answer..."):

            docs = st.session_state.retriever.invoke(query)

            context = "\n".join(
                [doc.page_content for doc in docs]
            )

            prompt = f"""
            Answer the question using ONLY the context below.

            If the answer is not present in the context,
            say:
            "I couldn't find that information in the PDF."

            Context:
            {context}

            Question:
            {query}
            """

            llm = load_llm()

            response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response)

        # Optional Debug Section
        with st.expander("Retrieved Chunks"):
            for i, doc in enumerate(docs, start=1):
                st.write(f"Chunk {i}")
                st.write(doc.page_content)
                st.divider()