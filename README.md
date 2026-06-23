# RAG Chat Application

A Retrieval-Augmented Generation (RAG) based chatbot that enables users to ask questions from uploaded documents and receive context-aware answers using Large Language Models (LLMs).

## 🚀 Features

- Upload PDF documents
- Extract and process document content
- Split text into chunks for efficient retrieval
- Generate embeddings using Hugging Face models
- Store embeddings in a vector database
- Retrieve relevant document chunks
- Generate accurate answers using an LLM
- Interactive web interface built with Streamlit

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Backend
- Python

### AI & NLP
- LangChain
- Hugging Face Embeddings
- Ollama

### Vector Database
- ChromaDB

### Document Processing
- PyPDFLoader
- RecursiveCharacterTextSplitter

## 📂 Project Structure

```text
Ragchat/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── PDFs
│
├── chroma_db/
│
└── myenv/
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Ragchat.git
cd Ragchat
```

### 2. Create Virtual Environment

```bash
python -m venv myenv
```

### 3. Activate Virtual Environment

Windows:

```bash
myenv\Scripts\activate
```

Linux/Mac:

```bash
source myenv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

```bash
streamlit run app.py
```

## 📖 How It Works

1. User uploads PDF documents.
2. Documents are loaded and processed.
3. Text is split into smaller chunks.
4. Embeddings are generated using Hugging Face models.
5. Embeddings are stored in ChromaDB.
6. User asks a question.
7. Relevant document chunks are retrieved.
8. LLM generates a context-aware answer.

## 🎯 Future Enhancements

- Multi-PDF support
- Chat history memory
- Hybrid search (Keyword + Vector Search)
- Cloud deployment
- Authentication and user management
- Support for multiple LLM providers

## 🤝 Contributing

Contributions are welcome. Feel free to fork the repository and submit pull requests.

## 📜 License

This project is developed for educational and learning purposes.

## 👨‍💻 Author

**Kushendra Unnam**

B.Tech Computer Science and Engineering  
Institute of Aeronautical Engineering, Hyderabad
