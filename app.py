from flask import Flask, render_template, request
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from ask import ask_query  # Your query logic here

app = Flask(__name__, static_url_path='/static')

UPLOAD_FOLDER = 'uploaded_docs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
VECTOR_STORE_PATH = "vector_store"

if os.path.exists(VECTOR_STORE_PATH):
    vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    documents = []
    for file_name in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, file_name)
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file_name.endswith(".docx"):
            loader = Docx2txtLoader(path)
        elif file_name.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        documents.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(VECTOR_STORE_PATH)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    query = request.form.get("prompt")
    answer = ask_query(vectorstore, None, query)
    return {"answer": answer}

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    if file:
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    return "File uploaded successfully. Please go back and query."

if __name__ == "__main__":
    app.run(debug=True)
