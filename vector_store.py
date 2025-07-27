from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from data_loader import load_all_documents
import os

def create_vector_store(model_path="models/all-MiniLM-L6-v2", data_path="data"):
    print("[INFO] Loading documents...")
    documents = load_all_documents(data_path)

    # Split text into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    print("[INFO] Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
    model_name="models/all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True}
)


    print("[INFO] Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Save to disk
    os.makedirs("vector_store", exist_ok=True)
    vectorstore.save_local("vector_store")
    print("[✅] Vector store saved at ./vector_store")

if __name__ == "__main__":
    create_vector_store()
