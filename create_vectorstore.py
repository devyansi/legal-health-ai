from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredPDFLoader, UnstructuredWordDocumentLoader, CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

import os

# ✅ Load all documents from /data directory
def load_all_documents(directory):
    all_docs = []

    # Load .txt files
    txt_loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
    all_docs.extend(txt_loader.load())

    # Load .pdf files
    pdf_loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=UnstructuredPDFLoader)
    all_docs.extend(pdf_loader.load())

    # Load .docx files
    docx_loader = DirectoryLoader(directory, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader)
    all_docs.extend(docx_loader.load())

    # Load .csv files
    csv_loader = DirectoryLoader(directory, glob="**/*.csv", loader_cls=CSVLoader)
    all_docs.extend(csv_loader.load())

    return all_docs

# ✅ Main logic
if __name__ == "__main__":
    print("📄 Loading documents...")
    documents = load_all_documents("data")

    print("✂️ Splitting documents...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    print("🔤 Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("📦 Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(texts, embedding_model)

    save_path = "vector_store"
    print(f"💾 Saving FAISS index to '{save_path}'...")
    vectorstore.save_local(save_path)

    print("✅ FAISS vector store created and saved successfully.")
