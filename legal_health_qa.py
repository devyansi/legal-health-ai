from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os

# Load documents from the "data" folder
def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue
        documents.extend(loader.load())
    return documents

# Split documents into smaller chunks
def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Create and save FAISS vector store
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

# Load FAISS vector store
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings)

# Ask a query
def ask_query(vectorstore, llm, query):
    from langchain.chains import RetrievalQA

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )
    return qa_chain.run(query)

# ============================
# MAIN CODE STARTS HERE
# ============================

query = input("Enter your query: ")

# Step 1: Load & process documents
print("\n📄 Loading and processing documents...\n")
documents = load_documents("data")
chunks = split_documents(documents)

# Step 2: Create and save FAISS vector store
print("🔍 Creating and saving FAISS vector store...\n")
vectorstore = create_vectorstore(chunks)

# Step 3: Load local LLM model using HuggingFacePipeline
print("🤖 Loading local LLM and answering the query...\n")
pipe = pipeline(
    "text-generation",
    model="gpt2",  # lightweight, offline, free model — can be replaced with others
    tokenizer="gpt2",
    max_new_tokens=256,
    temperature=0.5,
    device=-1  # -1 for CPU, 0 for GPU
)
llm = HuggingFacePipeline(pipeline=pipe)

# Step 4: Ask the query
answer = ask_query(vectorstore, llm, query)
print("\n🧠 Answer:\n", answer)

