import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables from .env
load_dotenv()

# Get the token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize the client
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    token=hf_token
)

# Generate text
response = client.text_generation(
    prompt="Explain diabetes in simple words.",
    max_new_tokens=100,
    temperature=0.7
)

print(response)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
from data_loader import load_all_documents

def initialize_chain():
    documents = load_all_documents('data')
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docsearch = FAISS.from_documents(texts, embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    return chain, docsearch

initialize_chain()
