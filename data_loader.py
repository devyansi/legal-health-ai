import os
from langchain.schema import Document  # If you're returning LangChain Document objects

def load_all_documents(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()

        documents.append(Document(page_content=text))
    
    return documents


