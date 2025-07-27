# save_model.py

from sentence_transformers import SentenceTransformer

# Load the model from Hugging Face and save it locally
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('models/all-MiniLM-L6-v2')

print("✅ Model downloaded and saved successfully to 'models/all-MiniLM-L6-v2'")
