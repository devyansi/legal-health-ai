from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer once
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_new_tokens=max_tokens, temperature=temperature)
    return f"You asked: {prompt}. (Your AI's answer would go here.)"
