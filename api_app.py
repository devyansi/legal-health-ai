from flask import Flask, request, jsonify
from ai_response import generate_answer

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    answer = generate_answer(prompt)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
