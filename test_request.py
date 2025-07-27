import requests

url = "http://127.0.0.1:5000/ask"
data = {"prompt": "What is AI in healthcare?"}
r = requests.post(url, json=data)

print("Status Code:", r.status_code)
print("Response:", r.text)
