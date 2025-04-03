import requests

url = "http://127.0.0.1:5000/query"
headers = {"Content-Type": "application/json"}
data = {"query": "What is the meaning of life?"}

response = requests.post(url, json=data, headers=headers)
print(response.json())
