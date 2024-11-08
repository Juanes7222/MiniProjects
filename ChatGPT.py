import requests

data = {
    "contents": [
        {"role": "user", "parts": [{"text": "Hello"}]},
        {"role": "model", "parts": [{"text": "Great to meet you. What would you like to know?"}]},
        {"role": "user", "parts": [{"text": "I have 2 dogs in my house."}]},
        {"role": "user", "parts": [{"text": "How many paws are in my house?"}]}
    ]
}

url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"

key = "AIzaSyARO6gCNUAoKGKM7tWcBtWidUQu0irUyFo"

headers = {
    "Content-Type": "application/json",
    "x-goog-api-key": key
}

response = requests.post(url=url, headers=headers, json=data)

print(response.json())