
import requests
import json

AUTH_TOKEN = "AUTH_TOKEN"

url = "URL"

payload = {}

headers = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    result = response.json()
    print("Response:", result)
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
