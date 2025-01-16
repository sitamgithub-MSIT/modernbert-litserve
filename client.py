import requests
from pprint import pprint

url = "http://localhost:8000/predict"

# Define the text and labels for classification
payload = {
    "text": "Angela Merkel is a politician in Germany and leader of the CDU",
    "labels": ["politics", "economy", "entertainment", "environment"],
    "hypothesis_template": "This text is about {}",
    "multi_label": False,
}

# Send the request to the server
response = requests.post(url, json=payload)

# Print the predictions
pprint(response.json())
