import requests
from pprint import pprint


def test_server(input_text):
    """
    Sends a POST request to the server with the given input text and prints the server's response.

    Args:
        input_text (str): The text to be sent to the server for prediction.

    Returns: None
    """
    # API endpoint URL for the server
    url = "http://127.0.0.1:8000/predict"

    # Send a POST request with the request payload
    payload = {"input_text": input_text}
    response = requests.post(url, json=payload)

    # Print the response from the server
    pprint(response.json())


if __name__ == "__main__":
    # Sample input text for testing. e.g., "The capital of France is [MASK]."
    input_text = "He walked to the [MASK]."
    test_server(input_text)
