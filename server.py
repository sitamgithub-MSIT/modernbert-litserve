# Import necessary libraries
import torch
from transformers import pipeline
import litserve as ls


class ModernBertAPI(ls.LitAPI):
    """
    ModernBertAPI is a subclass of ls.LitAPI that provides an interface to a BERT-based model for the "fill-mask" task.

    Methods:
        - setup(device): Initializes the pipeline with the specified device.
        - decode_request(request): Decodes the input request to extract the input text.
        - predict(input_text): Uses the pipeline to predict the masked tokens in the input text.
        - encode_response(results): Encodes the prediction results into a dictionary format.
    """

    def setup(self, device):
        """
        Sets up the pipeline for the fill-mask task using the specified device.
        """
        # Initialize the pipeline with the specified device
        model_name = "answerdotai/ModernBERT-base"
        self.pipeline = pipeline(
            "fill-mask",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device=device,
        )

    def decode_request(self, request):
        """
        Decodes the input request to extract the input text.
        """
        # Extract the input text from the request
        return request["input_text"]

    def predict(self, input_text):
        """
        Generates a prediction based on the provided input text.
        """
        # Generate a prediction using the pipeline
        return self.pipeline(input_text)

    def encode_response(self, results):
        """
        Encodes the given results into a dictionary format.
        """
        # Return the results in a dictionary format
        return {"token": results[0]["token_str"], "sequence": results[0]["sequence"]}


if __name__ == "__main__":
    # Create an instance of the ModernBertAPI and run the LitServer
    api = ModernBertAPI()
    server = ls.LitServer(api, track_requests=True)
    server.run(port=8000)
