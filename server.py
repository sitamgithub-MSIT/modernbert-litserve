# Import necessary libraries
import torch
from transformers import pipeline
import litserve as ls


class ModernBertAPI(ls.LitAPI):
    """
    ModernBertAPI is a subclass of ls.LitAPI that provides an interface to a ModernBERT-based model for the "zero-shot classification" task.

    Methods:
        - setup(device): Initializes the pipeline with the specified device.
        - decode_request(request): Decodes the incoming request to extract the inputs.
        - predict(data): Uses the pipeline to predict the classification results.
        - encode_response(results): Encodes the prediction results into a dictionary format.
    """

    def setup(self, device):
        """
        Sets up the pipeline for the zero-shot classification task.
        """
        # Initialize the pipeline with the specified device
        model_name = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
        self.pipeline = pipeline(
            "zero-shot-classification", 
            model=model_name, 
            torch_dtype=torch.bfloat16, 
            device=device
        )

    def decode_request(self, request):
        """
        Decodes the input request to extract the inputs.
        """
        # Decode the incoming request
        text = request["text"]
        labels = request["labels"]
        hypothesis_template = request.get(
            "hypothesis_template", "This example is {}"
        )
        multi_label = request.get("multi_label", False)
        return text, labels, hypothesis_template, multi_label

    def predict(self, data):
        """
        Generates a prediction based on the provided inputs.
        """
        # Perform zero-shot classification
        text, labels, hypothesis_template, multi_label = data
        return self.pipeline(
            text,
            labels,
            hypothesis_template=hypothesis_template,
            multi_label=multi_label,
        )

    def encode_response(self, results):
        """
        Encodes the given results into a dictionary format.
        """
        # Encode the response as JSON
        output = [
            {"label": label, "score": f"{score * 100:.2f}%"}
            for label, score in zip(results["labels"], results["scores"])
        ]
        output.sort(key=lambda x: float(x["score"].strip("%")), reverse=True)
        return {"predictions": output}


if __name__ == "__main__":
    # Create an instance of the ModernBertAPI and run the LitServer
    api = ModernBertAPI()
    server = ls.LitServer(api, track_requests=True)
    server.run(port=8000)
