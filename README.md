# ModernBERT LitServe

[![Open In Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/sitammeur/studios/modernbert-litserve)

[ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) models offer significant improvements over the original BERT, boasting faster training, better performance on downstream tasks, and enhanced efficiency through architectural changes and optimized training techniques. This project demonstrates the use of the ModernBERT model for the fill-mask task served using LitServe, an easy-to-use, flexible serving engine for AI models built on FastAPI.

## Project Structure

The project is structured as follows:

- `server.py`: The file containing the main code for the web server.
- `client.py`: The file containing the code for client-side requests.
- `LICENSE`: The license file for the project.
- `README.md`: The README file that contains information about the project.
- `assets`: The folder containing screenshots for working on the application.

## Tech Stack

- Python (for the programming language)
- PyTorch (for the deep learning framework)
- Hugging Face Transformers Library (for the model)
- LitServe (for the serving engine)

## Getting Started

To get started with this project, follow the steps below:

1. Run the server: `python server.py`
2. Upon running the server successfully, you will see uvicorn running on port 8000.
3. Open a new terminal window.
4. Run the client: `python client.py`

Now, you can see the output of the model based on the text input. The model will predict the masked word in the sentence.

## Usage

The project can be used to serve the ModernBERT model using LitServe. Here, the model is used for the fill-mask task, where it predicts the masked word in the sentence. After fine-tuning, the model can also be used for various NLP downstream tasks.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please raise an issue to discuss the changes you want to make. Once the changes are approved, you can create a pull request.

## License

This project is licensed under the [Apache-2.0 License](LICENSE).

## Contact

If you have any questions or suggestions about the project, feel free to contact me on my GitHub profile.

Happy coding! 🚀
