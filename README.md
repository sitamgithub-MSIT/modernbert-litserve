# ModernBERT LitServe

[ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) models offer significant improvements over original BERT, boasting faster training, better performance on downstream tasks, and enhanced efficiency through architectural changes and optimized training techniques. This project demonstrates the use of ModernBERT model for the fill-mask task served using LitServe which is an easy-to-use, flexible serving engine for AI models built on FastAPI.

## Project Structure

The project is structured as follows:

- `server.py`: The file containing the main code for the web server.
- `client.py`: The file containing the code for client-side requests.
- `LICENSE`: The license file for the project.
- `README.md`: The README file that contains information about the project.
- `assets`: The folder that contains the screenshots for working on the application.

## Tech Stack

- Python (for the programming language)
- PyTorch (for the deep learning framework)
- Hugging Face Transformers Library (for the model)
- LitServe (for the web application)

## Getting Started

To get started with this project, follow the steps below:

1. Run the server: `python server.py`
2. Sending requests: `python client.py`

Now, you can see the output of the model based on the text input. The model will predict the masked word in the sentence.

## Usage

The project can be used to serve the ModernBERT model using LitServe. Here the model is used for fill-mask task where the model predicts the masked word in the sentence. Other than that, the model can be used for various NLP downstream tasks after fine-tuning.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please raise an issue to discuss the changes you want to make. Once the changes are approved, you can create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or suggestions regarding the project, feel free to reach out to me on my GitHub profile.

Happy coding! 🚀
