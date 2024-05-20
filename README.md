# ECG Story Generator
This project utilizes a GPT-2 model to generate stories based on ECG (Electrocardiogram) data. The ECG data is processed by a custom PyTorch model, and the output is used as input for the GPT-2 model to create unique stories.

## Installation
To install the necessary dependencies, run the following command:
```pip install torch transformers fastapi python-multipart uvicorn ```
## Usage
### Starting the FastAPI Server
First, start the FastAPI server by running:

uvicorn main:app --reload
### Sending a Request
To generate a story, send a POST request to the /generate_story/ endpoint with an ECG data file. The data file should be in text format, with each line containing space-separated floating-point numbers representing ECG readings.

Example of sending a request using curl:
curl -X POST "http://127.0.0.1:8000/generate_story/" -F "file=@path_to_your_ecg_data.txt"

The server will process the ECG data using the custom PyTorch model, and then use the output as input for the GPT-2 model to generate a story. The generated story will be returned in the response to the POST request.

## Models
This project employs two models:

1. Custom PyTorch Model: This model processes the ECG data and is loaded from a file named model.pth.

2. GPT-2 Model: This model generates stories based on the processed ECG data. It is loaded from the pre-trained gpt2 model provided by the transformers library.'

#Contributing
Contributions are welcome! If you wish to contribute, please fork the repository and create a pull request. Make sure to update tests as appropriate and adhere to the coding standards.
