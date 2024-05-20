from fastapi import FastAPI, UploadFile, File
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import nn
from typing import List
import io

# Import model classes from model_api.py
from model_api import (
    RecurrentAutoencoder, create_dataset, process_data, generate_story, device
)

app = FastAPI()

# Load models and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
THRESHOLD = 26

# Ensure the custom class is available in the current scope
ecg_model = torch.load('model.pth', map_location=device)

@app.post("/generate_story/")
async def generate_story_from_ecg(file: UploadFile = File(...)):
    # Read uploaded file content
    content = await file.read()
    data_str = content.decode("utf-8")

    # Process data
    lines = data_str.splitlines()
    data = [list(map(float, line.split())) for line in lines]

    # Create dataset
    new_dataset, seq_len, n_features = create_dataset(data)
    ecg_data = new_dataset

    # Generate story
    story = generate_story(ecg_model, ecg_data, tokenizer, model, THRESHOLD)

    return {"story": story}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
