import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))  # (batch_size, seq_len, num of features)
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        return hidden_n.reshape((self.n_features, self.embedding_dim))  # num of units/neurons into the hidden layer

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)  # dense output layer; contains 140 examples

    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))  # reshaping
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))
        return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def create_dataset(sequences):
    dataset = [torch.tensor(s).unsqueeze(1).to(device) for s in sequences]  # Convert each sequence to a tensor & add a dimension
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features

def process_data(file_path):
    # Read data from txt file
    with open(file_path, 'r') as f:
        data = f.read().splitlines()
    # Split data into list of strings
    data = [line.split() for line in data]
    # Convert list of strings into list of integers
    data = [list(map(float, line)) for line in data]
    return data

def predict(model, data):
    heart_patterns = [model(seq) for seq in data]
    pred_losses = [torch.rand(1).item() * 50 for _ in data]  # Example losses for demonstration
    return heart_patterns, pred_losses

def generate_story(ecg_model, ecg_data, tokenizer, model, threshold):
    heart_patterns, pred_losses = predict(ecg_model, ecg_data)
    if isinstance(pred_losses, list):
        pred_losses = torch.tensor(pred_losses)
    prompt = "Details the patient's cardiovascular health based on these ECG data and advises:\n"
    if all(l <= threshold for l in pred_losses):
        prompt += "The patient's heart rate is stable and within normal limits.\n"
    elif any(l > threshold for l in pred_losses):
        prompt += "The patient has some abnormal heart rate patterns, doctors need to monitor closely.\n"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    model.config.pad_token_id = model.config.eos_token_id
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=150, num_return_sequences=1, do_sample=True, top_k=20, top_p=0.95, num_beams=1, temperature=1.2, early_stopping=True)
    story = tokenizer.decode(output[0], skip_special_tokens=True)
    if len(story) > 150:
        story = story[:story.rfind('.') + 1]
    return story

def main():
    # Load models and tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    THRESHOLD = 26

    # Ensure the custom class is available in the current scope
    ecg_model = torch.load('model.pth', map_location=device)
    # Process data
    data_file = 'D:/Github/THT_Delta_Cognition/Data/New_p2.txt'
    new = process_data(data_file)
    new_dataset, seq_len, n_features = create_dataset(new)
    ecg_data = new_dataset

    # Generate and print story
    story = generate_story(ecg_model, ecg_data, tokenizer, model, THRESHOLD)
    print(story)

if __name__ == "__main__":
    main()
