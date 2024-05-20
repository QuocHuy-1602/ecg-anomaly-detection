# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import torch
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
import arff
from torch import nn, optim

import torch.nn.functional as F

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

sns.set(style= 'whitegrid', palette= 'muted', font_scale= 1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

"""<a id="section-two"></a>
## Data
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # if gpu is available use gpu

# Load ARFF file
with open('D:\Github\THT_Delta_Cognition\Data\ECG5000_TRAIN.arff', 'r') as f:
    train_dict = arff.load(f)

# Convert to pandas DataFrame
train = pd.DataFrame(train_dict['data'], columns=[i[0] for i in train_dict['attributes']])

# Load ARFF file
with open('D:\Github\THT_Delta_Cognition\Data\ECG5000_TEST.arff', 'r') as f:
    test_dict = arff.load(f)

# Convert to pandas DataFrame
test = pd.DataFrame(test_dict['data'], columns=[i[0] for i in test_dict['attributes']])

train.head()

test.head()

df = pd.concat([train, test])
df = df.sample(frac=1.0)  # shuffling dataframe
df.shape

CLASS_NORMAL = 1

class_names = ['Normal', 'R on T', 'PVC', 'SP', 'UB']

"""## Data Preprocessing"""

#changing name of the target column

new_columns = list(df.columns)
new_columns[-1] = 'target'
df.columns =new_columns
df.columns

"""<a id="section-four"></a>
## Data Exploration
"""

# total examples in each class
df.target.value_counts()

ax = sns.countplot(x=df.target, data=df)
ax.set_xticklabels(class_names)

# checking mean values of each class
def plot_time_series_class(data, class_name, ax, n_steps=10):
  time_series_df = pd.DataFrame(data)

  smooth_path = time_series_df.rolling(n_steps).mean()
  path_deviation = 2 * time_series_df.rolling(n_steps).std()

  under_line = (smooth_path - path_deviation)[0]
  over_line = (smooth_path + path_deviation)[0]

  ax.plot(smooth_path, linewidth=2)
  ax.fill_between(
    path_deviation.index,
    under_line,
    over_line,
    alpha=.125
  )                                             # standard deviation
  ax.set_title(class_name)

classes = df.target.unique()

fig, axs = plt.subplots(
  nrows=len(classes) // 3 + 1,
  ncols=3,
  sharey=True,
  figsize=(10, 4)
)

for i, cls in enumerate(classes):
  ax = axs.flat[i]
  data = df[df.target == cls] \
    .drop(labels='target', axis=1) \
    .mean(axis=0) \
    .to_numpy()                         # taking mean value for each column by axis = 0
  plot_time_series_class(data, class_names[i], ax)

fig.delaxes(axs.flat[-1])               # deleting last axis cause we only have 5 classes
fig.tight_layout();

"""<a id="section-three"></a>
## Data Preprocessing
"""

normal_df = df[df.target == str(CLASS_NORMAL)].drop(labels='target', axis = 1)
normal_df.shape

"""Merging all other classes and mark them as anomalies ; dropping labels:"""

anomaly_df = df[df.target != str(CLASS_NORMAL)].drop(labels='target', axis=1)
anomaly_df.shape

type(anomaly_df)

"""Splitting the normal examples into train, validation and test sets:"""

# shuffling not needed, did already

# splitting normal dataset to train & val
train_df, val_df = train_test_split(normal_df, test_size=0.15, random_state= RANDOM_SEED)

# splitting val dataframe to val & test
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state= RANDOM_SEED)

test_df.shape

# list of np arrays
train_sequences = train_df.astype(np.float32).to_numpy().tolist()
val_sequences = val_df.astype(np.float32).to_numpy().tolist()
test_sequences = test_df.astype(np.float32).to_numpy().tolist()
anomaly_sequences = anomaly_df.astype(np.float32).to_numpy().tolist()

train_sequences

# converting sequences to torch tensors
# training model single example at a time so that batch size is 1

def create_dataset(sequences):

  dataset = [torch.tensor(s).unsqueeze(1) for s in sequences] # converting each sequence to a tensor & adding a dimension

  n_seq, seq_len, n_features = torch.stack(dataset).shape

  return dataset, seq_len, n_features

"""Each Time Series will be converted to a 2D Tensor in the shape sequence length x number of features (140x1 in our case).

Creating some datasets:
"""

# creating tensors

train_dataset, seq_len, n_features = create_dataset(train_sequences)

val_dataset, seq_len, n_features = create_dataset(val_sequences)

test_dataset, seq_len, n_features = create_dataset(test_sequences)

test_anomaly_dataset, seq_len, n_features = create_dataset(anomaly_sequences)

train_dataset, seq_len, n_features = create_dataset(train_sequences)
print(seq_len, n_features)

print(len(train_dataset))
type(train_dataset)

test_normal_dataset, seq_len, n_features = create_dataset(test_sequences)

"""<a id="section-five"></a>
## Building an LSTM Autoencoder
"""

# using Recurrent Autoencoder and tweaking with a linear layer into the decoder

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
    x = x.reshape((1, self.seq_len, self.n_features))   # (batch_size, seq_len, num of features)

    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)

    return hidden_n.reshape((self.n_features, self.embedding_dim))  # num of units/neurons into the hidden layer

# passing in results from the encoder

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

    # outputting univariate time series with 140 predictions
    self.output_layer = nn.Linear(self.hidden_dim, n_features)  # dense output layer; contains 140 examples

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))    # reshaping

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

"""Our Autoencoder passes the input through the Encoder and Decoder. Let's create an instance of it:"""

model = RecurrentAutoencoder(seq_len, n_features, 128)
model = model.to(device)        # moving model to gpu

"""<a id="section-six"></a>
## Training
"""

def train_model(model, train_dataset, val_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)     # summing error; L1Loss = mean absolute error in torch

  history = dict(train=[], val=[])                      # recording loss history

  for epoch in range(1, n_epochs + 1):
    model = model.train()

    train_losses = []

    for seq_true in train_dataset:     # iterate over each seq for train data
      optimizer.zero_grad()            # no accumulation

      seq_true = seq_true.to(device)   # putting sequence to gpu
      seq_pred = model(seq_true)       # prediction

      loss = criterion(seq_pred, seq_true)  # measuring error

      loss.backward()                  # backprop
      optimizer.step()

      train_losses.append(loss.item())  # record loss by adding to training losses

    val_losses = []
    model = model.eval()
    with torch.no_grad():  # requesting pytorch to record any gradient for this block of code
      for seq_true in val_dataset:
          seq_true = seq_true.to(device)   # putting sequence to gpu
          seq_pred = model(seq_true)       # prediction

          loss = criterion(seq_pred, seq_true)  # recording loss

          val_losses.append(loss.item())    # storing loss into the validation losses
    train_loss = np.mean(train_losses)   # computing loss on training and val data for this epoch
    val_loss = np.mean(val_losses)
    history['train'].append(train_loss)
    history['val'].append(val_loss)

    print(f'Epoch {epoch}: train loss = {train_loss}, val loss = {val_loss}')

  return model.eval(), history      # after training, returning model to evaluation mode

model, history = train_model(model, train_dataset, val_dataset, n_epochs= 150)

ax = plt.figure().gca()

ax.plot(history['train'])
ax.plot(history['val'])
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Loss over training epochs')
plt.show();

"""<a id="section-seven"></a>
## Saving the model
"""

MODEL_PATH = 'model.pth1'

torch.save(model, MODEL_PATH)

# loading pre-trained model from checkpoint

model = torch.load('model.pth')
model = model.to(device)

"""<a id="section-eight"></a>
## Choosing a threshold
"""

def predict(model, dataset):
  predictions, losses = [], []
  criterion = nn.L1Loss(reduction = 'sum').to(device) #L1 to gpu

  with torch.no_grad():
    model = model.eval()      # putting in evaluation mode
    for seq_true in dataset:
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)  # calculating loss

      predictions.append(seq_pred.cpu().numpy().flatten()) # appending predictions & loss to results
      losses.append(loss.item())
  return predictions, losses

"""The helper function goes through each example in the dataset and records the predictions and losses (reconstruction error)."""

_, losses = predict(model, train_dataset)

# plotting the loss/ reconstruction error
sns.distplot(losses, bins= 50, kde= True)

"""The majority of the loss (reconstruction error) seems to be below 25. So normal heartbeat threshold can be around 25.

Identifying *false positives* is okay but not detecting an anomaly can cause fatal outcome. So threshold should be low.
"""

THRESHOLD = 26

"""<a id="section-nine"></a>
## Evaluation

Using the threshold, we can turn the problem into a simple binary classification task:

- If the reconstruction loss for an example is below the threshold, we'll classify it as a *normal* heartbeat
- Alternatively, if the loss is higher than the threshold, we'll classify it as an anomaly

<a id="section-ten"></a>
### Normal hearbeat Predictions

Let's check how well our model does on normal heartbeats. We'll use the normal heartbeats from the test set (our model haven't seen those):
"""

predictions, pred_losses = predict(model, test_normal_dataset)

sns.distplot(pred_losses, bins= 50, kde= True);

"""Seems like majority of the reconstruction error of heartbeat examples of the test set are below threshold 26, whereas occasionally some still have more reconstruction error."""

correct = sum(l <= THRESHOLD for l in pred_losses)
print(f'Correct normal precitions: {correct}/{len(test_normal_dataset)}')

"""<a id="section-eleven"></a>
### Anomaly predictions
"""

anomaly_dataset = test_anomaly_dataset[:len(test_normal_dataset)]

predictions, pred_losses = predict(model, anomaly_dataset)

sns.distplot(pred_losses, bins= 50, kde= True);

correct = sum(l > THRESHOLD for l in pred_losses)
print(f'Correct anomaly precitions: {correct}/{len(anomaly_dataset)}')

"""<a id="section-twelve"></a>
## Looking at Examples
"""

def plot_prediction(data, model, title, ax):
    predictions, pred_losses = predict(model, [data])

    ax.plot(data, label= 'true')
    ax.plot(predictions[0], label= 'predicted')
    ax.set_title(f'{title} (loss: {np.around(pred_losses[0], 2)})') # showing loss in the title with 2 decimal points
    ax.legend()

# 1st row is 6 examples from normal heartbeat, next row represents anomalies
fig, axs = plt.subplots(
    nrows= 2,
    ncols= 6,
    sharex = True,
    sharey = True,
    figsize = (16, 6)
)

# looping through 6 examples
for i, data in enumerate(test_normal_dataset[:6]):
    plot_prediction(data, model, title= 'Normal', ax = axs[0, i])

for i, data in enumerate(test_anomaly_dataset[:6]):
    plot_prediction(data, model, title= 'Anomaly', ax = axs[1, i])

fig.tight_layout();

"""<a id="section-thirteen"></a>
## Generate story
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# Tải mô hình và bộ tokenizer của GPT-3
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

ecg_model = torch.load('model.pth')

def process_data(file_path):
    # read data from txt file
    with open(file_path, 'r') as f:
        data = f.read().splitlines()

    # split data into list of strings
    data = [line.split() for line in data]

    # convert list of strings into list of integers
    data = [list(map(float, line)) for line in data]

    return data

ecg_model = torch.load('model.pth')

def generate_story(ecg_data):
    # heart_patterns = ecg_model(ecg_data)
    heart_patterns, pred_losses  = predict(ecg_model, ecg_data)

    # converting heart patterns to string
    if isinstance(pred_losses, list):
        pred_losses = torch.tensor(pred_losses)

    prompt = "Details the patient's cardiovascular health based on these ECG data and advises:\n"
    if all(l <= THRESHOLD for l in pred_losses):
        prompt += "The patient's heart rate is stable and within normal limits.\n"
    elif any(l > THRESHOLD for l in pred_losses):
        prompt += "The patient has some abnormal heart rate patterns, doctors need to monitor closely.\n"

    # using GPT-3 model to generate story
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones_like(input_ids)
    model.config.pad_token_id = model.config.eos_token_id
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=150, num_return_sequences=1, do_sample=True, top_k=20, top_p=0.95, num_beams=1, temperature=1.2, early_stopping=True)
    story = tokenizer.decode(output[0], skip_special_tokens=True)

    # Cắt câu khi đạt đến độ dài tối đa
    if len(story) > 150:
        story = story[:story.rfind('.') + 1]

    return story

new = process_data('D:/Github/THT_Delta_Cognition/Data/New_p2.txt')
# new = process_data('D:/Github/THT_Delta_Cognition/Data/New_p1.txt')
# converting sequences to torch tensors
new_dataset, seq_len, n_features = create_dataset(new)
# new_dataset, seq_len, n_features = create_dataset(new)
ecg_data = new_dataset

story = generate_story(ecg_data)

print(story)

new = process_data('D:/Github/THT_Delta_Cognition/Data/New_p1.txt')
# converting sequences to torch tensors
new_dataset, seq_len, n_features = create_dataset(new)
# new_dataset, seq_len, n_features = create_dataset(new)
ecg_data = new_dataset

story = generate_story(ecg_data)

print(story)

