import sys
import os
sys.path.insert(0, os.path.abspath('..'))

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from data import DataPreparation
from autoencoder import *


MODEL = '2dconv'


# Fetch data
print("*******FETCHING DATA********")
data_prep = DataPreparation()
ticker = 'AAPL'
start_date = '2022-01-01'
end_date = '2023-01-01'
data = data_prep.fetch_financial_data(ticker, start_date, end_date)
processed_data = data_prep.preprocess_data(data)
num_features = processed_data.shape[1]
print(data.head())
print(processed_data.head())
print("Data Shape: ", data.shape)
print("Processed Data Shape: ", processed_data.shape)
print("*********************************************\n\n\n")
print("*********************************************")

# Create sequences
print("*******CREATING SEQUENCES********")
sequence_length = 10
sequences = data_prep.create_sequences(processed_data, sequence_length)
print(sequences[0])
print("Sequences Type: ", type(sequences))
print("Sequences Shape: ", sequences.shape)
if MODEL != '2dconv':
    sequences = sequences.reshape(sequences.shape[0], -1)
    print("Data Flattened Shape: ", sequences.shape)
print("*********************************************\n\n\n")
print("*********************************************")

# Split data into training and validation sets
print("*******SPLITTING DATA********")
X_train_array, X_val_array = train_test_split(sequences, test_size=0.2, random_state=42)
# Convert data to PyTorch tensors
X_train = torch.from_numpy(X_train_array).float()
X_val = torch.from_numpy(X_val_array ).float()
if MODEL == '2dconv':
    X_train = X_train.unsqueeze(1)
    X_val = X_val.unsqueeze(1)
print("X_train Shape: ", X_train.shape)
print("X_val Shape: ", X_val.shape)
print("X_train Type: ", type(X_train))
print("X_val Type: ", type(X_val))
print("*********************************************\n\n\n")
print("*********************************************")

# Create DataLoader objects
print("*******CREATING DATALOADERS********")
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size, shuffle=False)
print("Train Loader Length: ", len(train_loader))
print("Val Loader Length: ", len(val_loader))
print("Batch Shape: ", next(iter(train_loader))[0].shape)
print("*********************************************\n\n\n")
print("*********************************************")

# Initialize the autoencoder model
print("*******INITIALIZING AUTOENCODER********")
if MODEL == '2dconv':
    model = Conv2DAutoencoder()
else:
    model = Autoencoder(sequence_length=sequence_length, num_features=num_features)
print("Model: ", model)
print("*********************************************\n\n\n")
print("*********************************************")

# Training the model
print("*******TRAINING AUTOENCODER********")
# Early stopping parameters
patience = 5
min_delta = 0.0001
best_val_loss = np.inf
counter = 0

# Training parameters
num_epochs = 200
learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        X_batch = batch[0].to(device)
        X_pred = model(X_batch)
        loss = criterion(X_pred, X_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            X_batch = batch[0].to(device)
            X_pred = model(X_batch)
            loss = criterion(X_pred, X_batch)
            val_loss += loss.item()
        val_loss /= len(val_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Early stopping
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
    if counter >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break
print("*********************************************\n\n\n")
print("*********************************************")

# Test model.encode() and print the encoded representation of the first sequence in the validation set
print("*******TESTING MODEL********")
model.eval()
with torch.no_grad():
    encoded = model.encode(X_val[0].unsqueeze(0).to(device))
print("Encoded Representation: ", encoded) if MODEL != '2dconv' else print("Encoded Representation: too long!")
print("Encoded Shape: ", encoded.shape)
print("*********************************************\n\n\n")
print("*********************************************")

# Save the model
print("*******SAVING MODEL********")
# Model name structure: autoencoder_encoded.shape[1]_val_loss.pth (e.g. autoencoder_3_0.001.pth)
model_name = f'models/autoencoder_{encoded.shape[1]}_{best_val_loss:.3f}.pth'
torch.save(model.state_dict(), model_name)
print("Model Saved: ", model_name)
print("*********************************************\n\n\n")
print("*********************************************")

# Load the saved model for a test
print("*******LOADING MODEL********")
# Load the model
if MODEL == '2dconv':
    model = Conv2DAutoencoder()
else:
    model = Autoencoder(sequence_length=sequence_length, num_features=num_features)
model.load_state_dict(torch.load(model_name))
model.eval()
print("Model Loaded: ", model)
# Test model.encode() and print the encoded representation of the first sequence in the validation set
with torch.no_grad():
    encoded = model.encode(X_val[0].unsqueeze(0))
print("Encoded Representation: ", encoded) if MODEL != '2dconv' else print("Encoded Representation: too long!")
print("Encoded Shape: ", encoded.shape)
print("*********************************************\n\n\n")
print("*********************************************")
