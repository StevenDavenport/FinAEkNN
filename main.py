"""
This is the main file for the KNN on financial data project.
This project uses a CNN autoencoder to reduce the dimensionality of the data, and then uses a KNN classifier to classify the data.
"""
import torch

#from knn import KNN
from autoencoder import Autoencoder, Conv2DAutoencoder
from data import DataPreparation

AUTOENCODER = '2dconv'

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    # Fetch data
    print("*******FETCHING DATA********")
    data_prep = DataPreparation()
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-03-03'
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
    if AUTOENCODER != '2dconv':
        sequences = sequences.reshape(sequences.shape[0], -1)
        print("Data Flattened Shape: ", sequences.shape)
    print("*********************************************\n\n\n")
    print("*********************************************")


    # Split data into training and validation sets
    print("*******BUILDING INPUT TENSORS********")
    # Convert data to PyTorch tensors
    input_tensor = torch.from_numpy(sequences).float().to(device)
    if AUTOENCODER == '2dconv':
        input_tensor = input_tensor.unsqueeze(1)
    print("Input Tensor: ", type(input_tensor))
    print("Input Tensor Shape: ", input_tensor.shape)
    print("*********************************************\n\n\n")
    print("*********************************************")

    print("*******LOADING AUTOENCODER********")
    # Load the autoencoder
    path_to_autoencoder = 'autoencoder/models/autoencoder_5_0.011.pth'
    if AUTOENCODER == '2dconv':
        autoencoder = Conv2DAutoencoder()
    else:
        autoencoder = Autoencoder(sequence_length=sequence_length, num_features=num_features)
    autoencoder.load_state_dict(torch.load(path_to_autoencoder))
    autoencoder.eval().to(device)
    print("Model Loaded: ", autoencoder)
    # Test autoencoder.encode() and print the encoded representation of the first sequence in the validation set
    with torch.no_grad():
        encoded = autoencoder.encode(input_tensor[0].unsqueeze(0))
    print("Encoded Representation: ", encoded) if AUTOENCODER != '2dconv' else print("Encoded Representation: too long!")
    print("Encoded Shape: ", encoded.shape)
    print("*********************************************\n\n\n")
    print("*********************************************")


    # Run Autoencoder -> encode data
    print("*******ENCODING DATA********")
    with torch.no_grad():
        encoded_data = autoencoder.encode(input_tensor)
    print("Encoded Data Shape: ", encoded_data.shape)
    print("*********************************************\n\n\n")
    print("*********************************************")

    # Run KNN on encoded data
    print("*******RUNNING KNN********")
    #knn = KNN()
    # TODO: Implement KNN


if __name__ == '__main__':
    main()
