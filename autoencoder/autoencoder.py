import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, sequence_length=10, num_features=10):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(sequence_length * num_features, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, sequence_length * num_features),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

class Conv2DAutoencoder(nn.Module):
    def __init__(self, bottleneck_size=5):
        super(Conv2DAutoencoder, self).__init__()

        # Assuming input shape [batch_size, 1, num_features, sequence_length]
        # where num_features is treated as height, and sequence_length as width

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )

        # Compresstion layer
        self.compressor = nn.Linear(32 * 3 * 3, bottleneck_size)

        # Expandtion layer
        self.expander = nn.Linear(bottleneck_size, 32 * 3 * 3)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(6,7), stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(7,7), stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #print("Input Shape: ", x.shape)
        x = self.encoder(x)
        #print("Encoded Shape: ", x.shape)
        x = x.view(x.size(0), -1)
        #print("Flattened Shape: ", x.shape)
        x = self.compressor(x)
        #print("Compressed Shape: ", x.shape)
        x = self.expander(x)
        #print("Expanded Shape: ", x.shape)
        x = x.view(x.size(0), 32, 3, 3)
        #print("Dilated Shape: ", x.shape)
        x = self.decoder(x)
        #print("Decoded Shape: ", x.shape)
        return x

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.compressor(x)
        return x

