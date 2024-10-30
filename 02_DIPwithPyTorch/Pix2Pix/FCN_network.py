import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        ### FILL: add more CONV Layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Decoder (Deconvolutional Layers)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        # Since last layer outputs RGB channels, may need specific activation function
        self.final_layer = nn.Conv2d
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function

    def forward(self, x):
        # Encoder forward pass
        
        # Decoder forward pass
        
        ### FILL: encoder-decoder forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Decoder forward pass
        x = self.deconv1(x)
        x = self.deconv2(x)

        # Final layer
        output = self.final_layer(x)

        # output = ...
        
        return output
    