import torch.nn as nn
import torch
from torchvision import transforms
from torch.nn import functional as F

theta_deg = 180

def init_weights(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

#! Models defined below


def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), 
        #batch normalization implementation
        nn.LazyBatchNorm2d())#,


class NiN(nn.Module):
    def __init__(self, lr=0.01, num_classes=1):
        super().__init__()
        #self.save_hyperparameters()
        self.shared = nn.Sequential(
            nin_block(96, kernel_size=64, strides=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, kernel_size=40, strides=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(3, stride=2),
            nin_block(384, kernel_size=32, strides=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(3, stride=2)
            )

        self.split = nn.Sequential(
            nin_block(num_classes, kernel_size=24, strides=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())
        
        self.net = nn.Sequential(
            nin_block(96, kernel_size=5, strides=3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, kernel_size=3, strides=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(3, stride=2),
            nin_block(384, kernel_size=3, strides=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(3, stride=2),
            nin_block(2, kernel_size=3, strides=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)
        # x = self.shared(x)
        # horizontal = self.split(x)
        # vertical = self.split(x)
        # return torch.cat((horizontal, vertical), dim=1)


#! muti class output regression network

class EyeDiameterNet(nn.Module):
    def __init__(self):
        super(EyeDiameterNet, self).__init__()

        # Shared layers
        self.shared_conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 125 * 125, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)  # Two outputs: vertical and horizontal diameters
        )

    def forward(self, x):
        x = self.shared_conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        diameters = self.fc_layers(x)
        return diameters