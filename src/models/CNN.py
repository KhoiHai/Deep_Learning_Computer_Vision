import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    '''
    Self-designed Convolution Neural Network
    '''
    def __init__(self, input_size = [28, 28], in_channels: int = 1, num_classes: int = 10):
        '''
        in_channel (int): Channel in the input
        num_classes (int): Number of classes for classification
        '''
        super().__init__()

        H, W = input_size

        # H/2 x W/2
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 16, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )

        # H/4 x W/4
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )

        # H/8 x W/8
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * int(H/8) * int(W/8), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x