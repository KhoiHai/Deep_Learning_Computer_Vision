import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    '''
    LeNet5 Architecture
    '''
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        '''
        in_channel (int): Channel in the input
        num_classes (int): Number of classes for classification
        '''
        super().__init__()

        # C1: Convolution Layer
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 6, kernel_size = 5)

        # C2: Convolution Layer
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)

        # C3: Convolution Layer
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = 5)

        # Fully connected layer
        self.fc1 = nn.Linear(in_features = 120, out_features = 84)
        self.fc2 = nn.Linear(in_features = 84, out_features = num_classes)

    def forward(self, x):
        x = F.tanh(self.conv1(x)) 
        x = F.avg_pool2d(x, 2)

        x = F.tanh(self.conv2(x))
        x = F.avg_pool2d(x, 2)

        x = F.tanh(self.conv3(x))
        x = torch.flatten(x, 1)

        x = F.tanh(self.fc1(x))
        x = self.fc2(x)

        return x