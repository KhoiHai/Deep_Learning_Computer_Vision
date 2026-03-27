import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN2Layers(nn.Module):
    '''
    Fully-connected ANN with 2 hidden layers
    Input:
        Input Layer: W x H nodes
        Hidden Layer 1: 256 nodes
        Hidden Layer 2: 64 nodes
        Output Layer: N nodes (N: number of class)
    '''
    def __init__(self, input_size, hidden_size = [256, 64], num_classes = 10):
        '''
        Args:
            input_size (int): Flatten size of the image H x W x C
            hidden_sizes (list of int): Number of neurons of the hidden layers in ANN
            num_classes (int): Number of classes
        '''
        # Define Layers
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

class ANN5Layers(nn.Module):
    '''
    Fully-connected ANN with 5 hidden layers
    Input:
        Input Layer: W x H nodes
        Hidden Layer 1: 512 nodes
        Hidden Layer 2: 256 nodes
        Hidden Layer 3: 128 nodes
        Hidden Layer 4: 64 nodes
        Hidden Layer 5: 32 nodes
        Output Layer: N nodes (N: number of class)
    '''
    def __init__(self, input_size, hidden_sizes = [512, 256, 128, 64, 32], num_classes = 10):
        '''
        Args:
            input_size (int): Flatten size of the image H x W x C
            hidden_sizes (list of int): Number of neurons of the hidden layers in ANN
            num_classes (int): Number of classes
        '''
        super().__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        self.fc6 = nn.Linear(hidden_sizes[4], num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x