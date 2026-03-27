import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Trainer:
    '''
    Class for training models

    Attributes:
        model: Model using (for example: ANN, CNN, etc)
        train_loader, test_loader: Training and Validation Loader 
        optimizer: Name of the optimizer used (for example: adam, sgd, etc)
        lr (float): Learning Rate
        criterion: Loss function used, possibly Cross Entropy Loss
        device: Training device (cpu or gpu)
    '''

    def __init__(self, model, train_loader, test_loader, optimizer = "adam", lr = 0.001, criterion = None, device = "cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        # Loss
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

        # Optimizer
        if optimizer == "adam":
            self.optimizer = optim.AdamW(self.model.parameters(), lr = lr, weight_decay=1e-2)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr = lr, momentum = 0.0)
        elif optimizer == "sgd_momentum":
            self.optimizer = optim.SGD(self.model.parameters(), lr = lr, momentum = 0.9)
        else:
            raise ValueError(f"Optimizer {optimizer} is not founded")
        
    def train_one_epoch(self):
        # Set the model training mode
        self.model.train()
        running_loss = 0
        # Loop every batch
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            # Reset optimizer
            self.optimizer.zero_grad()
            # Foward
            outputs = self.model(x)
            # Calculate the loss
            loss = self.criterion(outputs, y)
            # Back probagation
            loss.backward() # Calculate gradients
            self.optimizer.step() # Update weights
            running_loss += loss.item() * x.size(0)
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss 
    
    def validate(self):
        # Set the model validation mode
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)

                loss = self.criterion(outputs, y)
                total_loss += loss.item() * x.size(0)

                _, preds = torch.max(outputs, 1)
                total += y.size(0)
                correct += (preds == y).sum().item()
        return correct/total, total_loss/total