# Pytorch Lib
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Python Lib
import os

# Handwritten MNIST
def get_mnist_loader(data_dir: str, batch_size: int = 64, val_split_factor: float = 0.1):
    '''
    Create DataLoaders for MNIST dataset

    Args: 
        data_dir (str): Path to dataset directory
        batch_size (int): Batch size for training/testing
        val_split_factor (float): Ratio for validation split

    Output:
        train_loader, val_loader, test_loader
    '''
    
    # Checking data existence
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok = True)

    # Transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset
    full_train_dataset = datasets.MNIST(
        root = data_dir,
        train = True,
        transform = transform,
        download = False
    )

    test_dataset = datasets.MNIST(
        root = data_dir,
        train = False,
        transform = transform,
        download = False
    )

    # Train/Validation split
    val_size = int(len(full_train_dataset) * val_split_factor)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size]
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 2
    )

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_mnist_loader(data_dir = "./data")
    print(len(train_loader.dataset))
    print(len(val_loader.dataset))
    print(len(test_loader.dataset))