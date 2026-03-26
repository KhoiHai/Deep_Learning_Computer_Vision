# Pytorch Lib
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Python Lib
import os

# Medical MNIST
def get_medical_mnist_loader(data_dir: str, batch_size: int = 64, val_split_factor: float = 0.1, test_split_factor: float = 0.15):
    '''
    Create DataLoaders for Medical MNIST

    Args:
        data_dir (str): Path to dataset directory
        batch_size (int): Batch size for train/test
        val_split_factor (float): Ratio for validation split
        test_split_factor (float): Ratio for test split

    Output:
        train_loader, val_loader, test_loader
    '''
    # Check data existence
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok = False)

    # Transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,], [0.5])
    ])

    # Load dataset
    full_dataset = datasets.ImageFolder(
        root = data_dir,
        transform = transform
    )

    total_size = len(full_dataset)
    test_size = int(total_size * test_split_factor)
    all_train_size = total_size - test_size
    val_size = int(all_train_size * val_split_factor)
    train_size = all_train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size]
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
    train_loader, val_loader, test_loader = get_medical_mnist_loader(data_dir = "./data/MedicalMNIST")
    print(len(train_loader.dataset))
    print(len(val_loader.dataset))
    print(len(test_loader.dataset))