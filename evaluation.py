import os
import json
import csv
import torch

from src.datasets.mnist_loader import get_mnist_loader
from src.datasets.fashion_mnist_loader import get_fashion_mnist_loader
from src.datasets.medical_mnist_loader import get_medical_mnist_loader

from src.trainers.trainer import Trainer
from src.models.ANN import ANN5Layers, ANN2Layers

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            _, preds = torch.max(outputs, 1)

            total += y.size(0)
            correct += (preds == y).sum().item()

    acc = correct / total
    return acc

def evaluation(config_path: str = "./configs/ANN/MNIST_vanilla_sgd_5.json"):
    # ---------------- CONFIG ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("[PROCESSOR] GPU is used")
    else:
        print("[PROCESSOR] CPU is used")

    if not os.path.exists(config_path):
        raise FileExistsError(f"Configuration path {config_path} is not found!")
    print("[LOADING CONFIGURATION] Start loading configuration file")

    with open(config_path, "r") as f:
        config = json.load(f)

    dataset_name = config["dataset_name"]
    dataset_path = config["dataset_path"]
    model_name = config["model_name"]
    input_size = config["input_size"]
    batch_size = config["batch_size"]

    num_classes = config["num_classes"]
    num_epochs = config["num_epochs"]
    optimizer_name = config["optimizer_name"]
    lr = config["lr"]

    exp_dir = config["output_path"]
    os.makedirs(exp_dir, exist_ok=True)

    # ---------------- DATA ----------------
    print("[LOADING DATASET PHASE]")
    if dataset_name == "MNIST":
        print("[LOADING DATASET] Loading Handwritten MNIST dataset ...")
        train_loader, val_loader, test_loader = get_mnist_loader(data_dir = dataset_path, batch_size = batch_size)
        print("[LOADING DATASET] Loading successfully Handwritten MNIST dataset")
    elif dataset_name == "Fashion-MNIST":
        print("[LOADING DATASET] Loading Fashion MNIST dataset ...")
        train_loader, val_loader, test_loader = get_fashion_mnist_loader(data_dir = dataset_path, batch_size = batch_size)
        print("[LOADING DATASET] Loading successfully Fashion MNIST dataset")
    elif dataset_name == "Medical-MNIST":
        print("[LOADING DATASET] Loading Medical MNIST dataset ...")
        train_loader, val_loader, test_loader = get_medical_mnist_loader(data_dir = dataset_path, batch_size = batch_size)
        print("[LOADING DATASET] Loading successfully Medical MNIST dataset")
    else:
        raise ValueError(f"Dataset name {dataset_name} is not found!")

    # ---------------- MODEL ----------------
    print("[LOADING MODEL PHASE]")
    if model_name == "ANN2Layers":
        model = ANN2Layers(input_size = input_size, num_classes = num_classes)
        print("[LOADING MODEL] Successfully load model ANN 2-Hidden Layers")
    elif model_name == "ANN5Layers":
        model = ANN5Layers(input_size = input_size, num_classes = num_classes) 
        print("[LOADING MODEL] Successfully load model ANN 5-Hidden Layers")

    # ------------- LOAD WEIGHTS --------
    pretrained_path = exp_dir + "/best_model.pth"
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model = model.to(device)

    # ------------- TEST ------------------
    test_acc = test(model, test_loader, device)

    print("=" * 30)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("=" * 30)

    # ------------- SAVE RESULT ------------------
    result_txt_path = os.path.join(exp_dir, "test_results.txt")

    with open(result_txt_path, "w") as f:
        f.write("===== TEST RESULTS =====\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Optimizer: {optimizer_name}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write("\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")

    print(f"[SAVED] Test results saved to {result_txt_path}")