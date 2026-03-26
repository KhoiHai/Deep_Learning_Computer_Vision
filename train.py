import os
import json
import csv
import torch

from src.datasets.mnist_loader import get_mnist_loader
from src.datasets.fashion_mnist_loader import get_fashion_mnist_loader
from src.datasets.medical_mnist_loader import get_medical_mnist_loader

from src.trainers.trainer import Trainer
from src.models.ANN import ANN

def main():
    # ---------------- CONFIG ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("./configs/ANN/MNIST_vanilla_sgd.json", "r") as f:
        config = json.load(f)

    dataset_name = config["dataset_name"]
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
        train_loader, val_loader, test_loader = get_mnist_loader("./data", batch_size = batch_size)
        print("[LOADING DATASET] Loading successfully Handwritten MNIST dataset")
    elif dataset_name == "Fashion-MNIST":
        train_loader, val_loader, test_loader = get_fashion_mnist_loader("./data", batch_size = batch_size)
        print("[LOADING DATASET] Loading successfully Fashion MNIST dataset")
    elif dataset_name == "Medical-MNIST":
        train_loader, val_loader, test_loader = get_medical_mnist_loader("./data/MedicalMNIST", batch_size = batch_size)
        print("[LOADING DATASET] Loading successfully Medical MNIST dataset")
    else:
        raise ValueError(f"Dataset name {dataset_name} is not found!")

    # ---------------- MODEL ----------------
    print("[LOADING MODEL PHASE]")
    model = ANN(input_size=input_size, num_classes=num_classes)
    print("[LOADING MODEL] Successfully load model")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        optimizer=optimizer_name,
        lr=lr,
        device=device
    )
    print("[TRAINING PHASE]")

    # ---------------- CSV LOG ----------------
    csv_path = os.path.join(exp_dir, "log.csv")

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_acc"])

    # ---------------- TRAIN ----------------
    best_val_acc = 0

    for epoch in range(num_epochs):
        train_loss = trainer.train_one_epoch()
        val_acc = trainer.validate()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Log CSV
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, val_acc])

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(trainer.model.state_dict(), os.path.join(exp_dir, "best_model.pth"))

    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results saved in {exp_dir}")

if __name__ == "__main__":
    main()