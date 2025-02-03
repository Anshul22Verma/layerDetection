import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import time
from typing import List
from tqdm import tqdm

from model.contrastive_base import ContrastiveModelWrapper, FineTuneContrastiveBaseModel
from loader.contrastive_loader import ContrastiveDataset
from loader.classification_loader import ClassificationDataset
from utils.loss import NTXentLoss
from utils.augment import basic_classification_augmentation


def pre_train_epochs(
    model, train_loader, 
    optimizer, criterion,
    writer, device, epochs: int=100,
):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x1, x2 in tqdm(train_loader, desc=f"Pre-training epoch {epoch}", total=len(train_loader)):
            x1, x2 = x1.to(device), x2.to(device)

            optimizer.zero_grad()
            z1, z2 = model(x1), model(x2)
            loss = criterion(z1, z2)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")
    print("Pretraining complete! Model saved.")
    return model


def pre_train_model(
    architecture: str, embedding_dim: int, pretrained: bool,
    images: List, batch_size: int, num_workers: int, model_dir: str,
    lr: float, temperature: float = 0.5, 
    epochs: int = 100,
    model_name: str = 'base.pth'
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ContrastiveDataset(image_paths=images)
    # use drop-last to drop the incomplete batch
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    log_dir = os.path.join(model_dir, "runs", f"SimCLR_{architecture}_{int(time.time())}")
    writer = SummaryWriter(log_dir)

    model = ContrastiveModelWrapper(
        architecture=architecture, embedding_dim=embedding_dim, pretrained=pretrained
    )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = NTXentLoss(temperature=temperature)

    model = pre_train_epochs(model=model, train_loader=train_loader, optimizer=optimizer, 
                             criterion=criterion, writer=writer, device=device, epochs=epochs)
    torch.save(model, os.path.join(model_dir, model_name))
    writer.close()
    return model


# ---- Fine-Tuning The pre-trained model ---- #
def fine_tune_model(
    model, train_loader: DataLoader, val_loader: DataLoader, 
    optimizer, criterion, writer, device, 
    epochs: int=10, model_path: str = "ft_model.pth"
):
    model.to(device)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i, data in tqdm(enumerate(train_loader), desc=f"Training epoch {epoch}", total=len(train_loader)):
            x, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Log to TensorBoard
        train_loss /= len(train_loader)
        writer.add_scalar(f"{model.name}/Train Loss", train_loss, epoch)
        writer.add_scalar("Fine-Tuning Loss", loss.item(), epoch * len(train_loader) + i)
        
        # ---- Validation ---- #
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, labels in tqdm(val_loader, desc=f"Validating epoch {epoch}", total=len(val_loader)):
                x, labels = x.to(device), labels.to(device)
                outputs = model(x)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5  # Multi-label thresholding
                correct += (preds == labels).sum().item()
                total += labels.numel()
        val_loss /= len(val_loader)
        val_acc = correct / total
        writer.add_scalar(f"{model.name}/Validation Loss", val_loss, epoch)
        writer.add_scalar(f"{model.name}/Validation Accuracy", val_acc, epoch)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}")
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, model_path)
    return model


def train_classifier_w_pretraining(
    pre_trained_model: ContrastiveModelWrapper, num_classes: int, 
    image_paths: List, labels: List, uq_classes: List,
    lr: float, batch_size: int,
    epochs: int, model_dir: str

):
    """Train model with self-supervised learning, then fine-tune for classification."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10  # Change this based on your dataset
    pretrained_model = pre_trained_model
    model = FineTuneContrastiveBaseModel(pretrained_model, num_classes).to(device)

    log_dir = os.path.join(model_dir, "runs", f"ft_{pre_train_model.name}_{int(time.time())}")
    writer = SummaryWriter(log_dir)
    
    transform = basic_classification_augmentation()
    dataset = ClassificationDataset(image_paths=image_paths, labels=labels, uq_classes=uq_classes, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # Multi-label classification loss

    fine_tune_model(model=model, train_loader=train_loader, val_loader=val_loader, 
                    optimizer=optimizer, criterion=criterion,
                    writer=writer, device=device, epochs=epochs,
                    model_path=os.path.join(model_dir, f"ft_{pre_trained_model.name}.pth"))
    model = torch.load(os.path.join(model_dir, f"ft_{pre_trained_model.name}.pth"), weights_only=False)
    return model
