import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import time
from typing import List
torch.manual_seed(2025)

from loader.classification_loader import ClassificationDataset
from model.base_multilabel import PretrainedModelWrapper
from utils.augment import basic_classification_augmentation


# ---- Fine-Tuning Function ---- #
def fine_tune_model(
        model, 
        train_loader, val_loader,
        criterion, optimizer, 
        writer, device, 
        epochs: int = 10,
        model_path: str = 'model.pth'):
    """Fine-tunes a pretrained model for classification."""
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


def train_classifier(
    architecture: str, num_classes: int, 
    image_paths: List, labels: List, uq_classes: List,
    lr: float, batch_size: int,
    epochs: int, model_dir: str

):
    """Train model with self-supervised learning, then fine-tune for classification."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PretrainedModelWrapper(architecture=architecture, num_classes=num_classes, pretrained=True)
    
    log_dir = os.path.join(model_dir, "runs", f"{architecture}_{int(time.time())}")
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
                    model_path=os.path.join(model_dir, f"{architecture}.pth"))
    writer.close()
    model = torch.load(os.path.join(model_dir, f"direct_ft_{architecture}.pth"), weights_only=False)
    return model
