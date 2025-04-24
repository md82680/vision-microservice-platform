"""
Model Trainer
Handles the training of machine learning models
"""
import os
import logging
import time
import asyncio
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = os.getenv("DATA_PATH", "./data")
os.makedirs(DATA_PATH, exist_ok=True)

# Define CIFAR-10 classes
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def get_model(model_type: str, num_classes: int = 10) -> nn.Module:
    """
    Create and initialize a PyTorch model
    
    Args:
        model_type: Type of model to create (e.g. "resnet18")
        num_classes: Number of output classes
        
    Returns:
        Initialized PyTorch model
    """
    if model_type == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "resnet34":
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def get_cifar10_dataloaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 training and validation dataloaders
    
    Args:
        batch_size: Batch size for training
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),  # Resize to match ResNet input
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(224),  # Resize to match ResNet input
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=DATA_PATH, 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR10(
        root=DATA_PATH, 
        train=False, 
        download=True, 
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_epoch(model: nn.Module, train_loader: DataLoader, 
               criterion: nn.Module, optimizer: optim.Optimizer, 
               device: torch.device) -> Dict[str, float]:
    """
    Train the model for one epoch
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    # Calculate metrics
    train_loss = running_loss / total
    train_acc = correct / total
    
    return {
        "loss": train_loss,
        "accuracy": train_acc
    }

def validate(model: nn.Module, val_loader: DataLoader, 
            criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    """
    Validate the model
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Calculate metrics
    val_loss = running_loss / total
    val_acc = correct / total
    
    return {
        "loss": val_loss,
        "accuracy": val_acc
    }

async def train_cifar10_model(
    model_type: str = "resnet18",
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    save_path: str = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Train a model on CIFAR-10 dataset
    
    Args:
        model_type: Type of model to train
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        save_path: Path to save the trained model
        
    Returns:
        Tuple of (model_path, metrics)
    """
    # Use asyncio to simulate non-blocking behavior in FastAPI
    loop = asyncio.get_event_loop()
    
    def _train():
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Create model
        model = get_model(model_type)
        model = model.to(device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Get dataloaders
        train_loader, val_loader = get_cifar10_dataloaders(batch_size)
        
        # Training loop
        metrics = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "best_val_acc": 0.0,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_type": model_type,
            "total_params": sum(p.numel() for p in model.parameters())
        }
        
        start_time = time.time()
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Train
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_metrics = validate(model, val_loader, criterion, device)
            
            # Save metrics
            metrics["train_loss"].append(train_metrics["loss"])
            metrics["train_acc"].append(train_metrics["accuracy"])
            metrics["val_loss"].append(val_metrics["loss"])
            metrics["val_acc"].append(val_metrics["accuracy"])
            
            # Update best validation accuracy
            if val_metrics["accuracy"] > metrics["best_val_acc"]:
                metrics["best_val_acc"] = val_metrics["accuracy"]
            
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
        
        # Calculate training time
        total_time = time.time() - start_time
        metrics["training_time"] = total_time
        
        # Save model
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")
        else:
            save_path = f"./models/trained/{model_type}_cifar10.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")
        
        return save_path, metrics
    
    # Run the training function in a separate thread to not block the event loop
    return await loop.run_in_executor(None, _train) 