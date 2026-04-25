"""Training loop and optimization utilities."""
import copy
from typing import Dict, List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one epoch of training or validation.

    Args:
        model: PyTorch model.
        loader: DataLoader for the epoch.
        criterion: Loss criterion.
        optimizer: Optimizer (None for validation).
        device: Torch device.

    Returns:
        Tuple of (epoch_loss, epoch_accuracy).
    """
    is_train = optimizer is not None
    model.train(mode=is_train)

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    class_names: List[str],
    output_path,
) -> Dict[str, List[float]]:
    """Train the model for specified epochs.

    Args:
        model: PyTorch model.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        epochs: Number of epochs.
        learning_rate: Learning rate for optimizer.
        device: Torch device.
        class_names: List of class names.
        output_path: Path to save best model.

    Returns:
        Dictionary with training history.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            checkpoint = {
                "model_state_dict": best_state,
                "class_names": class_names,
                "best_val_acc": best_val_acc,
            }
            torch.save(checkpoint, output_path)

    model.load_state_dict(best_state)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {output_path}")
    return history
