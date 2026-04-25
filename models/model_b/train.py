"""Training loop with recall-focused optimization."""
import copy
from typing import Dict, List, Tuple

import torch
from sklearn.metrics import precision_recall_curve
from torch import nn, optim
from torch.utils.data import DataLoader


def calculate_class_weights(loader: DataLoader) -> torch.Tensor:
    """Calculate class weights to balance dataset.

    Args:
        loader: DataLoader to analyze.

    Returns:
        Class weights tensor.
    """
    class_counts = torch.zeros(2)
    for _, labels in loader:
        class_counts += torch.bincount(labels, minlength=2).float()

    weights = 1.0 / (class_counts + 1e-8)
    weights = weights / weights.sum()
    return weights


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    recall_weight: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Run one epoch with recall tracking.

    Args:
        model: ViT model.
        loader: DataLoader.
        criterion: Loss function.
        optimizer: Optimizer (None for validation).
        device: Device.
        recall_weight: Weight for recall loss.

    Returns:
        Tuple of (loss, accuracy, precision, recall).
    """
    is_train = optimizer is not None
    model.train(mode=is_train)

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(pixel_values)
            logits = outputs.logits
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        running_loss += loss.item() * pixel_values.size(0)
        predictions = logits.argmax(dim=1)

        all_preds.extend(predictions.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate metrics
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    accuracy = (all_preds == all_labels).float().mean().item()

    # Calculate precision and recall
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return epoch_loss, accuracy, precision, recall


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    output_path,
    recall_weight: float = 2.0,
) -> Dict[str, List[float]]:
    """Train model with recall optimization.

    Args:
        model: ViT model.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        epochs: Number of epochs.
        learning_rate: Learning rate.
        device: Device.
        output_path: Path to save best model.
        recall_weight: Weight for recall vs accuracy trade-off.

    Returns:
        Training history.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=1e-5,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "train_recall": [],
        "val_loss": [],
        "val_acc": [],
        "val_recall": [],
    }

    best_val_recall = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_prec, train_recall = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            recall_weight=recall_weight,
        )

        val_loss, val_acc, val_prec, val_recall = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_recall"].append(train_recall)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_recall"].append(val_recall)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Recall: {train_recall:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Recall: {val_recall:.4f}"
        )

        # Save best model based on Recall (minimize False Negatives)
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_state = copy.deepcopy(model.state_dict())
            checkpoint = {
                "model_state_dict": best_state,
                "best_val_recall": best_val_recall,
                "best_val_acc": val_acc,
            }
            torch.save(checkpoint, output_path)
            print(f"✓ Saved best model (Recall: {best_val_recall:.4f})")

    model.load_state_dict(best_state)
    print(f"\nBest validation recall: {best_val_recall:.4f}")
    print(f"Best model saved to: {output_path}")
    return history
