"""Evaluation metrics and reporting utilities."""
from typing import List

import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    """Evaluate model on validation/test set.

    Args:
        model: Trained ViT model.
        loader: DataLoader for evaluation.
        device: Torch device.
    """
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    confidences: List[float] = []

    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device)
        outputs = model(pixel_values)
        logits = outputs.logits

        predictions = logits.argmax(dim=1).cpu().tolist()
        probs = torch.softmax(logits, dim=1)
        max_probs = probs.max(dim=1).values.cpu().tolist()

        y_true.extend(labels.tolist())
        y_pred.extend(predictions)
        confidences.extend(max_probs)

    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("\n=== Confusion Matrix ===")
    print(cm)
    print(f"[[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f" [FN={cm[1,0]}, TP={cm[1,1]}]]")

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=["Whole", "Damaged"], zero_division=0))

    print("=== Summary Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    # Detailed metrics for damage detection
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]

    print(f"\nDamage Detection Metrics:")
    print(f"True Positives  (Damaged detected):   {tp}")
    print(f"False Positives (Whole as Damaged):   {fp}")
    print(f"False Negatives (Damaged as Whole):   {fn}  ⚠️ CRITICAL")
    print(f"True Negatives  (Whole detected):     {tn}")
