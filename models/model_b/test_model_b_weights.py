#!/usr/bin/env python
"""Validate Model B checkpoint on validation data with PASS/FAIL thresholds."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Test Model B weights on validation data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data" / "model b" / "damage_assessment",
        help="Dataset root used by Model B loader",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=root / "weights" / "model b" / "best_damage_detector.pth",
        help="Checkpoint path",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="If > 0, evaluate only this many validation batches for a fast smoke check",
    )
    parser.add_argument("--min-acc", type=float, default=0.90)
    parser.add_argument("--min-damage-recall", type=float, default=0.85)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for evaluation",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")

    if torch.cuda.is_available():
        try:
            _ = torch.randn(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError:
            print("WARNING: CUDA detected but unusable. Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")


def main() -> int:
    args = parse_args()

    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from models.model_b.data import build_dataloaders
    from models.model_b.inference import load_trained_model

    if not args.data_dir.exists():
        print(f"FAIL: Data directory not found: {args.data_dir}")
        return 1

    if not args.model_path.exists():
        print(f"FAIL: Checkpoint not found: {args.model_path}")
        return 1

    num_workers = args.num_workers
    if os.name == "nt" and num_workers > 0:
        print("WARNING: On Windows, forcing num_workers=0 for stability.")
        num_workers = 0

    train_loader, val_loader, _ = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    try:
        model, _ = load_trained_model(args.model_path, device)
    except RuntimeError as exc:
        if "CUDA" in str(exc) and device.type == "cuda":
            print("WARNING: CUDA load failed; retrying on CPU.")
            device = torch.device("cpu")
            model, _ = load_trained_model(args.model_path, device)
        else:
            raise

    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch_idx, (pixel_values, labels) in enumerate(val_loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            logits = model(pixel_values).logits
            preds = logits.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    if not y_true:
        print("FAIL: No validation samples were evaluated.")
        return 1

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total = len(y_true)
    accuracy = (tp + tn) / max(1, total)
    damage_recall = tp / max(1, (tp + fn))
    damage_precision = tp / max(1, (tp + fp))

    print("\n=== Validation Summary ===")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Damage Recall  : {damage_recall:.4f}")
    print(f"Damage Precision: {damage_precision:.4f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    acc_ok = accuracy >= args.min_acc
    rec_ok = damage_recall >= args.min_damage_recall

    print("\n=== Threshold Check ===")
    print(f"Accuracy >= {args.min_acc:.2f}: {'PASS' if acc_ok else 'FAIL'}")
    print(f"Damage Recall >= {args.min_damage_recall:.2f}: {'PASS' if rec_ok else 'FAIL'}")

    if acc_ok and rec_ok:
        print("\nPASS: Model weights meet requested quality thresholds.")
        return 0

    print("\nFAIL: Model weights do not meet one or more thresholds.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
