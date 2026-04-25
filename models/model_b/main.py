"""CLI entry point for damage detector training."""
import argparse
import os
from pathlib import Path

from .config import TrainConfig
from .data import build_dataloaders
from .evaluate import evaluate_model
from .model import build_model
from .train import train_model
from .utils import get_device, set_seed


def parse_args() -> TrainConfig:
    """Parse command-line arguments.

    Returns:
        TrainConfig object.
    """
    root_dir = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Train ViT damage detector.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root_dir / "data" / "model b" / "damage_assessment",
        help=(
            "Dataset root. Supported layouts: "
            "(1) train/valid folders with Whole/Damaged subfolders, or "
            "(2) damage_assessment format (data/ + samples.json) with Whole images "
            "available in whole_pool/, Whole/, train/Whole/, or valid/Whole/."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=root_dir / "weights" / "model b" / "best_damage_detector.pth",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recall-weight", type=float, default=2.0)

    args = parser.parse_args()
    return TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        model_output=args.model_output,
        seed=args.seed,
        recall_weight=args.recall_weight,
    )


def main() -> None:
    """Main training pipeline."""
    config = parse_args()

    # On Windows, multiprocessing workers can fail under mixed Python envs.
    if os.name == "nt" and config.num_workers > 0:
        print("WARNING: Forcing num_workers=0 on Windows for stable DataLoader execution.")
        config.num_workers = 0

    set_seed(config.seed)

    if not config.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {config.data_dir}")

    device = get_device()
    print(f"Using device: {device}")

    print("\n=== Loading Data ===")
    train_loader, val_loader, processor = build_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    print("\n=== Building Model ===")
    model = build_model().to(device)

    print("\n=== Training (Recall-Focused) ===")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        device=device,
        output_path=config.model_output,
        recall_weight=config.recall_weight,
    )

    print("\n=== Validation Set Evaluation ===")
    evaluate_model(
        model=model,
        loader=val_loader,
        device=device,
    )

    print("\n=== Inference Example ===")
    print(
        "from models.model_b.inference import classify_damage\n"
        "class_name, confidence = classify_damage('path/to/car_image.jpg', "
        f"model_path='{config.model_output}')\n"
        "print(f'Status: {class_name}, Confidence: {confidence:.2%}')"
    )

    print("\n=== Multi-View Inspection Example ===")
    print(
        "from models.model_b.inspection import inspect_vehicle_sync\n"
        "images = ['front.jpg', 'back.jpg', 'left.jpg', 'right.jpg', 'roof.jpg']\n"
        "verdict, results = inspect_vehicle_sync(images, "
        f"model_path='{config.model_output}')\n"
        "print(f'Verdict: {verdict}')"
    )


if __name__ == "__main__":
    main()
