"""CLI entry point for training the body type classifier."""
import argparse
from pathlib import Path

from .config import TrainConfig
from .data import build_dataloaders
from .evaluate import evaluate_model
from .inference import classify_car_type
from .model import build_model
from .train import train_model
from .utils import get_device, set_seed


def parse_args() -> TrainConfig:
    """Parse command-line arguments.

    Returns:
        TrainConfig object with parsed arguments.
    """
    root_dir = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Train a ResNet-50 car body type classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root_dir / "data" / "raw" / "body_types",
        help="Path to ImageFolder dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path(__file__).resolve().parent / "best_body_classifier.pth",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        model_output=args.model_output,
        seed=args.seed,
    )


def main() -> None:
    """Main training pipeline."""
    config = parse_args()
    set_seed(config.seed)

    if not config.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {config.data_dir}")

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = build_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        val_split=config.val_split,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    model = build_model(num_classes=len(class_names)).to(device)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        device=device,
        class_names=class_names,
        output_path=config.model_output,
    )

    evaluate_model(
        model=model,
        loader=val_loader,
        class_names=class_names,
        device=device,
    )

    print("\nInference example:")
    print(
        "pred_class, confidence = classify_car_type('path/to/car_image.jpg', "
        f"model_path='{config.model_output}')"
    )


if __name__ == "__main__":
    main()
