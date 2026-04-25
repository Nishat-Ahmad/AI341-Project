# Models

This directory contains all machine learning models for Fleet-Vision autonomous vehicle inspection and dispatch system.

## Model A: Car Body Type Classifier

A ResNet-50 transfer learning model that classifies vehicle body types with 85.3% validation accuracy and 83% test accuracy.

### Architecture

- **Base Model**: ResNet-50 (pretrained on ImageNet)
- **Backbone**: Frozen (no gradient updates)
- **Custom Head**: 
  - Linear(2048 → 512)
  - ReLU + Dropout(0.3)
  - Linear(512 → 7)
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)

### Supported Classes

1. Convertible
2. Coupe
3. Hatchback
4. Pick-Up
5. Sedan
6. SUV
7. VAN

### Dataset

```
data/model a/
├── train/          (5,350 images)
├── valid/          (1,397 images)
└── test/           (802 images)
```

### Performance

**Validation Set (1,397 images)**
- Accuracy: 85.33%
- Precision: 85.56%
- Recall: 85.33%
- F1-Score: 85.32%

**Test Set (802 images)**
- Accuracy: 83.04%
- Precision: 83.33%
- Recall: 83.04%
- F1-Score: 82.97%

**Per-Class F1 Scores (Validation)**
- Convertible: 0.96
- Coupe: 0.72
- Hatchback: 0.75
- Pick-Up: 0.89
- Sedan: 0.76
- SUV: 0.82
- VAN: 0.97

### Quick Start

#### Train
```bash
cd d:\Code\FleetThing
python models\model_a\main.py --epochs 10 --batch-size 32
```

#### Custom Parameters
```bash
python models\model_a\main.py \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.0005
```

#### Inference
```python
from models.model_a.inference import classify_car_type

pred_class, confidence = classify_car_type(
    'path/to/car_image.jpg',
    model_path='weights/model a/best_body_classifier.pth'
)
print(f"Predicted: {pred_class}, Confidence: {confidence:.2%}")
```

### Module Structure

```
model_a/
├── __init__.py          # Package exports
├── config.py            # Constants & TrainConfig
├── data.py              # DataLoaders & transforms
├── model.py             # ResNet-50 architecture
├── train.py             # Training loop
├── evaluate.py          # Metrics & evaluation
├── inference.py         # Inference & model loading
├── utils.py             # Device & seed utilities
└── main.py              # CLI entry point
```

### Weights

Trained model checkpoint: `weights/model a/best_body_classifier.pth`

Contains:
- model_state_dict
- class_names
- best_val_acc

### Requirements

- PyTorch (CUDA-enabled)
- torchvision
- scikit-learn (for metrics)
- Pillow (for image loading)

Install: `pip install -r requirements.txt`

---

## Model B: Damage Assessment Classifier

*(Coming soon)*

---

## Adding New Models

1. Create folder: `models/model_x/`
2. Add `__init__.py`, `config.py`, `data.py`, `model.py`, `train.py`, `evaluate.py`, `inference.py`, `main.py`
3. Follow Model A structure and patterns
4. Update this README with results and usage
