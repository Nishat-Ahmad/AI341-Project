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

A Swin-based binary damage detector that classifies a vehicle image as either `Whole` or `Damaged` and can also produce a Grad-CAM heatmap for damaged results.

### Architecture

- **Base Model**: `microsoft/swin-tiny-patch4-window7-224`
- **Backbone**: Frozen transformer backbone
- **Custom Head**: Hugging Face image classification head with 2 output labels
- **Loss**: CrossEntropyLoss
- **Optimizer**: AdamW
- **Training Focus**: Recall-oriented checkpoint selection to reduce missed damage cases

### Supported Classes

1. Whole
2. Damaged

### Dataset

The training pipeline supports two layouts:

```
data/model b/
├── train/
│   ├── Whole/
│   └── Damaged/
└── valid/
    ├── Whole/
    └── Damaged/
```

It can also read a `damage_assessment` layout with `samples.json` and a `data/` folder, plus optional `whole_pool/`, `Whole/`, or `train/Whole` / `valid/Whole` negatives.

### Performance

Model B is tuned primarily for damage detection recall, so the best checkpoint is saved when validation recall improves. This is intended to avoid false negatives where a damaged vehicle is incorrectly treated as safe.

### Quick Start

#### Train
```bash
cd d:\Code\FleetThing
python models\model_b\main.py --epochs 15 --batch-size 32 --recall-weight 2.0
```

#### Custom Parameters
```bash
python models\model_b\main.py \
  --epochs 20 \
  --batch-size 16 \
  --learning-rate 0.0001 \
  --recall-weight 2.0
```

#### Inference
```python
from models.model_b.inference import classify_damage

status, confidence = classify_damage(
    'path/to/car_image.jpg',
    model_path='weights/model b/best_damage_detector.pth'
)
print(f"Status: {status}, Confidence: {confidence:.2%}")
```

#### Heatmap Generation
```python
from models.model_b.grad_cam import generate_damage_heatmap

heatmap_path, heatmap = generate_damage_heatmap(
    image_path='path/to/car_image.jpg',
    model_path='weights/model b/best_damage_detector.pth',
)
print(heatmap_path)
```

### Module Structure

```
model_b/
├── __init__.py          # Package exports
├── config.py            # Constants & TrainConfig
├── data.py              # Datasets, splits, and transforms
├── model.py             # Swin image classifier
├── train.py             # Training loop
├── evaluate.py          # Metrics & evaluation
├── inference.py         # Inference & model loading
├── grad_cam.py          # Heatmap generation
├── inspection.py        # Multi-angle inspection helpers
└── main.py              # CLI entry point
```

### Weights

Trained model checkpoint: `weights/model b/best_damage_detector.pth`

Contains:
- `model_state_dict`
- `best_val_recall`
- `best_val_acc`

### Requirements

- PyTorch
- torchvision
- transformers
- scikit-learn (for metrics)
- Pillow (for image loading)
- OpenCV (for heatmap rendering)

Install: `pip install -r requirements.txt`

---

## Adding New Models

1. Create folder: `models/model_x/`
2. Add `__init__.py`, `config.py`, `data.py`, `model.py`, `train.py`, `evaluate.py`, `inference.py`, `main.py`
3. Follow Model A structure and patterns
4. Update this README with results and usage
