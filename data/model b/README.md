# Model B Training Data

This directory contains training data for the Car Damage Detector (Model B) using Vision Transformer.

## Required Structure

```
model b/
├── train/
│   ├── Whole/          # Images of undamaged cars
│   └── Damaged/        # Images of damaged cars
└── valid/
    ├── Whole/          # Validation images of undamaged cars
    └── Damaged/        # Validation images of damaged cars
```

## Data Requirements

- **Image Format**: JPG or PNG
- **Image Size**: Any size (ViTImageProcessor will resize to 224x224)
- **Minimum Images per Set**: 10+ images (preferably 50+ for better training)
- **Class Balance**: Try to keep roughly equal numbers of Whole and Damaged images

## Directory Setup

The following directories have been created:
- `train/Whole/`
- `train/Damaged/`
- `valid/Whole/`
- `valid/Damaged/`

## Adding Training Data

1. **For Whole (Undamaged) Cars**: Place images of vehicles with no visible damage
2. **For Damaged Cars**: Place images of vehicles with visible damage (dents, scratches, broken parts, etc.)
3. **Training/Validation Split**: Put ~80% of your data in `train/` and ~20% in `valid/`

## Starting Training

Once you have added images to the directories, run:

```bash
python run_damage_training.py --epochs 15 --batch-size 32 --recall-weight 2.0
```

The trained model will be saved to: `weights/model b/best_damage_detector.pth`

## Demo Mode

For testing purposes, placeholder images have been created using sample data from Model A.
These demo images allow you to verify the training pipeline works before adding real damage detection data.
