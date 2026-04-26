Fleet-Vision
=============

Fleet-Vision is a vehicle inspection and dispatch demo. It combines:

- Model A: vehicle body classification
- Model B: damage detection with heatmaps
- OpenRouteService: route distance and ETA lookup
- FastAPI: upload and dispatch endpoint

## Setup

```bash
git lfs pull
pip install -r requirements.txt
```

Or with Poetry:

```bash
poetry install
```

## Run

Start the API:

```bash
uvicorn api.main:app --reload
```

Open the app in your browser and upload the five required images plus a destination.

## Key files

- [api/main.py](api/main.py): FastAPI app and `/request-ride` endpoint
- [langchain_orchestrator.py](langchain_orchestrator.py): model and routing workflow
- [body_classifier.py](body_classifier.py): Model A wrapper
- [damage_detector.py](damage_detector.py): Model B wrapper
- [utils/routing_service.py](utils/routing_service.py): ORS routing and geocoding

## Notes

- Model weights are stored with Git LFS under `weights/`.
- Dataset folders live under `data/`.
