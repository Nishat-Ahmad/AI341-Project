---
title: Fleet-Vision
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

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

## Model Documentation

For detailed Model A and Model B architectures, hyperparameters, evaluation metrics, and training guides, see [models/README.md](models/README.md).

## Notes

- Model weights are stored with Git LFS under `weights/`.
- Dataset folders live under `data/`.

## CI/CD: GitHub -> Hugging Face (Docker Space)

This repository includes a Docker-based CI/CD flow:

- CI workflow: `.github/workflows/ci.yml`
	- Builds the Docker image on GitHub Actions.
	- Runs a smoke test against `GET /health`.
- Deploy workflow: `.github/workflows/deploy-hf-space.yml`
	- Syncs repository content to a Hugging Face Space repo.
	- Hugging Face then builds and runs the app from `Dockerfile`.

### Required GitHub secrets

Set these in your GitHub repository settings -> Secrets and variables -> Actions:

- `HF_TOKEN`: Hugging Face user access token with write permission to Spaces.
- `HF_SPACE_REPO`: Space identifier in the form `username/space-name`.

### Required Hugging Face Space secret

Set this in your Hugging Face Space settings -> Variables and secrets:

- `ORS_API_KEY`: OpenRouteService API key used by routing service.

### Deploy behavior

- Any push to `main` triggers CI and deploy workflows.
- Deploy copies files to the Space repository and pushes a commit.
- If there are no file changes, deploy exits without creating a commit.
