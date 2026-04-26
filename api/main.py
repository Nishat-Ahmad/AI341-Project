"""Fleet-Vision FastAPI server."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from langchain_orchestrator import FleetVisionOrchestrator
from utils.routing_service import RoutingServiceError

app = FastAPI(title="Fleet-Vision API", version="1.0.0")

root_dir = Path(__file__).resolve().parents[1]
static_dir = root_dir / "app" / "static"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = FleetVisionOrchestrator()


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}


@app.get("/")
async def index() -> FileResponse:
    index_path = static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_path)


@app.post("/request-ride")
async def request_ride(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    roof: UploadFile = File(...),
    destination: str = Form(...),
) -> dict:
    if not destination.strip():
        raise HTTPException(status_code=400, detail="Destination cannot be empty.")

    temp_dir = Path(tempfile.mkdtemp(prefix="fleet_vision_"))

    try:
        uploads = {
            "Front": front,
            "Back": back,
            "Left": left,
            "Right": right,
            "Roof": roof,
        }

        image_paths: dict[str, str] = {}
        for angle, upload in uploads.items():
            if upload is None:
                raise HTTPException(status_code=400, detail=f"Missing required image for {angle}.")
            suffix = Path(upload.filename or f"{angle.lower()}.jpg").suffix or ".jpg"
            output_path = temp_dir / f"{angle.lower()}{suffix}"
            with output_path.open("wb") as f:
                shutil.copyfileobj(upload.file, f)
            image_paths[angle] = str(output_path)

        report = await orchestrator.run(image_paths=image_paths, destination=destination)
        return report

    except RoutingServiceError as exc:
        raise HTTPException(status_code=502, detail=f"Routing failure: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
