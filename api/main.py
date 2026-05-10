"""Fleet-Vision FastAPI server."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
import traceback
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from langchain_orchestrator import FleetVisionOrchestrator
from utils.routing_service import RoutingServiceError

app = FastAPI(title="Fleet-Vision API", version="1.0.0")

root_dir = Path(__file__).resolve().parents[1]
static_dir = root_dir / "app"
outputs_dir = root_dir / "outputs"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Ensure runtime-generated heatmaps are always served in Docker/Spaces, even if
# the repository did not include an empty `outputs/` directory at build time.
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")

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
    start_location: str = Form(...),
    destination: str = Form(...),
) -> dict:
    if not start_location.strip():
        raise HTTPException(status_code=400, detail="Start location cannot be empty.")
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

        # Validate saved images to catch corrupted/unsupported uploads early
        for angle, path_str in image_paths.items():
            try:
                with Image.open(path_str) as im:
                    im.verify()
            except UnidentifiedImageError as img_exc:
                raise ValueError(f"Invalid or corrupted image for {angle}: {img_exc}") from img_exc
            except Exception as img_exc:  # unexpected PIL errors
                raise ValueError(f"Failed to process image for {angle}: {img_exc}") from img_exc

        report = await orchestrator.run(
            image_paths=image_paths,
            start_location=start_location.strip(),
            destination=destination.strip(),
        )
        return report

    except RoutingServiceError as exc:
        raise HTTPException(status_code=502, detail=f"Routing failure: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        try:
            logs_dir = outputs_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            tb_text = traceback.format_exc()
            meta = {"image_paths": image_paths if "image_paths" in locals() else None}
            log_path = logs_dir / f"traceback_{ts}.log"
            with log_path.open("w", encoding="utf-8") as lf:
                lf.write(f"Exception: {exc}\n\n")
                lf.write(f"Meta: {meta}\n\n")
                lf.write(tb_text)
            detail_msg = f"Internal error; traceback written to /outputs/logs/{log_path.name}"
        except Exception:
            detail_msg = "Internal error; failed to write traceback."
        raise HTTPException(status_code=500, detail=detail_msg) from exc
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
