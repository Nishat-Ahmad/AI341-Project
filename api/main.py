"""Fleet-Vision FastAPI server."""
from __future__ import annotations

import shutil
import os
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

# Ensure logs dir exists and write a startup marker before heavy initialization
logs_dir = outputs_dir / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
ts_start = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
startup_marker = logs_dir / f"server_start_{ts_start}.log"
with startup_marker.open("w", encoding="utf-8") as f:
    f.write(f"Server process starting at {ts_start}\n")

# Instantiate orchestrator inside try/except so startup exceptions are captured
try:
    orchestrator = FleetVisionOrchestrator()
except Exception as exc:
    tb = traceback.format_exc()
    err_path = logs_dir / f"startup_traceback_{ts_start}.log"
    with err_path.open("w", encoding="utf-8") as ef:
        ef.write(f"Exception during orchestrator init: {exc}\n\n")
        ef.write(tb)
    # Re-raise so the runtime logs also capture the error
    raise


@app.get("/debug/logs")
async def list_logs() -> dict:
    """Return a list of log filenames under the outputs/logs directory."""
    logs_dir = outputs_dir / "logs"
    if not logs_dir.exists():
        return {"logs": []}
    files = []
    for p in sorted(logs_dir.iterdir(), key=lambda x: x.name):
        if p.is_file():
            files.append({"name": p.name, "size": p.stat().st_size})
    return {"logs": files}


@app.get("/debug/logs/{name}")
async def get_log(name: str) -> FileResponse:
    """Serve a specific log file from outputs/logs safely."""
    safe_name = Path(name).name
    log_path = outputs_dir / "logs" / safe_name
    if not log_path.exists() or not log_path.is_file():
        raise HTTPException(status_code=404, detail="Log not found")
    return FileResponse(log_path)


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

        # Optionally persist input images for debugging (set FLEET_SAVE_INPUTS=true)
        try:
            if os.getenv("FLEET_SAVE_INPUTS", "false").lower() in ("1", "true", "yes"):
                inputs_dir = outputs_dir / "debug_inputs"
                ts_debug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                dest_dir = inputs_dir / ts_debug
                dest_dir.mkdir(parents=True, exist_ok=True)
                for angle, pth in image_paths.items():
                    try:
                        shutil.copy(pth, dest_dir / f"{angle}_{Path(pth).name}")
                    except Exception:
                        # Don't fail the request if saving debug inputs fails
                        pass
        except Exception:
            pass

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
