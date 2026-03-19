"""
FastAPI application entry point.

Startup:
  1. Load .env
  2. Build image cache (image_resolver)
  3. Load all XLSX data (data_loader)
  4. Init lock manager (lock_manager)

Routes:
  /api/samples/*     — sample management
  /api/labels/*      — label save/load
  /api/progress      — labeling progress
  /api/admin/reload  — hot-reload XLSX data
  /images/{filename} — serve image files
  /                  — serve static/index.html
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core import data_loader, image_resolver, lock_manager
from routers import labels, samples

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="VQA Labeling Tool", version="1.0.0")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    image_base_dir = os.getenv("IMAGE_BASE_DIR", "~/LMUData/images")
    lock_timeout = int(os.getenv("LOCK_TIMEOUT_SECONDS", "300"))

    logger.info(f"Building image cache from {image_base_dir}")
    image_resolver.build_cache(image_base_dir)

    logger.info("Loading XLSX data…")
    data_loader.load_all()

    logger.info(f"Initializing lock manager (timeout={lock_timeout}s)")
    lock_manager.init(timeout_seconds=lock_timeout)

    Path("labels").mkdir(exist_ok=True)
    logger.info("Server ready")


# ---------------------------------------------------------------------------
# API routers
# ---------------------------------------------------------------------------

app.include_router(samples.router, prefix="/api")
app.include_router(labels.router, prefix="/api")


# ---------------------------------------------------------------------------
# Image serving
# ---------------------------------------------------------------------------

@app.get("/images/{filename:path}")
async def serve_image(filename: str):
    # Use only the basename to prevent path traversal
    safe_name = Path(filename).name
    abs_path = image_resolver.resolve(safe_name)
    if abs_path is None or not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail=f"Image '{safe_name}' not found")
    return FileResponse(abs_path)


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

@app.post("/api/admin/reload")
async def reload_data():
    """Re-read all XLSX files without restarting the server."""
    image_base_dir = os.getenv("IMAGE_BASE_DIR", "~/LMUData/images")
    image_resolver.build_cache(image_base_dir)
    data_loader.load_all()
    total = len(data_loader.list_samples())
    valid = sum(1 for s in data_loader.list_samples().values() if s["is_valid"])
    return {"status": "reloaded", "total": total, "valid": valid}


# ---------------------------------------------------------------------------
# Static files (must come last to avoid catching /api/*)
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"

if STATIC_DIR.exists():
    # Serve /static/* assets (css, js)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        # Return index.html for any unknown path (SPA routing)
        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(STATIC_DIR / "index.html"))
