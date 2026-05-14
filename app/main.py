from __future__ import annotations

import asyncio
import concurrent.futures
import json
import pathlib
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .detector import detect_plate, annotate_image
from .recognizer import get_recognizer

# ---------------------------------------------------------------------------
# Thread pool — single worker to avoid OOM on low-RAM servers
# ---------------------------------------------------------------------------
_inference_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


# ---------------------------------------------------------------------------
# Gallery helpers
# ---------------------------------------------------------------------------
def _read_gallery() -> list[dict]:
    try:
        if config.GALLERY_JSON.exists():
            return json.loads(config.GALLERY_JSON.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _append_gallery(record: dict) -> None:
    records = _read_gallery()
    records.append(record)
    # Keep only last N records (e.g. 500) to avoid unbounded growth
    if len(records) > 500:
        records = records[-500:]
    config.GALLERY_JSON.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _prune_gallery() -> None:
    """Remove gallery records whose annotated file no longer exists."""
    records = _read_gallery()
    if not records:
        return
    before = len(records)
    records = [r for r in records
               if not r.get("annotated_url")
               or (config.PROCESSED_DIR / pathlib.Path(r["annotated_url"]).name).exists()]
    if len(records) != before:
        config.GALLERY_JSON.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Background cleanup
# ---------------------------------------------------------------------------
async def periodic_cleanup():
    """Delete annotated images older than ANNOTATED_FILE_CLEANUP_AGE.
    Skips gallery records and original images."""
    while True:
        await asyncio.sleep(config.CLEANUP_INTERVAL)
        try:
            now = time.time()
            for f in config.PROCESSED_DIR.iterdir():
                if not f.is_file():
                    continue
                # Never delete gallery metadata or original uploads
                if f.name == config.GALLERY_JSON.name or "_orig" in f.name:
                    continue
                if (now - f.stat().st_mtime) > config.ANNOTATED_FILE_CLEANUP_AGE:
                    f.unlink(missing_ok=True)
            # Also prune gallery records pointing to cleaned annotated images
            _prune_gallery()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    config.PROCESSED_DIR.mkdir(exist_ok=True)

    cleanup_task = asyncio.create_task(periodic_cleanup())

    # Pre-warm the recognition model
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: get_recognizer().initialize())

    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Chinese License Plate Recognition", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
app.mount("/processed", StaticFiles(directory=str(config.PROCESSED_DIR)), name="processed")


# ---------------------------------------------------------------------------
# Inference helper (runs in thread pool)
# ---------------------------------------------------------------------------
def _process_image_bytes(image_bytes: bytes) -> dict:
    start = time.perf_counter()

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        elapsed = (time.perf_counter() - start) * 1000
        return {"success": False, "error": "无法解码图片，请上传有效的图片文件",
                "processing_time_ms": round(elapsed, 1)}

    detected, warped_plate, box_points, candidate, reason = detect_plate(img)

    if not detected or warped_plate is None:
        elapsed = (time.perf_counter() - start) * 1000
        return {"success": True, "detected": False,
                "plate_number": None, "confidence": 0.0,
                "annotated_image_url": None,
                "processing_time_ms": round(elapsed, 1),
                "error": reason}

    recognizer = get_recognizer()
    plate_text, confidence = recognizer.recognize(warped_plate)

    if not plate_text or confidence < 0.3:
        elapsed = (time.perf_counter() - start) * 1000
        err_msg = f"已定位车牌，但字符识别置信度过低 ({confidence:.1%})" if confidence > 0 else "已定位车牌区域，但未能识别出字符"
        return {"success": True, "detected": False,
                "plate_number": None, "confidence": round(confidence, 4),
                "annotated_image_url": None,
                "processing_time_ms": round(elapsed, 1),
                "error": err_msg}

    unique_name = (f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                   f"{uuid.uuid4().hex[:8]}")
    annotated_name = f"{unique_name}_annotated.jpg"
    original_name = f"{unique_name}_orig.jpg"
    annotated_path = config.PROCESSED_DIR / annotated_name
    original_path = config.PROCESSED_DIR / original_name

    # Save original
    cv2.imwrite(str(original_path), img)

    # Save annotated
    annotate_image(img, box_points, plate_text, annotated_path)

    # Append to gallery
    timestamp = datetime.now().isoformat(timespec="seconds")
    _append_gallery({
        "plate_number": plate_text,
        "confidence": round(confidence, 4),
        "timestamp": timestamp,
        "annotated_url": f"/processed/{annotated_name}",
        "original_url": f"/processed/{original_name}",
    })

    elapsed = (time.perf_counter() - start) * 1000
    return {
        "success": True,
        "detected": True,
        "plate_number": plate_text,
        "confidence": round(confidence, 4),
        "annotated_image_url": f"/processed/{annotated_name}",
        "processing_time_ms": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def read_root():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = pathlib.Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(400, detail=f"不支持的图片格式: {ext}。"
                            f"支持的格式: {', '.join(config.ALLOWED_EXTENSIONS)}")

    contents = await file.read()
    if len(contents) > config.MAX_UPLOAD_SIZE:
        raise HTTPException(400, detail=f"文件过大。最大支持: "
                            f"{config.MAX_UPLOAD_SIZE // (1024 * 1024)} MB")
    if len(contents) == 0:
        raise HTTPException(400, detail="上传的文件为空")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_inference_executor, _process_image_bytes, contents)

    if not result.get("success"):
        return JSONResponse(status_code=422, content=result)

    return result


@app.get("/gallery")
async def get_gallery():
    """Return gallery records sorted newest first."""
    records = await asyncio.get_event_loop().run_in_executor(
        None, _read_gallery)
    records.reverse()  # newest first
    return records
