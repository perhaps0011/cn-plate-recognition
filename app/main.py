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
# 固定单线程处理推理任务，避免 2 核 1G 服务器上多线程争抢内存导致 OOM
# ---------------------------------------------------------------------------
_inference_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


# ---------------------------------------------------------------------------
# Gallery helpers
# 图库用 JSON 文件做持久化，不依赖数据库
# gallery.json 结构: [{ plate_number, confidence, timestamp, annotated_url,
#                        original_url }, ...]
# ---------------------------------------------------------------------------
def _read_gallery() -> list[dict]:
    """从 JSON 文件读取全部图库记录，返回列表（可能为空）。"""
    try:
        if config.GALLERY_JSON.exists():
            return json.loads(config.GALLERY_JSON.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _append_gallery(record: dict) -> None:
    """追加一条识别记录到图库，超过 500 条则截断旧记录。"""
    records = _read_gallery()
    records.append(record)
    if len(records) > 500:
        records = records[-500:]
    config.GALLERY_JSON.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _prune_gallery() -> None:
    """清理图库中指向已删除标注图的记录（标注图被清理后更新图库）。"""
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
    """定时清理过期文件（asyncio 后台任务）。

    标注图（_annotated.jpg）超过 1 小时后删除，节省磁盘空间；
    原图（_orig.jpg）和图库（gallery.json）保留，不被清理。
    """
    while True:
        await asyncio.sleep(config.CLEANUP_INTERVAL)
        try:
            now = time.time()
            for f in config.PROCESSED_DIR.iterdir():
                if not f.is_file():
                    continue
                # 跳过图库文件和原图——这些永久保留
                if f.name == config.GALLERY_JSON.name or "_orig" in f.name:
                    continue
                if (now - f.stat().st_mtime) > config.ANNOTATED_FILE_CLEANUP_AGE:
                    f.unlink(missing_ok=True)
            # 同步清理图库中已删除文件的引用
            _prune_gallery()
        except Exception:
            pass


# ---------- Lifecycle ----------
# FastAPI 的 lifespan 机制：在应用启动时执行初始化，关闭时执行清理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 确保 processed/ 目录存在（首次启动时创建）
    config.PROCESSED_DIR.mkdir(exist_ok=True)

    # 2. 启动后台清理任务（定时删除过期的标注图）
    cleanup_task = asyncio.create_task(periodic_cleanup())

    # 3. 预加载 PaddleOCR 模型
    # 首次请求才加载模型会导致第一个请求特别慢，这里提前加载好
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: get_recognizer().initialize())

    # yield 之后的部分在应用关闭时执行
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
# 对外接口接收 bytes，内部完成 检测 → 识别 → 保存 完整流程
# 之所以放在线程池执行，是因为 PaddleOCR 推理是同步且耗时的，
# 放到独立线程中避免阻塞 FastAPI 的事件循环
# ---------------------------------------------------------------------------
def _process_image_bytes(image_bytes: bytes) -> dict:
    start = time.perf_counter()

    # bytes → numpy → OpenCV 解码
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        elapsed = (time.perf_counter() - start) * 1000
        return {"success": False, "error": "无法解码图片，请上传有效的图片文件",
                "processing_time_ms": round(elapsed, 1)}

    # 第一步：三策略检测 → 定位车牌区域
    detected, warped_plate, box_points, candidate, reason = detect_plate(img)

    if not detected or warped_plate is None:
        elapsed = (time.perf_counter() - start) * 1000
        return {"success": True, "detected": False,
                "plate_number": None, "confidence": 0.0,
                "annotated_image_url": None,
                "processing_time_ms": round(elapsed, 1),
                "error": reason}

    # 第二步：校正后的车牌区域 → OCR 字符识别
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

    # 第三步：识别成功 → 保存原图和标注图到 processed/ 目录
    # 文件名用时间戳 + UUID 片段确保唯一性，避免并发覆盖
    unique_name = (f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                   f"{uuid.uuid4().hex[:8]}")
    annotated_name = f"{unique_name}_annotated.jpg"
    original_name = f"{unique_name}_orig.jpg"
    annotated_path = config.PROCESSED_DIR / annotated_name
    original_path = config.PROCESSED_DIR / original_name

    cv2.imwrite(str(original_path), img)
    annotate_image(img, box_points, plate_text, annotated_path)

    # 第四步：写入图库，前端通过 GET /gallery 读取
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
    """上传图片 → 返回识别结果。

    接口说明：
      - 接受 multipart/form-data 格式的文件上传
      - 验证文件类型和大小，不符合则返回 400
      - 识别任务提交到线程池执行，不阻塞事件循环
      - 成功返回 { success, detected, plate_number, confidence,
                   annotated_image_url, processing_time_ms }
    """
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
    """返回图库记录列表，按时间倒序（最新的在前面）。

    图库存储在 processed/gallery.json，是一个 JSON 数组，
    每条记录结构：{ plate_number, confidence, timestamp, annotated_url, original_url }
    """
    records = await asyncio.get_event_loop().run_in_executor(
        None, _read_gallery)
    records.reverse()  # newest first
    return records
