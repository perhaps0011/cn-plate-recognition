from __future__ import annotations

import cv2
import numpy as np
import pathlib

from . import config


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization for low-light images."""
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT,
                            tileGridSize=config.CLAHE_TILE_GRID_SIZE)
    return clahe.apply(gray)


def auto_canny(gray: np.ndarray) -> np.ndarray:
    """Canny edge detection with auto threshold from median."""
    median = np.median(gray)
    low = int(max(0, (1.0 - 0.33) * median))
    high = int(min(255, (1.0 + 0.33) * median))
    return cv2.Canny(gray, low, high, apertureSize=config.CANNY_APERTURE)


def white_balance(image: np.ndarray) -> np.ndarray:
    """Simple white balance via gray-world assumption."""
    result = image.astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg = (avg_b + avg_g + avg_r) / 3.0
    if avg_b > 0:
        result[:, :, 0] *= avg / avg_b
    if avg_g > 0:
        result[:, :, 1] *= avg / avg_g
    if avg_r > 0:
        result[:, :, 2] *= avg / avg_r
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def resize_if_needed(image: np.ndarray, max_dim: int = config.MAX_IMAGE_DIMENSION) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _order_corner_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect




# ---------------------------------------------------------------------------
# OpenCV preprocessing (fallback paths)
# ---------------------------------------------------------------------------


def find_plate_candidates_opencv(image: np.ndarray, binary: np.ndarray) -> tuple[list[dict], str]:
    """Sobel edge → contour → HSV verify."""
    candidates = []
    img_area = image.shape[0] * image.shape[1]

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return [], "未找到任何轮廓"

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue

        aspect_ratio = w / h
        rect_area = w * h
        cnt_area = cv2.contourArea(cnt)
        solidity = cnt_area / rect_area if rect_area > 0 else 0

        if aspect_ratio < config.ASPECT_RATIO_MIN or aspect_ratio > config.ASPECT_RATIO_MAX:
            continue
        area_ratio = rect_area / img_area
        if area_ratio < config.AREA_RATIO_MIN or area_ratio > config.AREA_RATIO_MAX:
            continue
        if solidity < config.SOLIDITY_MIN:
            continue

        roi = image[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, config.BLUE_LOWER, config.BLUE_UPPER)
        blue_ratio = cv2.countNonZero(mask) / (w * h)

        if blue_ratio >= config.MIN_BLUE_PIXEL_RATIO:
            candidates.append({
                "contour": cnt,
                "bbox": (x, y, w, h),
                "blue_ratio": blue_ratio,
                "solidity": solidity,
                "aspect_ratio": aspect_ratio,
                "source": "opencv_edge",
            })

    candidates.sort(key=lambda c: c["blue_ratio"], reverse=True)
    return candidates, f"找到 {len(candidates)} 个候选"


def find_plate_candidates_by_color(image: np.ndarray) -> tuple[list[dict], str]:
    """HSV color mask → contour (fallback)."""
    candidates = []
    img_area = image.shape[0] * image.shape[1]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, config.BLUE_LOWER, config.BLUE_UPPER)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return [], "颜色检测：未找到蓝色区域"

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue

        aspect_ratio = w / h
        rect_area = w * h
        area_ratio = rect_area / img_area

        if area_ratio < config.AREA_RATIO_MIN or area_ratio > config.AREA_RATIO_MAX:
            continue
        if aspect_ratio < config.ASPECT_RATIO_MIN or aspect_ratio > config.ASPECT_RATIO_MAX:
            continue

        cnt_area = cv2.contourArea(cnt)
        solidity = cnt_area / rect_area if rect_area > 0 else 0

        candidates.append({
            "contour": cnt,
            "bbox": (x, y, w, h),
            "blue_ratio": 1.0,
            "solidity": solidity,
            "aspect_ratio": aspect_ratio,
            "source": "color",
        })

    candidates.sort(key=lambda c: c["solidity"], reverse=True)
    return candidates, f"颜色检测：找到 {len(candidates)} 个候选"


# ---------------------------------------------------------------------------
# Canny edge detection strategy (alternative to Sobel)
# ---------------------------------------------------------------------------
def find_plate_candidates_canny(image: np.ndarray) -> tuple[list[dict], str]:
    """CLAHE → Canny edge → larger morph close → contour → HSV verify.

    Catches plates that Sobel misses (low contrast / blurry edges).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = apply_clahe(gray)
    edges = auto_canny(enhanced)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (config.MORPH_KERNEL_WIDTH_LARGE, config.MORPH_KERNEL_HEIGHT_LARGE)
    )
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], "Canny检测：未找到任何轮廓"

    img_area = image.shape[0] * image.shape[1]
    candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue

        aspect_ratio = w / h
        rect_area = w * h
        cnt_area = cv2.contourArea(cnt)
        solidity = cnt_area / rect_area if rect_area > 0 else 0
        area_ratio = rect_area / img_area

        if aspect_ratio < config.ASPECT_RATIO_MIN or aspect_ratio > config.ASPECT_RATIO_MAX:
            continue
        if area_ratio < config.AREA_RATIO_MIN or area_ratio > config.AREA_RATIO_MAX:
            continue
        if solidity < config.SOLIDITY_MIN:
            continue

        # HSV blue verification
        roi = image[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, config.BLUE_LOWER, config.BLUE_UPPER)
        blue_ratio = cv2.countNonZero(mask) / (w * h)

        if blue_ratio >= config.MIN_BLUE_PIXEL_RATIO:
            candidates.append({
                "contour": cnt,
                "bbox": (x, y, w, h),
                "blue_ratio": blue_ratio,
                "solidity": solidity,
                "aspect_ratio": aspect_ratio,
                "source": "canny",
            })

    candidates.sort(key=lambda c: c["blue_ratio"], reverse=True)
    return candidates, f"Canny检测：找到 {len(candidates)} 个候选"


# ---------------------------------------------------------------------------
# Extraction & annotation
# ---------------------------------------------------------------------------
def extract_plate_region(image: np.ndarray, candidate: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract and deskew the plate region via perspective transform."""
    # OpenCV contour — use minAreaRect
    cnt = candidate["contour"]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    ordered_src = _order_corner_points(box.astype("float32"))

    (tl, tr, br, bl) = ordered_src
    width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))

    width = max(int(width), 80)
    height = max(int(height), 25)

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_src, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped, box


def annotate_image(image: np.ndarray, box: np.ndarray, plate_text: str,
                   output_path: pathlib.Path) -> None:
    """Draw plate bounding box and text on the image, save to output_path."""
    annotated = image.copy()

    cv2.drawContours(annotated, [box], -1, (0, 255, 0), thickness=2)

    x, y, w, h = cv2.boundingRect(box)
    text_x, text_y = x, max(y - 10, 10)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    (tw, th), _ = cv2.getTextSize(plate_text, font, font_scale, thickness)
    cv2.rectangle(annotated,
                  (text_x, text_y - th - 6),
                  (text_x + tw + 6, text_y + 4),
                  (0, 0, 0), -1)

    cv2.putText(annotated, plate_text, (text_x + 3, text_y - 3),
                font, font_scale, (0, 255, 0), thickness)

    cv2.imwrite(str(output_path), annotated)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def detect_plate(image: np.ndarray) -> tuple[bool, np.ndarray | None, np.ndarray | None, dict | None, str]:
    """Three-strategy plate detection.

    Strategy 1: Sobel X → Otsu → Morph close → HSV verify (original, works for most cases)
    Strategy 2: CLAHE → Canny edge → Morph close → HSV verify (low-light / blurry)
    Strategy 3: HSV color mask → contour (last resort for difficult cases)

    Returns (detected, warped_plate, box_points, candidate, reason).
    """
    image = resize_if_needed(image)

    # Strategy 1: original Sobel pipeline (no CLAHE — it hurts clean images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_KERNEL, config.GAUSSIAN_SIGMA)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0,
                        ksize=config.SOBEL_KSIZE,
                        scale=config.SOBEL_SCALE,
                        delta=config.SOBEL_DELTA)
    sobel_abs = cv2.convertScaleAbs(sobel_x)
    _, thresh = cv2.threshold(sobel_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (config.MORPH_KERNEL_WIDTH, config.MORPH_KERNEL_HEIGHT)
    )
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,
                              iterations=config.MORPH_ITERATIONS)
    candidates, diag = find_plate_candidates_opencv(image, closed)

    # Strategy 2: Canny edge (complementary — catches low-contrast/blurry plates)
    if not candidates:
        candidates, diag = find_plate_candidates_canny(image)

    # Strategy 3: HSV color mask (last resort)
    if not candidates:
        candidates, diag = find_plate_candidates_by_color(image)

    if not candidates:
        return False, None, None, None, diag

    best = candidates[0]
    warped, box = extract_plate_region(image, best)

    # Upscale if too small for OCR
    if warped.shape[1] < 120:
        sf = max(120 / warped.shape[1], 2.0)
        new_w = int(warped.shape[1] * sf)
        new_h = int(warped.shape[0] * sf)
        warped = cv2.resize(warped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return True, warped, box, best, "ok"
