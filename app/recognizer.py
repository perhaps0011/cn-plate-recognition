from __future__ import annotations

import numpy as np

from . import config


class PlateRecognizer:
    """PaddleX-based license plate text recognizer. Lazy-loaded singleton."""

    def __init__(self):
        self.model = None

    def initialize(self) -> None:
        """Load PaddleX text recognition model (called once on first use)."""
        from paddlex import create_model
        self.model = create_model(config.PADDLEX_REC_MODEL)

    def recognize(self, plate_img: np.ndarray) -> tuple[str, float]:
        """Recognize text from a cropped plate image. Returns (text, confidence)."""
        if self.model is None:
            self.initialize()

        if plate_img is None or plate_img.size == 0:
            return "", 0.0

        h, w = plate_img.shape[:2]
        if h < 20 or w < 100:
            return "", 0.0

        results = list(self.model.predict(plate_img))

        if results and len(results) > 0:
            text = str(results[0].get("rec_text", "")).strip()
            confidence = float(results[0].get("rec_score", 0.0))
            text = text.replace(" ", "").replace("O", "0").replace("I", "1")
            return text, confidence

        return "", 0.0


# Singleton accessor
_recognizer: PlateRecognizer | None = None


def get_recognizer() -> PlateRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = PlateRecognizer()
    return _recognizer
