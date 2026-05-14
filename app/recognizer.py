from __future__ import annotations

import numpy as np

from . import config


class PlateRecognizer:
    """PaddleX 文字识别封装——延迟初始化 + 单例模式。

    模型（PP-OCRv4_mobile_rec，10.8MB）在首次调用时才下载加载，
    避免启动时等待。后续请求复用已加载的模型实例（单例模式）。
    """

    def __init__(self):
        self.model = None

    def initialize(self) -> None:
        """首次加载 PaddleX 模型，会自动从 ModelScope 下载 PP-OCRv4 模型文件。"""
        from paddlex import create_model
        self.model = create_model(config.PADDLEX_REC_MODEL)

    def recognize(self, plate_img: np.ndarray) -> tuple[str, float]:
        """识别校正后的车牌图片，返回（文字, 置信度）。"""
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
            # 后处理：去除空格，替换常见误识（O→0, I→1）
            text = text.replace(" ", "").replace("O", "0").replace("I", "1")
            return text, confidence

        return "", 0.0


# 模块级单例，全局只保留一个模型实例以节省内存
_recognizer: PlateRecognizer | None = None


def get_recognizer() -> PlateRecognizer:
    """获取单例识别器。第一次调用时创建实例，后续复用。"""
    global _recognizer
    if _recognizer is None:
        _recognizer = PlateRecognizer()
    return _recognizer
