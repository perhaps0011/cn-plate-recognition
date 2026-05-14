#!/usr/bin/env python3
"""车牌识别系统入口——同时支持 Web 服务和命令行模式。

用法:
  python run.py                       # 启动 Web 服务器，访问 http://localhost:8000
  python run.py predict <图片路径>     # 命令行模式，直接识别指定图片

命令行模式使用说明:
  predict 子命令接收一张图片，输出检测结果到终端，
  同时在当前目录下生成 annotated_<原文件名>.jpg 标注图。
"""
import argparse
import sys
from pathlib import Path

import cv2

# 将项目根目录加入 Python 搜索路径，确保 import app 能够找到
sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_server():
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
        log_level="info",
    )


def run_predict(image_path: str):
    """命令行模式：读取图片 → 检测 → 识别 → 终端输出结果并保存标注图。"""
    from app.detector import annotate_image, detect_plate
    from app.recognizer import get_recognizer

    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        sys.exit(1)

    print(f"图片: {image_path}  ({img.shape[1]}x{img.shape[0]})")
    print()

    detected, warped, box, _, reason = detect_plate(img)

    if not detected or warped is None:
        print(f"车牌检测: 失败")
        print(f"原因: {reason}")
        sys.exit(0)

    recognizer = get_recognizer()
    plate_text, confidence = recognizer.recognize(warped)

    print(f"车牌检测: 成功")
    print(f"车牌号码: {plate_text or '无法识别'}")
    print(f"置信度:   {confidence:.2%}" if confidence > 0 else "置信度:   0%")
    print(f"检测耗时:  {reason}")

    if box is not None and plate_text:
        out_path = f"annotated_{Path(image_path).stem}.jpg"
        annotate_image(img, box, plate_text, Path(out_path))
        print(f"标注图片: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Chinese License Plate Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode")

    # predict subcommand
    p = sub.add_parser("predict", help="Recognize plate from an image file")
    p.add_argument("image", help="Path to image file")
    p.add_argument("-o", "--output", action="store_true",
                   help="Save annotated image (always enabled for convenience)")

    # If no subcommand, start the server
    args = parser.parse_args()

    if args.mode == "predict":
        run_predict(args.image)
    else:
        run_server()


if __name__ == "__main__":
    main()
