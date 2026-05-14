# 基于 OpenCV 与 PaddleOCR 的车牌识别系统

基于 OpenCV + PaddleOCR 的中国蓝牌车牌检测与识别系统，支持 Web 上传识别和本地命令行识别。

## 功能

- **车牌检测**：三策略 OpenCV 图像处理（Sobel 边缘检测、Canny 边缘检测、HSV 颜色掩码）
- **字符识别**：基于 PaddleOCR PP-OCRv4 的端到端文字识别
- **Web 界面**：拖拽上传图片，实时显示标注结果，Apple 风格 UI
- **识别图库**：自动保存识别记录，支持历史查询和原图对比
- **命令行模式**：无需启动 Web 服务，一键识别

## 项目结构

```
vision/
├── app/
│   ├── __init__.py
│   ├── config.py         # 配置参数
│   ├── detector.py       # 车牌检测（OpenCV 三策略）
│   ├── main.py           # FastAPI Web 服务
│   └── recognizer.py     # PaddleOCR 字符识别
├── static/
│   └── index.html        # 前端页面
├── processed/            # 识别结果存储（自动生成）
├── requirements.txt
├── run.py                # 启动入口
└── README.md
```

## 快速开始

### 环境要求

- Python 3.8+
- pip

### 安装

```bash
# 克隆项目
git clone https://github.com/perhaps0011/vision.git
cd vision

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate   # Linux
# venv\Scripts\activate    # Windows

# 安装依赖
pip install -r requirements.txt
```

### 使用

**命令行识别：**
```bash
python run.py predict model/test_2.jpg
```

**启动 Web 服务：**
```bash
python run.py
# 访问 http://localhost:8000
```

项目 `model/` 目录下提供了示例图片可用于测试。

## 检测策略

| 策略 | 方法 | 适用场景 |
|------|------|----------|
| 1 | CLAHE → Sobel X → Otsu → HSV 验证 | 正常光照、对比度好的图片 |
| 2 | CLAHE → Canny 边缘检测 → HSV 验证 | 低光照、模糊图片 |
| 3 | HSV 颜色掩码 → 轮廓提取 | 蓝色车体、边缘不清晰的情况 |

## 技术栈

- **后端**：FastAPI + Uvicorn
- **前端**：纯 HTML/CSS/JS（Apple 风格 UI）
- **图像处理**：OpenCV（NumPy）
- **文字识别**：PaddleOCR（PP-OCRv4）
- **部署**：Linux + 宝塔面板反向代理

## 许可

MIT License
