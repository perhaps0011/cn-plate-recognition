import pathlib

# ---------- Paths ----------
# 项目路径全部基于 BASE_DIR 计算，确保在不同目录下启动都能找到文件
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
STATIC_DIR = BASE_DIR / "static"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024      # 10 MB，超过则拒绝上传
ANNOTATED_FILE_CLEANUP_AGE = 3600       # 标注图 1 小时后自动删除，释放磁盘空间
CLEANUP_INTERVAL = 600                   # 每 10 分钟执行一次清理任务
GALLERY_RETENTION = 86400 * 30           # 图库记录保留 30 天
GALLERY_JSON = PROCESSED_DIR / "gallery.json"

# ---------- Preprocessing ----------
# 高斯模糊核大小：5×5 足够消除噪点，又不至于模糊掉车牌边缘
GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 0       # 0 = OpenCV 根据核大小自动计算 sigma

# ---------- CLAHE Contrast Enhancement ----------
# CLAHE 用于低光照图增强对比度，只在 Canny 检测路径中使用
# clipLimit=2.0 限制对比度放大倍数，防止噪点被过度放大
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)   # 8×8 分块，块越小局部对比度越强

# ---------- Sobel Edge Detection ----------
# Sobel X 方向：水平梯度能突出车牌字符的竖边（字符在水平方向变化剧烈）
SOBEL_KSIZE = 3          # 3×3 Sobel 核，兼顾计算速度和边缘精度
SOBEL_SCALE = 1          # 梯度缩放系数，保持原始强度
SOBEL_DELTA = 0          # 不额外增加偏移

# ---------- Canny Edge Detection ----------
# Canny 用双阈值，此处的值是后备值；
# 实际代码中优先使用自动阈值（基于图像中位亮度计算）
CANNY_THRESHOLD_LOW = 80
CANNY_THRESHOLD_HIGH = 200
CANNY_APERTURE = 3

# ---------- Morphological Operations ----------
# 水平方向拉长核（17×3）：车牌字符是水平排列的，水平方向闭运算可以连接断裂的字符
MORPH_KERNEL_WIDTH = 17
MORPH_KERNEL_HEIGHT = 3
MORPH_ITERATIONS = 2

# Canny 路径使用更大的核（25×5）：因为 Canny 边缘更细碎，需要更强的连接能力
MORPH_KERNEL_WIDTH_LARGE = 25
MORPH_KERNEL_HEIGHT_LARGE = 5

# ---------- Multi-scale ----------
# 对同一张图尝试 1.0、0.75、0.5 三种缩放，增加小尺寸车牌的检测机会
SCALE_FACTORS = [1.0, 0.75, 0.5]

# ---------- Contour Filtering ----------
# 蓝牌宽高比约 3:1 ~ 5:1，放宽到 1.5~6.0 以容纳拍摄角度带来的透视变形
ASPECT_RATIO_MIN = 1.5
ASPECT_RATIO_MAX = 6.0
AREA_RATIO_MIN = 0.0003       # 候选区域至少占图 0.03%，排除噪点
AREA_RATIO_MAX = 0.35         # 最多 35%，避免选中整辆车
SOLIDITY_MIN = 0.15           # 最小充实度，排除 L 形等不规则轮廓

# ---------- HSV Blue Verification ----------
# 蓝牌底色在 HSV 空间的范围（H 色调:88~135 对应蓝色）
# S 和 V 范围放宽，以适应褪色或阴影中的车牌
BLUE_LOWER = (88, 35, 35)
BLUE_UPPER = (135, 255, 255)
MIN_BLUE_PIXEL_RATIO = 0.06  # 候选区域内蓝色像素至少占 6%

# ---------- Image Resizing ----------
# 超过此尺寸的图片等比缩放到最长边 1600px，减少处理时间
MAX_IMAGE_DIMENSION = 1600

# ---------- Recognition ----------
# PADDLEX_DET_MODEL disabled: PaddlePaddle 3.3.1 oneDNN bug
# (ConvertPirAttribute2RuntimeAttribute not support ArrayAttribute<DoubleAttribute>)
# See: onednn_instruction.cc:118
# Use improved OpenCV pipeline instead.
PADDLEX_REC_MODEL = "PP-OCRv4_mobile_rec"

# ---------- Allowed Upload Formats ----------
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ---------- Device ----------
DEVICE = "cpu"
