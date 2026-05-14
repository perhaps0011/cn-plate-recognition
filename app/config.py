import pathlib

# ---------- Paths ----------
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
STATIC_DIR = BASE_DIR / "static"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024      # 10 MB
ANNOTATED_FILE_CLEANUP_AGE = 3600       # delete annotated files older than 1 hour
CLEANUP_INTERVAL = 600                   # run cleanup every 10 minutes
GALLERY_RETENTION = 86400 * 30           # gallery records kept for 30 days
GALLERY_JSON = PROCESSED_DIR / "gallery.json"

# ---------- Preprocessing ----------
GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 0

# ---------- CLAHE Contrast Enhancement ----------
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# ---------- Sobel Edge Detection ----------
SOBEL_KSIZE = 3
SOBEL_SCALE = 1
SOBEL_DELTA = 0

# ---------- Canny Edge Detection ----------
CANNY_THRESHOLD_LOW = 80
CANNY_THRESHOLD_HIGH = 200
CANNY_APERTURE = 3

# ---------- Morphological Operations ----------
MORPH_KERNEL_WIDTH = 17
MORPH_KERNEL_HEIGHT = 3
MORPH_ITERATIONS = 2

# Large kernel (for Canny / second pass)
MORPH_KERNEL_WIDTH_LARGE = 25
MORPH_KERNEL_HEIGHT_LARGE = 5

# ---------- Multi-scale ----------
SCALE_FACTORS = [1.0, 0.75, 0.5]       # try multiple scales

# ---------- Contour Filtering ----------
ASPECT_RATIO_MIN = 1.5
ASPECT_RATIO_MAX = 6.0
AREA_RATIO_MIN = 0.0003       # min 0.03% of image area (relaxed a bit)
AREA_RATIO_MAX = 0.35         # max 35% of image area
SOLIDITY_MIN = 0.15

# ---------- HSV Blue Verification ----------
BLUE_LOWER = (88, 35, 35)
BLUE_UPPER = (135, 255, 255)
MIN_BLUE_PIXEL_RATIO = 0.06

# ---------- Image Resizing ----------
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
