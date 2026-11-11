import os

# --- Input Data Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MAP_PATH = os.path.join(DATA_DIR, "global_map.png")
FRAMES_DIR = os.path.join(DATA_DIR, "crops")

# --- Output ---
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, "trajectory.png")
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "trajectory.mp4")

# --- Localization Parameters ---
FEATURE_DETECTOR = "SIFT"  # Options: "ORB", "SIFT"
ORB_N_FEATURES = 2000
SIFT_N_FEATURES = 0

# Matcher settings
MATCHER_TYPE = "FLANN"  # Options: "BF", "FLANN"
KNN_K = 2
LOWE_RATIO_THRESH = 0.75

# Homography/Localization settings
MIN_MATCH_COUNT = 10  # Minimum number of good matches required to estimate position
RANSAC_REPROJ_THRESH = 5.0

# --- Trajectory Processing ---
SMOOTHING_WINDOW_SIZE = 7
SMOOTHING_POLY_ORDER = 2
INTERPOLATE_GAPS = True

# --- Visualization ---
POINT_COLOR = (255, 0, 0)
LINE_COLOR = (0, 0, 255)
START_COLOR = (0, 255, 0)
FINISH_COLOR = (0, 0, 255)
POINT_RADIUS = 5
LINE_THICKNESS = 2
MARKER_RADIUS = 10

# --- Bounding Box Visualization ---
BBOX_COLOR = (255, 0, 255)
BBOX_THICKNESS = 2
DRAW_FINAL_BBOX_ON_IMAGE = False

# --- Video Specific ---
VIDEO_FPS = 5
VIDEO_ROI_WIDTH = 800
VIDEO_ROI_HEIGHT = 600
VIDEO_POINT_COLOR = (255, 100, 0)
VIDEO_CURRENT_POINT_COLOR = (0, 255, 255)
VIDEO_CURRENT_POINT_RADIUS = 8
VIDEO_DRAW_PATH = True
VIDEO_DRAW_BBOX = True
