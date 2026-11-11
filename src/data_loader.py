import cv2
import os
import glob
from natsort import natsorted
from tqdm import tqdm

from . import config


def load_base_map():
    """Loads the base satellite map."""
    print(f"Loading base map from: {config.MAP_PATH}")
    if not os.path.isfile(config.MAP_PATH):
        raise FileNotFoundError(f"Base map image file not found at the specified path: {config.MAP_PATH}")

    base_map = cv2.imread(config.MAP_PATH, cv2.IMREAD_COLOR)
    if base_map is None:
        raise IOError(f"Base map image found at {config.MAP_PATH} but could not be read by OpenCV. Check file integrity and permissions.")
    print(f"Base map loaded successfully. Shape: {base_map.shape}")
    return base_map


def load_uav_frames():
    """Loads and sorts the UAV frames from the directory specified in config.FRAMES_DIR."""
    frame_pattern_png = os.path.join(config.FRAMES_DIR, "*.png")

    frame_files = glob.glob(frame_pattern_png)

    if not frame_files:
        raise FileNotFoundError(f"No frame images (.png) found in the directory specified by FRAMES_DIR: {config.FRAMES_DIR}")

    sorted_frame_files = natsorted(frame_files)
    print(f"Found {len(sorted_frame_files)} UAV frames in {config.FRAMES_DIR}. Loading...")

    frames = []
    for f_path in tqdm(sorted_frame_files, desc="Loading Frames"):
        frame = cv2.imread(f_path, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Warning: Could not load frame {f_path}")
            continue
        frames.append((os.path.basename(f_path), frame))

    if not frames:
        raise IOError(f"Found frame files in {config.FRAMES_DIR}, but none could be successfully loaded by OpenCV.")

    print(f"Loaded {len(frames)} frames successfully.")
    return frames
