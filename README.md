# UAV Trajectory Reconstruction from Images

This project reconstructs and visualizes the flight path of a UAV based on a given a satellite panoramic image of an area where an unknown good UAV should have flown and a series of photos taken from the UAV during its flight.

## Features

*   **UAV Frame Localization:** Locates each UAV frame within the larger satellite map using feature-based image matching (SIFT recommended, ORB configurable).
*   **Trajectory Reconstruction:** Assembles the sequence of localized frame centers into a flight path.
*   **Robust Processing:** Includes optional linear interpolation to fill gaps where localization might fail and Savitzky-Golay filtering to smooth the resulting trajectory.
*   **Static Visualization:** Generates a single output image (`trajectory.png`) showing the full base map with the complete, smoothed trajectory overlaid, including start and finish markers. Optionally displays the projected bounding box of the final localized frame.
*   **Dynamic Visualization:** Creates a video (`trajectory_zoomed.mp4`) showing:
    *   A **zoomed-in region** of the map dynamically following the UAV's estimated position.
    *   The reconstructed path drawn incrementally.
    *   A marker indicating the UAV's current position.
    *   Optionally, the **projected bounding box** of the current UAV frame, preserving its estimated size and rotation on the map.

## Approach

1.  **Feature Detection:** Keypoints and descriptors are extracted from the base satellite map (once) and each UAV frame using algorithms like SIFT (default, robust to scale/rotation) or ORB.
2.  **Feature Matching:** Descriptors from the UAV frame are matched against the base map descriptors using a k-Nearest Neighbors (KNN) approach (with BFMatcher or FLANN (recommended)).
3.  **Match Filtering:** Lowe's ratio test is applied to retain only high-quality, unambiguous matches.
4.  **Homography Estimation:** If sufficient good matches exist (`MIN_MATCH_COUNT`), `cv2.findHomography` with RANSAC is used to estimate the perspective transformation matrix (`H`) that maps the UAV frame onto the base map.
5.  **Localization:**
    *   The center point of the UAV frame is transformed using `H` to find its corresponding (x, y) coordinates on the base map.
    *   The corner points of the UAV frame are transformed using `H` to determine the vertices for the bounding box visualization.
6.  **Trajectory Assembly:** The sequence of localization results (point and `H` matrix, or `None` on failure) is stored.
7.  **Trajectory Processing:**
    *   **(Interpolation):** Optionally, gaps in the sequence of center points (where localization failed) are filled using linear interpolation.
    *   **(Smoothing):** The sequence of (potentially interpolated) center points is smoothed using a Savitzky-Golay filter to reduce jitter.
8.  **Visualization:**
    *   **Static Image:** The smoothed points are drawn as a path on the original base map. Start/end points are marked. The bounding box corresponding to the *last successfully localized frame* can be drawn using its stored `H` matrix.
    *   **Video:** For each frame index:
        *   The ROI (Region of Interest) center is determined by the corresponding smoothed point.
        *   An ROI is extracted from the base map.
        *   The path history, current point marker, and (optionally) the bounding box (using the *raw* `H` matrix for that specific frame index) are drawn onto the ROI.
        *   The resulting ROI frame is written to the video file.

## Project Structure
    task2/
    ├── main.py # Main execution script
    ├── src/ # Source code modules
    │ ├── init.py
    │ ├── config.py # Configuration parameters (paths, algorithms, visuals)
    │ ├── data_loader.py # Loads base map and UAV frames
    │ ├── localization.py # FrameLocalizer class (feature matching, homography)
    │ ├── trajectory.py # Trajectory class (stores results, interpolates, smooths)
    │ ├── visualisation.py # Generates output image and video
    │ └── utils.py # Utility functions (if needed)
    ├── data/ # Input data folder
    │ ├── global_map.png # The large base satellite map
    │ └── crops/ # Folder containing the UAV image sequence
    │ ├── crop_0.png
    │ ├── crop_1.png
    │ └── ...
    ├── output/ # Output folder
    │ ├── trajectory.png # Result static image
    │ └── trajectory.mp4 # Result dynamic video
    ├── requirements.txt # Python dependencies
    └── README.md # This file
## Prerequisites

*   Python 3.7+
*   Virtual Environment (Recommended)
*   Dependencies listed in `requirements.txt`. Key dependencies include:
    *   `opencv-python` or `opencv-contrib-python` (Recommended for SIFT)
    *   `numpy`
    *   `scipy` (for smoothing)
    *   `natsort` (for correct frame sorting)
    *   `tqdm` (for progress bars)
