import cv2
import numpy as np

from . import config


class FrameLocalizer:
    """
    Class responsible for finding the location of a single UAV frame
    within the base satellite map using feature matching.
    """
    def __init__(self, base_map_img):
        print("Initializing Frame Localizer...")
        self.base_map_img = base_map_img
        self.base_map_gray = cv2.cvtColor(base_map_img, cv2.COLOR_BGR2GRAY)

        # Initialize feature detector
        if config.FEATURE_DETECTOR == "ORB":
            print(f"Using ORB detector with {config.ORB_N_FEATURES} features.")
            self.detector = cv2.ORB_create(nfeatures=config.ORB_N_FEATURES)
            matcher_norm = cv2.NORM_HAMMING
        elif config.FEATURE_DETECTOR == "SIFT":
            print("Using SIFT detector.")
            self.detector = cv2.SIFT_create(nfeatures=config.SIFT_N_FEATURES)
            matcher_norm = cv2.NORM_L2
        else:
            raise ValueError(f"Unsupported feature detector: {config.FEATURE_DETECTOR}")

        # Detect features in the base map
        print("Detecting features in base map...")
        self.kp_map, self.des_map = self.detector.detectAndCompute(self.base_map_gray, None)
        if self.des_map is None or len(self.des_map) == 0:
            raise RuntimeError("No features detected in the base map. Check map image quality or detector parameters.")
        print(f"Detected {len(self.kp_map)} features in base map.")

        # Initialize matcher
        if config.MATCHER_TYPE == "BF":
             # crossCheck=False for KNN matching
            self.matcher = cv2.BFMatcher(matcher_norm, crossCheck=False)
        elif config.MATCHER_TYPE == "FLANN":
            if matcher_norm == cv2.NORM_L2:  # SIFT
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            else:  # ORB
                FLANN_INDEX_LSH = 6
                index_params= dict(algorithm=FLANN_INDEX_LSH,
                                   table_number=6,  # 12
                                   key_size=12,     # 20
                                   multi_probe_level=1)  # 2
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unsupported matcher type: {config.MATCHER_TYPE}")
        print("Frame Localizer initialized.")

    def find_location(self, uav_frame_img):
        """
        Finds the location (center point) and perspective transform (H)
        of the UAV frame on the base map.

        Args:
            uav_frame_img: The image captured by the UAV.

        Returns:
            A tuple ((center_x, center_y), H) representing the center coordinates
            and the homography matrix, or None if localization fails.
        """
        uav_frame_gray = cv2.cvtColor(uav_frame_img, cv2.COLOR_BGR2GRAY)
        h_frame, w_frame = uav_frame_gray.shape

        # 1. Detect features in the UAV frame
        kp_frame, des_frame = self.detector.detectAndCompute(uav_frame_gray, None)

        if des_frame is None or len(kp_frame) < config.MIN_MATCH_COUNT:
            return None

        # 2. Match features using KNN
        if config.FEATURE_DETECTOR == "SIFT":
            if self.des_map.dtype != np.float32:
                self.des_map = self.des_map.astype(np.float32)
            if des_frame.dtype != np.float32:
                des_frame = des_frame.astype(np.float32)

        matches = self.matcher.knnMatch(des_frame, self.des_map, k=config.KNN_K)

        # 3. Apply Lowe's Ratio Test
        good_matches = []
        # Handle potential empty matches list
        if matches:
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < config.LOWE_RATIO_THRESH * n.distance:
                        good_matches.append(m)

        # 4. Estimate Homography
        if len(good_matches) >= config.MIN_MATCH_COUNT:
            src_pts = np.float32([kp_frame[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.kp_map[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, config.RANSAC_REPROJ_THRESH)

            if H is None:
                return None  # Homography failed

            # 5. Calculate frame center on the map
            frame_center = np.float32([[[w_frame / 2, h_frame / 2]]])
            map_center_transformed = cv2.perspectiveTransform(frame_center, H)

            if map_center_transformed is None or len(map_center_transformed) == 0:
                return None  # Transform failed

            center_x = int(map_center_transformed[0][0][0])
            center_y = int(map_center_transformed[0][0][1])

            map_h, map_w = self.base_map_gray.shape
            if 0 <= center_x < map_w and 0 <= center_y < map_h:
                return ((center_x, center_y), H)
            else:
                return None  # Center outside map

        else:
            return None  # Not enough good matches