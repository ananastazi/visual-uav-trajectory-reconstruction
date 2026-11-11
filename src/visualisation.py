import cv2
import numpy as np
from tqdm import tqdm

from . import config


def draw_bounding_box(image, uav_frame_shape, homography_matrix, color, thickness):
    """Draws the projected bounding box of a UAV frame onto the image."""
    if homography_matrix is None:
        return

    h, w = uav_frame_shape[:2]
    frame_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    try:
        # Transform corners to map coordinates
        map_corners = cv2.perspectiveTransform(frame_corners, homography_matrix)
        if map_corners is None:
            print("Warning: cv2.perspectiveTransform returned None for bounding box.")
            return

        pts = np.int32(map_corners.reshape(-1, 2))

        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    except cv2.error as e:
        print(f"Warning: OpenCV error drawing bounding box: {e}")
    except Exception as e:
        print(f"Warning: Unexpected error drawing bounding box: {e}")


def draw_trajectory(map_image, points, raw_localizations, uav_frames_data, **kwargs):
    """Draws the trajectory onto the map. Optionally adds final bbox."""
    point_color = kwargs.get('point_color', config.POINT_COLOR)
    line_color = kwargs.get('line_color', config.LINE_COLOR)
    start_color = kwargs.get('start_color', config.START_COLOR)
    finish_color = kwargs.get('finish_color', config.FINISH_COLOR)
    point_radius = kwargs.get('point_radius', config.POINT_RADIUS)
    line_thickness = kwargs.get('line_thickness', config.LINE_THICKNESS)
    marker_radius = kwargs.get('marker_radius', config.MARKER_RADIUS)
    draw_final_bbox = kwargs.get('draw_final_bbox', config.DRAW_FINAL_BBOX_ON_IMAGE)
    bbox_color = kwargs.get('bbox_color', config.BBOX_COLOR)
    bbox_thickness = kwargs.get('bbox_thickness', config.BBOX_THICKNESS)

    if not points:
        print("Warning: No points to draw.")
        return map_image.copy()

    output_image = map_image.copy()

    # --- Draw lines and points ---
    valid_points = [p for p in points if p is not None]
    if len(valid_points) > 1:
        for i in range(len(valid_points) - 1):
            p1 = valid_points[i]
            p2 = valid_points[i+1]
            if isinstance(p1, tuple) and isinstance(p2, tuple):
                try:
                    cv2.line(output_image, p1, p2, line_color, line_thickness, cv2.LINE_AA)
                except Exception as e:
                    print(f"Warning: Could not draw line between {p1} and {p2}. Error: {e}")

    for p in valid_points:
        if isinstance(p, tuple):
            try:
                cv2.circle(output_image, p, point_radius, point_color, -1, cv2.LINE_AA)
            except Exception as e:
                print(f"Warning: Could not draw point {p}. Error: {e}")

    # --- Mark Start and Finish ---
    first_valid_point = next((p for p in points if p is not None), None)
    last_valid_point = next((p for p in reversed(points) if p is not None), None)

    if first_valid_point:
        try:
            cv2.circle(output_image, first_valid_point, marker_radius, start_color, -1, cv2.LINE_AA)
            cv2.circle(output_image, first_valid_point, marker_radius, (0,0,0), line_thickness//2, cv2.LINE_AA)
            cv2.putText(output_image, "START", (first_valid_point[0] + marker_radius + 5, first_valid_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(output_image, "START", (first_valid_point[0] + marker_radius + 5, first_valid_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, start_color, 1, cv2.LINE_AA)
        except Exception as e:
            print(f"Warning: Could not draw start marker at {first_valid_point}. Error: {e}")

    if last_valid_point:
        try:
            cv2.rectangle(output_image,
                          (last_valid_point[0] - marker_radius, last_valid_point[1] - marker_radius),
                          (last_valid_point[0] + marker_radius, last_valid_point[1] + marker_radius),
                          finish_color, -1, cv2.LINE_AA)
            cv2.rectangle(output_image,
                          (last_valid_point[0] - marker_radius, last_valid_point[1] - marker_radius),
                          (last_valid_point[0] + marker_radius, last_valid_point[1] + marker_radius),
                          (0,0,0), line_thickness//2, cv2.LINE_AA)
            cv2.putText(output_image, "FINISH", (last_valid_point[0] + marker_radius + 5, last_valid_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(output_image, "FINISH", (last_valid_point[0] + marker_radius + 5, last_valid_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, finish_color, 1, cv2.LINE_AA)
        except Exception as e:
            print(f"Warning: Could not draw finish marker at {last_valid_point}. Error: {e}")

        # --- Draw Bounding Box for the last frame that had a successful localization ---
        if draw_final_bbox:
             last_valid_loc_index = -1
             for i in range(len(raw_localizations) - 1, -1, -1):
                 if raw_localizations[i] is not None:
                     last_valid_loc_index = i
                     break

             if last_valid_loc_index != -1 and last_valid_loc_index < len(uav_frames_data):
                 last_loc = raw_localizations[last_valid_loc_index]
                 last_frame_img = uav_frames_data[last_valid_loc_index][1]
                 print(f"Drawing final bounding box for frame index {last_valid_loc_index}")
                 draw_bounding_box(output_image, last_frame_img.shape, last_loc[1], bbox_color, bbox_thickness)
             else:
                 print("Could not find a valid final localization to draw bounding box.")

    return output_image


def save_image(image, path):
    """Saves the image to the specified path."""
    print(f"Saving output image to: {path}")
    success = cv2.imwrite(path, image)
    if not success:
        print(f"Error: Failed to save image to {path}")
    else:
        print("Image saved successfully.")


def create_video(base_map_img,
                 smoothed_points,
                 raw_localizations,
                 uav_frames_data,
                 output_path,
                 fps):
    """Creates a zoomed-in video visualizing the trajectory progression with frame bounding boxes."""

    if not smoothed_points:
        print("Warning: No smoothed points provided for video creation.")
        return

    map_h, map_w = base_map_img.shape[:2]
    roi_w, roi_h = config.VIDEO_ROI_WIDTH, config.VIDEO_ROI_HEIGHT

    roi_w = min(roi_w, map_w)
    roi_h = min(roi_h, map_h)

    print(f"Creating zoomed video ({roi_w}x{roi_h}) at: {output_path}")

    # Initialize Video Writer with ROI dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (roi_w, roi_h))

    if not video.isOpened():
        print(f"Error: Could not open video writer for path {output_path}")
        return

    num_frames = len(smoothed_points)

    for i in tqdm(range(num_frames), desc="Creating Zoomed Video"):
        current_smoothed_point = smoothed_points[i]

        if current_smoothed_point is None:
             # Find the last non-None point up to this frame
             last_good_point = next((p for p in reversed(smoothed_points[:i+1]) if p is not None), None)
             if last_good_point:
                 current_smoothed_point = last_good_point
             else:
                 current_smoothed_point = (map_w // 2, map_h // 2)
                 print(f"Warning: No valid point for video frame {i}, centering ROI on map center.")


        # --- Calculate ROI boundaries ---
        center_x, center_y = current_smoothed_point

        x_min = max(0, center_x - roi_w // 2)
        y_min = max(0, center_y - roi_h // 2)

        x_min = min(x_min, map_w - roi_w)
        y_min = min(y_min, map_h - roi_h)

        x_min = max(0, x_min)
        y_min = max(0, y_min)

        x_max = x_min + roi_w
        y_max = y_min + roi_h

        temp_canvas = base_map_img.copy()

        if config.VIDEO_DRAW_PATH:
            points_so_far = [p for p in smoothed_points[:i+1] if p is not None]
            if len(points_so_far) > 1:
                for j in range(len(points_so_far) - 1):
                     cv2.line(temp_canvas, points_so_far[j], points_so_far[j+1], config.LINE_COLOR, config.LINE_THICKNESS, cv2.LINE_AA)

        # Draw the current position marker prominently
        try:
             cv2.circle(temp_canvas, current_smoothed_point, config.VIDEO_CURRENT_POINT_RADIUS, config.VIDEO_CURRENT_POINT_COLOR, -1, cv2.LINE_AA)
             cv2.circle(temp_canvas, current_smoothed_point, config.VIDEO_CURRENT_POINT_RADIUS, (0,0,0), 1, cv2.LINE_AA) # Outline
        except Exception as e:
              print(f"Warning: Failed to draw current point marker at {current_smoothed_point}. Error: {e}")


        if config.VIDEO_DRAW_BBOX and i < len(raw_localizations):
            raw_loc = raw_localizations[i]
            if raw_loc is not None and i < len(uav_frames_data):
                 # raw_loc is ((x,y), H)
                 frame_img = uav_frames_data[i][1]
                 draw_bounding_box(temp_canvas, frame_img.shape, raw_loc[1], config.BBOX_COLOR, config.BBOX_THICKNESS)

        # --- Extract the ROI ---
        roi_frame = temp_canvas[int(y_min):int(y_max), int(x_min):int(x_max)]

        # --- Write ROI frame to video ---
        if roi_frame.shape[0] == roi_h and roi_frame.shape[1] == roi_w:
            video.write(roi_frame)
        else:
            print(f"Warning: ROI frame size mismatch for frame {i}. Expected ({roi_h},{roi_w}), Got {roi_frame.shape[:2]}. Resizing.")
            if roi_frame.size > 0:
                 resized_roi = cv2.resize(roi_frame, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
                 video.write(resized_roi)
            else:
                 black_frame = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
                 video.write(black_frame)


    video.release()
    print("Zoomed video created successfully.")