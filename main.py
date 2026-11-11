import time
from tqdm import tqdm

from src import config
from src import data_loader
from src.localization import FrameLocalizer
from src.trajectory import Trajectory
from src import visualisation

def run():
    start_time = time.time()

    # 1. Load Data
    try:
        base_map = data_loader.load_base_map()
        uav_frames_data = data_loader.load_uav_frames()
        if not uav_frames_data:
             print("Error: No UAV frames were loaded successfully.")
             return
    except (FileNotFoundError, IOError, Exception) as e:
        print(f"Error loading data: {e}")
        return

    # 2. Initialize Localization and Trajectory
    try:
        localizer = FrameLocalizer(base_map)
    except (RuntimeError, ValueError, Exception) as e:
        print(f"Error initializing FrameLocalizer: {e}")
        return

    trajectory = Trajectory()

    # 3. Process each frame
    print("\nProcessing UAV frames to find locations...")
    localization_times = []
    for frame_name, frame_img in tqdm(uav_frames_data, desc="Localizing Frames"):
        frame_start_time = time.time()
        try:
             localization_result = localizer.find_location(frame_img)
             trajectory.add_localization_result(localization_result)

             frame_end_time = time.time()
             localization_times.append(frame_end_time - frame_start_time)

        except Exception as e:
             import traceback
             print(f"\nError processing frame {frame_name}: {e}")
             traceback.print_exc()
             trajectory.add_localization_result(None)


    # 4. Process and Smooth Trajectory Points
    final_smoothed_points = trajectory.get_smoothed_points()

    raw_localizations = trajectory.get_raw_localizations()
    raw_valid_points = trajectory.get_valid_raw_points()

    print(f"\nLocalization summary:")
    print(f"  Successfully localized {len(raw_valid_points)} out of {len(uav_frames_data)} frames.")
    if localization_times:
        avg_time = sum(localization_times) / len(localization_times)
        print(f"  Average time per frame: {avg_time:.3f} seconds")
    num_final_valid = len([p for p in final_smoothed_points if p is not None])
    print(f"  Number of valid points in final trajectory (after processing): {num_final_valid} / {len(final_smoothed_points)}")

    # 5. Visualize Results - Static Image
    print("\nVisualizing trajectory on static image...")
    output_image = visualisation.draw_trajectory(
        base_map,
        final_smoothed_points,
        raw_localizations,
        uav_frames_data,
        point_color=config.POINT_COLOR,
        line_color=config.LINE_COLOR,
        start_color=config.START_COLOR,
        finish_color=config.FINISH_COLOR,
        point_radius=config.POINT_RADIUS,
        line_thickness=config.LINE_THICKNESS,
        marker_radius=config.MARKER_RADIUS,
        draw_final_bbox=config.DRAW_FINAL_BBOX_ON_IMAGE,
        bbox_color=config.BBOX_COLOR,
        bbox_thickness=config.BBOX_THICKNESS
    )

    # 6. Save Output Image
    visualisation.save_image(output_image, config.OUTPUT_IMAGE_PATH)

    # 7. Create Output Video
    create_video_flag = True
    if create_video_flag:
         if final_smoothed_points and any(p is not None for p in final_smoothed_points):
             print("\nCreating zoomed video...")
             visualisation.create_video(
                 base_map,
                 final_smoothed_points,
                 raw_localizations,
                 uav_frames_data,
                 config.OUTPUT_VIDEO_PATH,
                 config.VIDEO_FPS
             )
         else:
              print("Skipping video creation as there are no valid points in the final trajectory.")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print(f"Output image saved to: {config.OUTPUT_IMAGE_PATH}")
    if create_video_flag and final_smoothed_points and any(p is not None for p in final_smoothed_points):
        print(f"Output video saved to: {config.OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    run()