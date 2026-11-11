import numpy as np
from scipy.signal import savgol_filter

from . import config


class Trajectory:
    """Manages the sequence of estimated UAV localizations (point and H)."""

    def __init__(self):
        # Store tuples of ((x, y), H) or None
        self.raw_localizations = []
        self.smoothed_points = []

    def add_localization_result(self, localization_result):
        """Adds an estimated localization result ((point, H) or None)."""
        self.raw_localizations.append(localization_result)

    def get_raw_points(self):
        """Returns list of raw (x, y) points or None."""
        return [loc[0] if loc else None for loc in self.raw_localizations]

    def get_valid_raw_points(self):
        """Returns only the successfully localized raw points (x,y)."""
        return [loc[0] for loc in self.raw_localizations if loc is not None]

    def get_smoothed_points(self):
        """
        Returns smoothed trajectory points (list of (x, y) or None).
        Interpolation fills gaps in points.
        """
        if not self.smoothed_points:
            # Process using the raw points extracted from raw_localizations
            self._process_trajectory(self.get_raw_points(), interpolate=config.INTERPOLATE_GAPS)
        return self.smoothed_points # List of (x,y) or None

    def get_raw_localizations(self):
        """Returns the full list of raw localization results ((point, H) or None)."""
        return self.raw_localizations

    def _interpolate_gaps(self, points):
        """Linearly interpolates None values in a list of (x,y) points."""
        points_arr = np.array([(p[0] if p else np.nan, p[1] if p else np.nan) for p in points], dtype=float)
        n_points = len(points_arr)
        indices = np.arange(n_points)

        for dim in range(2):  # Interpolate x and y separately
            mask = np.isnan(points_arr[:, dim])
            valid_indices = indices[~mask]
            if np.any(mask) and len(valid_indices) >= 2:
                interpolated_values = np.interp(indices[mask], valid_indices, points_arr[valid_indices, dim])
                points_arr[mask, dim] = interpolated_values

        # Convert back to list of tuples, keeping None where interpolation wasn't possible (e.g., start/end Nones)
        interpolated = []
        for i in range(n_points):
            if np.isnan(points_arr[i, 0]) or np.isnan(points_arr[i, 1]):
                interpolated.append(None)
            else:
                interpolated.append((int(round(points_arr[i, 0])), int(round(points_arr[i, 1]))))
        return interpolated

    def _process_trajectory(self, raw_points_list, interpolate=True):
        """Internal method to interpolate and smooth the trajectory points."""
        points_to_process = list(raw_points_list)

        if interpolate:
            print("Interpolating missing points...")
            points_to_process = self._interpolate_gaps(points_to_process)

        valid_points_for_smoothing = [p for p in points_to_process if p is not None]

        if len(valid_points_for_smoothing) < config.SMOOTHING_WINDOW_SIZE or len(valid_points_for_smoothing) < config.SMOOTHING_POLY_ORDER + 1:
            print(f"Warning: Not enough valid points ({len(valid_points_for_smoothing)}) for smoothing "
                  f"(Window: {config.SMOOTHING_WINDOW_SIZE}, Order: {config.SMOOTHING_POLY_ORDER}). Using processed (potentially interpolated) points directly.")
            self.smoothed_points = points_to_process
            return

        print(f"Smoothing trajectory using Savitzky-Golay filter (Window: {config.SMOOTHING_WINDOW_SIZE}, Order: {config.SMOOTHING_POLY_ORDER})...")
        x_coords = np.array([p[0] for p in valid_points_for_smoothing])
        y_coords = np.array([p[1] for p in valid_points_for_smoothing])

        try:
            smooth_x = savgol_filter(x_coords, config.SMOOTHING_WINDOW_SIZE, config.SMOOTHING_POLY_ORDER)
            smooth_y = savgol_filter(y_coords, config.SMOOTHING_WINDOW_SIZE, config.SMOOTHING_POLY_ORDER)
        except ValueError as e:
            print(f"Error during smoothing: {e}. Using processed (potentially interpolated) points directly.")
            self.smoothed_points = points_to_process
            return

        # Map smoothed coordinates back into the full list structure
        smoothed_points_full = list(points_to_process)
        valid_idx_counter = 0
        for i in range(len(smoothed_points_full)):
            if smoothed_points_full[i] is not None:
                # Check if the index is within the bounds of smoothed results
                if valid_idx_counter < len(smooth_x):
                    smoothed_points_full[i] = (int(round(smooth_x[valid_idx_counter])), int(round(smooth_y[valid_idx_counter])))
                    valid_idx_counter += 1
                else:
                    print(f"Warning: Mismatch in smoothed points length at index {i}")

        self.smoothed_points = smoothed_points_full
        print("Smoothing complete.")