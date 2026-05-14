import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt


def show_temp_graph(signal, time, y_axis_label="Signal"):
    plt.plot(time, signal)  # t = time array, x = 1D signal
    plt.xlabel("Time")
    plt.ylabel(y_axis_label)
    plt.show()


# Requires 2 timeseries with equal number of entries?
def check_correlation_between_timeseries(timeseries1, timeseries2):
    r, p = stats.pearsonr(timeseries1, timeseries2)
    r_squared = r ** 2
    print(f"r^2 value: {r_squared}")
    print(f"p-value: {p}")
    print("")


# Prints the p-value and r^2 value of the correlation between scene lighting and eye contact by interpolating scene
# lighting onto eye contact timestamps. parent_folder is the folder containing the eye contact csv.
def check_correlation_lighting_eye_contact(parent_folder, illum_levels_folder):
    eye_contact_path = parent_folder / "mutual_eye_contact.csv"
    illum_levels_path = illum_levels_folder / "scene_illumination_levels.csv"

    df1 = pd.read_csv(eye_contact_path)
    df2 = pd.read_csv(illum_levels_path)

    eye_contact = df1["eye contact"]
    time_eye_contact = df1["timestamp [ns]"]
    illum_levels = df2["mean_intensity"]
    time_illum = df2["timestamp [ns]"]

    illum_interp = np.interp(time_eye_contact, time_illum, illum_levels)
    print("Correlation between eye contact and illumination levels:")
    check_correlation_between_timeseries(illum_interp, eye_contact)


# Makes a timeseries csv of illumination levels for every frame in the video. Output is saved in timeseries_folder.
# Note that we calculate illumination levels by taking a weighted sum of pixel intensities using a gaussian function
# centered at gaze position. This simulates the way that perceived scene illumination depends on gaze direction.
def extract_illumination_levels(video_path, timeseries_folder):

    def gaussian_weighted_intensity(frame, gaze_x, gaze_y, sigma=50):
        """
        frame: HxWx3 (uint8 or float)
        gaze_x, gaze_y: pixel coordinates (int)
        sigma: std dev of Gaussian (in pixels)

        returns: weighted mean intensity (float)
        """

        # Convert to grayscale intensity (more perceptually relevant than raw RGB mean)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        H, W = gray.shape

        # Create coordinate grid
        y = np.arange(H)
        x = np.arange(W)
        X, Y = np.meshgrid(x, y)

        # Gaussian weights centered at gaze
        dist2 = (X - gaze_x) ** 2 + (Y - gaze_y) ** 2
        weights = np.exp(-dist2 / (2 * sigma ** 2))

        # Normalize weights (important)
        weights /= weights.sum()

        # Weighted mean
        return np.sum(gray * weights)

    # Save path for the csv file in the same folder as video_path
    output_csv = timeseries_folder / "scene_illumination_levels.csv"

    # Get gaze dataframe (for approximating gaze position at each frame)
    gaze_path = timeseries_folder / "gaze.csv"
    df_gaze = pd.read_csv(gaze_path)
    time_window = df_gaze["timestamp [ns]"]
    start_time_ns = time_window.iloc[0]
    x_vals = df_gaze["gaze x [px]"]
    y_vals = df_gaze["gaze y [px]"]
    ts = time_window.to_numpy()
    x_vals = x_vals.to_numpy()
    y_vals = y_vals.to_numpy()

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # Get FPS for timestamps
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("FPS could not be determined.")

    frame_idx = 0
    illumination_values = []
    timestamps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Timestamp in seconds
        time_sec = frame_idx / fps
        time_ns = (time_sec / 1e9) + start_time_ns

        # Find the timestamp from the glasses data closest to time_ns (in terms of offset from start time) and retrieve
        # the gaze x [px] and gaze y [px] values. Use binary search (since timestamps are in chronological order) for
        # O(logN) retrieval time.
        i = np.searchsorted(ts, time_ns)

        if i == 0:
            x_pixel_val = x_vals[0]
            y_pixel_val = y_vals[0]
        elif i == len(ts):
            x_pixel_val = x_vals[-1]
            y_pixel_val = y_vals[-1]
        else:
            before = i - 1
            after = i
            if abs(ts[after] - time_ns) < abs(ts[before] - time_ns):
                x_pixel_val = x_vals[after]
                y_pixel_val = y_vals[after]
            else:
                x_pixel_val = x_vals[before]
                y_pixel_val = y_vals[before]

        # Compute illumination intensity using gaussian weighting centered at gaze position
        illumination_val = gaussian_weighted_intensity(frame, int(x_pixel_val.round()), int(y_pixel_val.round()))

        # Print a message for every 5 minutes of processed video
        if frame_idx % int(fps * 300) == 0:
            print(f"Illumination levels calculated for {int(time_sec)+1}s of video")

        illumination_values.append(illumination_val)
        timestamps.append(time_sec)

        frame_idx += 1

    cap.release()

    # Create DataFrame
    df = pd.DataFrame({
        "frame": range(len(illumination_values)),
        "timestamp [ns]": timestamps,
        "mean_intensity": illumination_values
    })

    # Save to CSV
    df.to_csv(output_csv, index=False)

    # We added the change from seconds to nanoseconds later on, hence the rewriting of the saved csv instead of
    # saving it correctly the first time around. But this functionally does the same thing.
    df_illum = pd.read_csv(output_csv)

    # Convert seconds to nanoseconds
    df_illum["timestamp [ns]"] = (df_illum["timestamp [ns]"] * 1e9).astype("int64")
    df_illum["timestamp [ns]"] = (df_illum["timestamp [ns]"] + start_time_ns)

    df_illum.to_csv(output_csv, index=False)

    print(f"Processed {frame_idx} frames.")
    print(f"Saved output to: {output_csv}")


# Regress illumination out of pupil diameter using time-aligned interpolation.
def regress_lighting_from_pupil_diameter(eye_states_path, illum_levels_path,
                                         illum_time_col="timestamp [ns]", illum_value_col="mean_intensity"):
    ts_folder = eye_states_path.parent

    # Load data
    pupil_df = pd.read_csv(eye_states_path)
    illum_df = pd.read_csv(illum_levels_path)

    time_pupil = pupil_df["timestamp [ns]"].values
    pupil_left = pupil_df["pupil diameter left [mm]"].values
    pupil_right = pupil_df["pupil diameter right [mm]"].values

    time_illum = illum_df[illum_time_col].values
    illum = illum_df[illum_value_col].values

    # Interpolate illumination to pupil timestamps
    illum_interp = np.interp(time_pupil, time_illum, illum)

    # Smoothing for noise reduction
    # pupil_left = uniform_filter1d(pupil_left, size=30)
    # pupil_right = uniform_filter1d(pupil_right, size=30)
    # illum_interp = uniform_filter1d(illum_interp, size=30)

    # -------------------------
    # Linear regression: pupil left
    # pupil_left ≈ beta0 + beta1 * illumination
    # -------------------------
    X = np.column_stack([
        np.ones(len(illum_interp)),
        illum_interp
    ])

    r, p = stats.pearsonr(illum_interp, pupil_left)
    r_squared = r**2
    print("Illumination on Pupil left:")
    print(f"r^2 value: {r_squared}")
    print(f"p-value: {p}")

    beta, _, _, _ = np.linalg.lstsq(X, pupil_left, rcond=None)
    beta0, beta1 = beta

    # Predicted pupil from illumination
    pupil_left_pred = beta0 + beta1 * illum_interp

    # Residual (illumination-regressed pupil)
    pupil_left_corrected = pupil_left - pupil_left_pred + np.mean(pupil_left)

    # -------------------------
    # Linear regression: pupil right
    # pupil_right ≈ beta0 + beta1 * illumination
    # -------------------------
    X = np.column_stack([
        np.ones(len(illum_interp)),
        illum_interp
    ])

    r, p = stats.pearsonr(illum_interp, pupil_right)
    r_squared = r ** 2
    print("Illumination on Pupil right:")
    print(f"r^2 value: {r_squared}")
    print(f"p-value: {p}")
    print("")

    beta, _, _, _ = np.linalg.lstsq(X, pupil_right, rcond=None)
    beta0, beta1 = beta

    # Predicted pupil from illumination
    pupil_right_pred = beta0 + beta1 * illum_interp

    # Residual (illumination-regressed pupil)
    pupil_right_corrected = pupil_right - pupil_right_pred + np.mean(pupil_right)

    # -------------------------
    # Save output
    # -------------------------
    out_df = pd.DataFrame({
        "timestamp [ns]": time_pupil,
        "pupil_left": pupil_left_corrected,
        "pupil_right": pupil_right_corrected
    })

    out_df.to_csv(ts_folder / "light-corrected_pupil_diameters.csv", index=False)


if __name__ == "__main__":
    illum_levels_folder1 = Path(r"C:\Users\adamf\OneDrive\Desktop\MAPLab\r34project\Test5\glasses1_timeseries\2026-02-11_00-23-47-47b67625")
    illum_levels_folder2 = Path(r"C:\Users\adamf\OneDrive\Desktop\MAPLab\r34project\Test5\glasses2_timeseries\2026-02-11_00-23-44-b5e2cb4f")
    parent_folder = Path(r"C:\Users\adamf\OneDrive\Desktop\MAPLab\r34project\Test5")

    check_correlation_lighting_eye_contact(parent_folder, illum_levels_folder1)
    check_correlation_lighting_eye_contact(parent_folder, illum_levels_folder2)
