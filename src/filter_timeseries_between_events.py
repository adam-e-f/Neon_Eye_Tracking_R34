import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline



'''
Function get_filtered_series() for returning a timeseries csv containing only the rows with timestamps between
the start_time and end_time defined in the function's arguments. We parameterize:
    a) The folder containing the original timeseries (we use this as the destination folder too)
    b) The filename of the original timeseries (Should include the '.csv' at the end)
    c) The start time
    d) The end time
'''


def get_filtered_series(timeseries_folder, timeseries_filename, start_time, end_time):
    timeseries_to_filter = timeseries_folder / timeseries_filename
    # Load CSV
    ts_df = pd.read_csv(timeseries_to_filter)

    # Filter timeseries
    filtered_ts = ts_df[(ts_df["timestamp [ns]"] >= start_time) &
                        (ts_df["timestamp [ns]"] <= end_time)]

    # Define output filename
    output_filename = "filtered_" + timeseries_filename
    output_path = timeseries_folder / output_filename

    filtered_ts.to_csv(output_path, index=False)
    return output_path


'''
Function combine() for creating a large csv file containing all of the relevant data from the other timeseries
csv files. The function assumes we have already filtered for the correct time window in the other csv files, and 
that filtered_gaze.csv, filtered_3d_eye_states, filtered_imu, and filtered_gaze_on_face all exist. We parameterize:

    a) The folder containing all of the aforementioned timeseries (we use this as the destination folder too)
    b) The numerical threshold for change in gaze x position per second used as a minimum for blink thresholding
    
From there we will also create boxcar signal timeseries data for blinks, saccades, and fixations.

The order for saving the columns should be:
timestamp, blinks, saccades, fixations, smooth pursuit, gaze x, gaze x raw, gaze y, gaze y raw, pupil diameter right,
pupil diameter right raw, pupil diameter left, pupil diameter left raw, gaze on face, gaze on face raw,
roll, pitch, yaw, gyro x, gyro y, gyro z, acceleration x, acceleration y, acceleration z.

Note that for columns ending in "raw" that is just the data from before we interpolated over regions with blinks. So,
gaze x raw is just gaze x from the original file, and gaze x in this new file has blink interpolation.
'''


def combine(timeseries_folder, DERIV_THRESH):
    # Instantiate DataFrames
    df_gaze = pd.read_csv(timeseries_folder / "filtered_gaze.csv")
    df_3d_eye_states = pd.read_csv(timeseries_folder / "filtered_3d_eye_states.csv")
    df_gaze_on_face = pd.read_csv(timeseries_folder / "filtered_gaze_on_face.csv")
    df_imu = pd.read_csv(timeseries_folder / "filtered_imu.csv")
    df_blinks = pd.read_csv(timeseries_folder / "blinks.csv")
    df_saccades = pd.read_csv(timeseries_folder / "saccades.csv")
    df_fixations = pd.read_csv(timeseries_folder / "fixations.csv")

    time_window = df_gaze["timestamp [ns]"]
    time_s = (time_window - time_window.iloc[0]) * 1e-9

    # Initialize blinks boxcar signal
    signal_blinks = np.zeros(len(time_window), dtype=int)
    # Fill in 1s for all times that blinks are happening
    for _, row in df_blinks.iterrows():
        blink_start = row["start timestamp [ns]"]
        blink_end = row["end timestamp [ns]"]
        active = (time_window >= blink_start) & (time_window <= blink_end)
        signal_blinks[active] = 1

    # Initialize saccades boxcar signal
    signal_saccades = np.zeros(len(time_window), dtype=int)
    # Fill in 1s for all times that saccades are happening
    for _, row in df_saccades.iterrows():
        saccade_start = row["start timestamp [ns]"]
        saccade_end = row["end timestamp [ns]"]
        active = (time_window >= saccade_start) & (time_window <= saccade_end)
        signal_saccades[active] = 1

    # Initialize fixations boxcar signal
    signal_fixations = np.zeros(len(time_window), dtype=int)
    # Fill in 1s for all times that fixations are happening
    for _, row in df_fixations.iterrows():
        fixation_start = row["start timestamp [ns]"]
        fixation_end = row["end timestamp [ns]"]
        active = (time_window >= fixation_start) & (time_window <= fixation_end)
        signal_fixations[active] = 1

    # Initialize smooth pursuit boxcar signal
    signal_smooth_pursuit = (1 - signal_saccades) * (1 - signal_fixations)

    # Take derivative of gaze x for thresholding blinks
    gaze_x = df_gaze["gaze x [px]"]
    dx = np.gradient(gaze_x, time_s)
    abs_dx = np.abs(dx)

    # Each segment is (start_idx, end_idx)
    def get_segments(mask):
        segments = []
        in_seg = False
        start = None

        for i, val in enumerate(mask):
            if val == 1 and not in_seg:
                in_seg = True
                start = i
            elif val == 0 and in_seg:
                segments.append((start, i - 1))
                in_seg = False

        if in_seg:
            segments.append((start, len(mask) - 1))

        return segments

    blink_segments = get_segments(signal_blinks)

    # Use thresholding to shrink the blink segments
    refined_segments = []

    for start, end in blink_segments:
        seg_abs_dx = abs_dx[start:end + 1]

        high = seg_abs_dx > DERIV_THRESH

        if not np.any(high):
            continue  # discard blink if nothing strong inside

        idx = np.where(high)[0]
        new_start = start + idx[0]
        new_end = start + idx[-1]

        refined_segments.append((new_start, new_end))

    thresholded_blinks = np.zeros_like(signal_blinks)

    for s, e in refined_segments:
        thresholded_blinks[s:e + 1] = 1

    # Interpolate other data streams over the blink segments
    # Interp gaze x
    gaze_x_clean = gaze_x.copy()
    gaze_x_clean[thresholded_blinks == 1] = np.nan

    valid = ~np.isnan(gaze_x_clean)

    spline = UnivariateSpline(
        time_s[valid],
        gaze_x_clean[valid],
        k=1,  # cubic spline
        s=0
    )

    gaze_x_interp = spline(time_s)
    gaze_x_clean[thresholded_blinks == 1] = gaze_x_interp[thresholded_blinks == 1]

    # Interp gaze y
    gaze_y = df_gaze["gaze y [px]"]
    gaze_y_clean = gaze_y.copy()
    gaze_y_clean[thresholded_blinks == 1] = np.nan

    valid = ~np.isnan(gaze_y_clean)

    spline = UnivariateSpline(
        time_s[valid],
        gaze_y_clean[valid],
        k=1,  # linear, but turn this number up for quadratic or cubic spline
        s=0
    )

    gaze_y_interp = spline(time_s)
    gaze_y_clean[thresholded_blinks == 1] = gaze_y_interp[thresholded_blinks == 1]

    # Interp pupil diameter right
    pupil_right = df_3d_eye_states["pupil diameter right [mm]"]
    pupil_right_clean = pupil_right.copy()
    pupil_right_clean[thresholded_blinks == 1] = np.nan

    valid = ~np.isnan(pupil_right_clean)

    spline = UnivariateSpline(
        time_s[valid],
        pupil_right_clean[valid],
        k=1,  # degree of spline
        s=0
    )

    pupil_right_interp = spline(time_s)
    pupil_right_clean[thresholded_blinks == 1] = pupil_right_interp[thresholded_blinks == 1]

    # Interp pupil diameter left
    pupil_left = df_3d_eye_states["pupil diameter left [mm]"]
    pupil_left_clean = pupil_left.copy()
    pupil_left_clean[thresholded_blinks == 1] = np.nan

    valid = ~np.isnan(pupil_left_clean)

    spline = UnivariateSpline(
        time_s[valid],
        pupil_left_clean[valid],
        k=1,  # degree of spline
        s=0
    )

    pupil_left_interp = spline(time_s)
    pupil_left_clean[thresholded_blinks == 1] = pupil_left_interp[thresholded_blinks == 1]

    # Initialize gaze on face boxcar
    signal_gaze_on_face = df_gaze_on_face["gaze on face"]

    # Interpolate gaze on face for blinks (for each blink window, fill with the value right before the blink starts)
    signal_gaze_on_face_interp = signal_gaze_on_face.copy()
    in_blink = False
    start = None

    for i in range(len(signal_gaze_on_face)):
        if thresholded_blinks[i] == 1 and not in_blink:
            in_blink = True
            start = i
        elif thresholded_blinks[i] == 0 and in_blink:
            end = i - 1
            if start > 0:
                signal_gaze_on_face_interp[start:end + 1] = signal_gaze_on_face_interp[start - 1]
            in_blink = False

    if in_blink and start > 0:
        signal_gaze_on_face_interp[start:] = signal_gaze_on_face_interp[start - 1]

    # Convert from TRUE/FALSE to 1/0
    signal_gaze_on_face = signal_gaze_on_face.astype(int).to_numpy()
    signal_gaze_on_face_interp = signal_gaze_on_face_interp.astype(int).to_numpy()

    # At this point we have generated all the necessary timeseries columns, including those interpolated over blinks.
    # We now organize them for saving in a csv.
    combined_df = df_gaze[["timestamp [ns]"]]
    combined_df.insert(1, "blinks", thresholded_blinks)
    combined_df.insert(2, "saccades", signal_saccades)
    combined_df.insert(3, "fixations", signal_fixations)
    combined_df.insert(4, "smooth pursuit", signal_smooth_pursuit)
    combined_df.insert(5, "gaze x [px]", gaze_x_clean)
    combined_df.insert(6, "gaze x [px] (raw)", df_gaze["gaze x [px]"])
    combined_df.insert(7, "gaze y [px]", gaze_y_clean)
    combined_df.insert(8, "gaze y [px] (raw)", df_gaze["gaze y [px]"])
    combined_df.insert(9, "pupil diameter right [mm]", pupil_right_clean)
    combined_df.insert(10, "pupil diameter right [mm] (raw)", df_3d_eye_states["pupil diameter right [mm]"])
    combined_df.insert(11, "pupil diameter left [mm]", pupil_left_clean)
    combined_df.insert(12, "pupil diameter left [mm] (raw)", df_3d_eye_states["pupil diameter left [mm]"])
    combined_df.insert(13, "gaze on face", signal_gaze_on_face_interp)
    combined_df.insert(14, "gaze on face (raw)", signal_gaze_on_face)

    ''' These IMU readings seem like they may have a different polling rate than the rest, causing their time
        series to have a different number of columns. For now we exclude them.
        
    combined_df.insert(15, "roll [deg]", df_imu["roll [deg]"])
    combined_df.insert(16, "pitch [deg]", df_imu["pitch [deg]"])
    combined_df.insert(17, "yaw [deg]", df_imu["yaw [deg]"])
    combined_df.insert(18, "gyro x [deg/s]", df_imu["gyro x [deg/s]"])
    combined_df.insert(19, "gyro y [deg/s]", df_imu["gyro y [deg/s]"])
    combined_df.insert(20, "gyro z [deg/s]", df_imu["gyro z [deg/s]"])
    combined_df.insert(21, "acceleration x [g]", df_imu["acceleration x [g]"])
    combined_df.insert(22, "acceleration y [g]", df_imu["acceleration y [g]"])
    combined_df.insert(23, "acceleration z [g]", df_imu["acceleration z [g]"])
    '''

    # Save as csv
    output_filename = "combined_timeseries.csv"
    output_path = timeseries_folder / output_filename
    combined_df.to_csv(output_path, index=False)
    return output_path
