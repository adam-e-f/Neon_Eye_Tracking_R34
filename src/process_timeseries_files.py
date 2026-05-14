import pandas as pd
import ast
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
import filter_timeseries_between_events


def show_temp_graph(signal, time, y_axis_label="Signal"):
    plt.plot(time, signal)  # t = time array, x = 1D signal
    plt.xlabel("Time")
    plt.ylabel(y_axis_label)
    plt.show()


'''
Takes the output from run_gaze_on_landmarks.py and generates a new csv file with boxcar timeseries for each of the
facial landmarks
'''


def extract_landmark_timeseries(landmarks_csv_path, output_folder):
    landmarks_labels = ["Not on landmark", "eye left", "eye right", "nose", "mouth"]
    time_col = "timestamp [ns]"
    landmarks_col = "landmark"

    # Load data
    df = pd.read_csv(landmarks_csv_path)

    # --- Step 1: Convert string -> actual list ---
    def parse_landmarks(x):
        if pd.isna(x):
            return []

        # Case 1: already a list
        if isinstance(x, list):
            return x

        # Case 2: string representation of list
        try:
            return ast.literal_eval(x)
        except:
            # fallback: assume delimiter-separated string
            return [item.strip() for item in str(x).split(",")]

    df[landmarks_col] = df[landmarks_col].apply(parse_landmarks)

    # --- Step 2: One-hot encode efficiently ---
    # explode → one row per label → get dummies → group back
    exploded = df[[time_col, landmarks_col]].explode(landmarks_col)

    dummies = pd.get_dummies(exploded[landmarks_col])

    # Combine with timestamps
    encoded = pd.concat([exploded[[time_col]], dummies], axis=1)

    # Aggregate back to original timepoints
    encoded = encoded.groupby(time_col).max().reset_index()

    # --- Step 3: Ensure all columns exist ---
    for label in landmarks_labels:
        if label not in encoded.columns:
            encoded[label] = 0

    # Order columns
    encoded = encoded[[time_col] + landmarks_labels]

    encoded[landmarks_labels] = encoded[landmarks_labels].astype(int)

    output_csv = output_folder / "gaze_on_facial_landmarks.csv"

    # Save
    encoded.to_csv(output_csv, index=False)

    return encoded


'''
Function which takes a timeseries file and a blinks csv path as input and generates a new timeseries which interpolates
over blink regions. Also takes a threshold as input which narrows down the blink regions using derivative of gaze x.
For boxcar signals, the blink region is filled with the value from right before the blink started.
For continuous signals, we do linear interpolation over the blink region.

ASSUMES that filtered_gaze.csv exists in the same folder as blinks.csv.
ASSUMES that we have filtered our timeseries already, meaning we are covering the same timepoints as filtered_gaze.csv

NOTE that since is_boxcar is passed as an argument, we can do interpolation over multiple columns from the same csv
file only if they are either all boxcar signals or all continuous. If you need interpolation over multiple signals from
the same file, but some are continuous and some are boxcars, you will need to do them separately (i.e., run this
function multiple times, with different column_labels lists). However, due to the way the output file is named, running
this function for a second time on the same input file will overwrite the output of the first run. So you will need to
rename the first output file before re-running the function.
'''
# TODO: Deal with the case where is_boxcar is false. Currently the function will only work for boxcar signals.


def interpolate_blinks(timeseries_path, column_labels: list, blinks_path, is_boxcar, DERIV_THRESH):

    # blinks_folder might sometimes be the same as output_folder. We generalize for cases where they aren't the same.
    # We create the new csv in output_folder. We assume filtered_gaze.csv is in blinks_folder.
    output_folder = timeseries_path.parent
    blinks_folder = blinks_path.parent

    # Load data frames
    df1 = pd.read_csv(timeseries_path)
    df_blinks = pd.read_csv(blinks_path)

    time_window = df1["timestamp [ns]"]

    # Initialize boxcar signal
    signal_blinks = np.zeros(len(time_window), dtype=int)

    # Fill in 1s for all times that blinks are happening
    for _, row in df_blinks.iterrows():
        blink_start = row["start timestamp [ns]"]
        blink_end = row["end timestamp [ns]"]

        active = (time_window >= blink_start) & (time_window <= blink_end)
        signal_blinks[active] = 1

    # Take derivative of gaze x for thresholding blinks
    df_gaze = pd.read_csv(blinks_folder / "filtered_gaze.csv")
    time_window_gaze = df_gaze["timestamp [ns]"]
    time_s_gaze = (time_window_gaze - time_window_gaze.iloc[0]) * 1e-9
    gaze_x = df_gaze["gaze x [px]"]
    dx = np.gradient(gaze_x, time_s_gaze)
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

    refined_blinks = np.zeros_like(signal_blinks)

    for s, e in refined_segments:
        refined_blinks[s:e + 1] = 1

    out_df = df1[["timestamp [ns]"]].copy()
    for label in column_labels:
        if is_boxcar:
            def fill_blinks_with_prev(signal, blink_mask):
                signal = signal.copy()
                in_blink = False
                start = None

                for i in range(len(signal)):
                    if blink_mask[i] == 1 and not in_blink:
                        in_blink = True
                        start = i
                    elif blink_mask[i] == 0 and in_blink:
                        end = i - 1
                        if start > 0:
                            signal[start:end + 1] = signal[start - 1]
                        in_blink = False

                if in_blink and start > 0:
                    signal[start:] = signal[start - 1]

                return signal

            # Replace all values in blink regions with whichever value came right before the blink
            col_values = df1[label]
            col_values_filled = fill_blinks_with_prev(col_values, refined_blinks)

            out_df[label] = col_values_filled

    timeseries_filename = os.path.splitext(os.path.basename(timeseries_path))[0]
    output_filename = timeseries_filename + "_interpolated_blinks.csv"
    out_path = output_folder / output_filename

    out_df.to_csv(out_path, index=False)
    return out_path


'''
Takes as input the paths to 2 landmarks timeseries csv files. Assumes 'eye left' and 'eye right' are column titles in
both and also assumes they have both been filtered in terms of start time and end time.
'''


def mutual_eye_contact(landmarks_path1, landmarks_path2, out_path):
    # Load Landmarks information
    df1 = pd.read_csv(landmarks_path1)
    df2 = pd.read_csv(landmarks_path2)

    t1 = df1["timestamp [ns]"]
    el1 = df1["eye left"]
    er1 = df1["eye right"]
    el2 = df2["eye left"]
    er2 = df2["eye right"]

    # Truncate to shorter length (this is needed because sometimes the files differ in length by 1 or 2 rows)
    n = min(len(el1), len(el2))
    t1 = t1.iloc[:n]
    el1 = el1.iloc[:n]
    er1 = er1.iloc[:n]
    el2 = el2.iloc[:n]
    er2 = er2.iloc[:n]

    person1_looking = el1.astype(bool) | er1.astype(bool)
    person2_looking = el2.astype(bool) | er2.astype(bool)

    eye_contact = (person1_looking & person2_looking).astype(int)
    out_df = t1.to_frame()
    out_df["eye contact"] = eye_contact
    out_df.to_csv(out_path, index=False)


'''
Does exactly as title suggests. We use linear interpolation, meaning when eye contact goes from 0 to 1 or 1 to 0, we 
will have some values in the interpolation that are between 0 and 1.
For the sake of the analyses, we exclude these points.

HIGHLY RECOMMENDED to first use functions from regress_out_lighting.py to acquire a timeseries of pupil diameter that
accounts for the effects of lighting on pupil diameter, and then use light_corrected_pupil_diameters.csv as input here.

We save the new eye contact csv file in the same folder as the pupil residual csv file.
'''


def interpolate_eye_contact_onto_pupil_timestamps(pupil_residual_path, eye_contact_path):
    output_folder = pupil_residual_path.parent

    # Load data
    pupil_df = pd.read_csv(pupil_residual_path)
    eye_contact_df = pd.read_csv(eye_contact_path)

    time_pupil = pupil_df["timestamp [ns]"].values
    time_eye_contact = eye_contact_df["timestamp [ns]"].values

    eye_contact = eye_contact_df["eye contact"].values

    # Interpolate illumination to pupil timestamps
    eye_contact_interp = np.interp(time_pupil, time_eye_contact, eye_contact)

    # -------------------------
    # Save output
    # -------------------------
    out_df = pd.DataFrame({
        "timestamp [ns]": time_pupil,
        "eye contact": eye_contact_interp
    })

    out_df.to_csv(output_folder / "eye_contact_w_pupil_timestamps.csv", index=False)


'''
Given the time series of eye contact, this function generates a csv of the start time and end time of each eye contact
that took place for >1 sample (i.e., excludes eye contact that appeared for only one frame of video before breaking).
This format is identical to blinks.csv and saccades.csv.
'''


def get_eye_contact_events(mutual_eye_contact_path, output_csv="eye_contact_events.csv"):
    parent_folder = mutual_eye_contact_path.parent

    df1 = pd.read_csv(mutual_eye_contact_path)
    timestamps = df1["timestamp [ns]"].to_numpy()
    eye_contact = df1["eye contact"].to_numpy()

    events = []
    in_event = False
    start_idx = None
    event_id = 1

    for i in range(len(eye_contact)):
        # Start of event
        if eye_contact[i] == 1 and not in_event:
            in_event = True
            start_idx = i

        # End of event
        elif eye_contact[i] == 0 and in_event:
            end_idx = i - 1
            event_length_samples = end_idx - start_idx + 1

            # Keep only events lasting >1 sample
            if event_length_samples > 1:
                start_time = timestamps[start_idx]
                end_time = timestamps[end_idx]
                length_ms = (end_time - start_time) / 1e6  # ns -> ms

                events.append({
                    "event_id": event_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "length [ms]": length_ms
                })
                event_id += 1

            in_event = False
            start_idx = None

    # Handle event that continues until final sample
    if in_event:
        end_idx = len(eye_contact) - 1
        event_length_samples = end_idx - start_idx + 1

        if event_length_samples > 1:
            start_time = timestamps[start_idx]
            end_time = timestamps[end_idx]
            length_ms = (end_time - start_time) / 1e6  # ns -> ms

            events.append({
                "event_id": event_id,
                "start_time": start_time,
                "end_time": end_time,
                "length [ms]": length_ms
            })

    events_df = pd.DataFrame(events)
    events_df.to_csv(parent_folder / output_csv, index=False)

    return events_df


'''
This function takes as input the dataframes for pupil diameter time series data and for eye contact events (in start
time / end time format). It epochs the pupil data by eye contact events, excluding trials shorter than the length
defined by window_length_ms, and outputs a csv file with all the epoched data, as well as graphs for all time windows
overlain and for the average of pupillary dynamics across the time window.

By default settings we do this only for pupil diameter right. To do this for pupil diameter left, just change the value
of the 'pupil_value_col' and 'prefix' arguments.
'''


def epoch_pupil_by_eye_contact(
    pupil_df,
    events_df,
    output_dir_path:Path,
    pupil_time_col="timestamp [ns]",
    pupil_value_col="pupil diameter right [mm]",
    event_start_col="start_time",
    event_length_col="length [ms]",
    window_length_ms=500,
    tmin_ms=-500,
    tmax_ms=1000,
    dt_ms=10,
    baseline_correct=True,
    prefix="pupil_right"
):
    """
    Epoch pupil data around eye-contact onset events.

    Parameters
    ----------
    pupil_df : pd.DataFrame
        DataFrame containing pupil timestamps and pupil values.
    events_df : pd.DataFrame
        DataFrame containing eye-contact events.
    pupil_time_col : str
        Column name for pupil timestamps (ns).
    pupil_value_col : str
        Column name for pupil diameter values.
    event_start_col : str
        Column name for eye-contact onset timestamps (ns).
    event_length_col : str
        Column name for eye-contact duration (ms).
    window_length_ms : float
        Minimum eye-contact duration required for inclusion.
    tmin_ms : float
        Epoch start relative to event onset (ms).
    tmax_ms : float
        Epoch end relative to event onset (ms).
    dt_ms : float
        Sampling interval for interpolation grid (ms).
    baseline_correct : bool
        Whether to subtract the value at onset.

    Returns
    -------
    dict with:
        epochs : np.ndarray
            Shape (n_events, n_timepoints)
        rel_time_ms : np.ndarray
            Relative time axis in ms
        mean : np.ndarray
            Mean across epochs
        sem : np.ndarray
            Standard error across epochs
        included_events : pd.DataFrame
            Events surviving filtering
    """

    # -------------------------
    # Extract pupil data
    # -------------------------
    time_ns = pupil_df[pupil_time_col].to_numpy()
    pupil = pupil_df[pupil_value_col].to_numpy()

    # convert ns -> ms for easier arithmetic
    time_ms = time_ns / 1e6

    # remove NaNs before interpolation
    valid = ~(np.isnan(time_ms) | np.isnan(pupil))
    time_ms = time_ms[valid]
    pupil = pupil[valid]

    # interpolation function
    interp_func = interp1d(
        time_ms,
        pupil,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    # -------------------------
    # Filter events by duration
    # -------------------------
    included_events = events_df[
        events_df[event_length_col] >= window_length_ms
    ].copy()

    event_times_ms = included_events[event_start_col].to_numpy() / 1e6

    # -------------------------
    # Create relative time axis
    # -------------------------
    rel_time_ms = np.arange(tmin_ms, tmax_ms + dt_ms, dt_ms)

    # -------------------------
    # Extract epochs
    # -------------------------
    epochs = []

    for ev in event_times_ms:
        sample_times = ev + rel_time_ms
        epoch = interp_func(sample_times)
        epochs.append(epoch)

    epochs = np.array(epochs)

    # -------------------------
    # Baseline correction
    # -------------------------
    if baseline_correct:
        baseline_mask = rel_time_ms < 0
        baseline = np.nanmean(
            epochs[:, baseline_mask],
            axis=1,
            keepdims=True
        )
        epochs = epochs - baseline

    # -------------------------
    # Compute summary stats
    # -------------------------
    mean_response = np.nanmean(epochs, axis=0)

    n_valid = np.sum(~np.isnan(epochs), axis=0)
    sem_response = np.nanstd(epochs, axis=0) / np.sqrt(n_valid)

    # -------------------------
    # Save epochs to CSV
    # -------------------------
    columns = [f"{t:.1f}" for t in rel_time_ms]
    epochs_df = pd.DataFrame(epochs, columns=columns)

    output_dir_path.mkdir(exist_ok=True)

    epochs_csv_path = output_dir_path / f"{prefix}_epochs.csv"
    epochs_df.to_csv(epochs_csv_path, index=False)

    # -------------------------
    # Plot 1: All epochs
    # -------------------------
    plt.figure()

    for ep in epochs:
        plt.plot(rel_time_ms, ep, alpha=0.3)

    plt.axvline(0, linestyle="--")
    plt.xlabel("Time (ms)")
    plt.ylabel("Pupil (baseline-corrected)")
    plt.title("All Eye Contact Epochs")

    plot1_path = output_dir_path / f"{prefix}_all_epochs.png"
    plt.savefig(plot1_path)
    plt.close()

    # -------------------------
    # Plot 2: Mean ± SEM
    # -------------------------
    plt.figure()

    plt.plot(rel_time_ms, mean_response)
    plt.fill_between(
        rel_time_ms,
        mean_response - sem_response,
        mean_response + sem_response,
        alpha=0.3
    )

    plt.axvline(0, linestyle="--")
    plt.xlabel("Time (ms)")
    plt.ylabel("Pupil (baseline-corrected)")
    plt.title("Average Pupil Response to Eye Contact")

    plot2_path = output_dir_path / f"{prefix}_mean_response.png"
    plt.savefig(plot2_path)
    plt.close()

    return {
        "epochs": epochs,
        "rel_time_ms": rel_time_ms,
        "mean": mean_response,
        "sem": sem_response,
        "included_events": included_events
    }


# For testing
if __name__ == "__main__":
    pupil_residual_path = Path(r"C:\Users\adamf\OneDrive\Desktop\MAPLab\r34project\Test4\glasses1_timeseries\2026-02-11_00-23-47-47b67625\filtered_light-corrected_pupil_diameters.csv")
    eye_contact_path = Path(r"C:\Users\adamf\OneDrive\Desktop\MAPLab\r34project\Test4\mutual_eye_contact.csv")
    interpolate_eye_contact_onto_pupil_timestamps(pupil_residual_path, eye_contact_path)

