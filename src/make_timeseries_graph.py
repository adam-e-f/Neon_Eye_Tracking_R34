import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from pathlib import Path


class GraphObj:
    def __init__(self, axis_title, is_step, show_blinks, file_path, column_name, range):
        self.axis_title = axis_title                    # string
        self.is_step = is_step                          # boolean
        self.show_blinks = show_blinks                  # boolean
        self.file_path = file_path                      # Path object (path to the csv)
        self.column_name = column_name                  # string (the name of the timeseries column in the csv)
        self.range = range                              # tuple of float or int (y-axis range)




'''
All of the rest of this file after this first function are older graphing methods from before generalizing.
'''


def make_graphs(graphs_list: list[GraphObj], save_folder: Path, output_filename):
    num_graphs = len(graphs_list)
    configs = []
    time_s = 0

    # Initialize DataFrames and Configs
    for i in range(len(graphs_list)):
        df1 = pd.read_csv(graphs_list[i].file_path)

        # Use the first graph's timeseries file to define the timebase for all the graphs
        if i == 0:
            time_window = df1["timestamp [ns]"]
            # Convert time to seconds for plotting
            time_s = (time_window - time_window.iloc[0]) * 1e-9

        if graphs_list[i].is_step:
            configs.append(
                dict(
                    df=df1, time=time_s, signal=df1[graphs_list[i].column_name], ylim=graphs_list[i].range,
                    ylabel=graphs_list[i].axis_title, kind="step", show_blinks=graphs_list[i].show_blinks
                )
            )
        else:
            configs.append(
                dict(df=df1, time_col="timestamp [ns]", val_col=graphs_list[i].column_name,
                     transform=lambda s: s.astype(float), ylim=graphs_list[i].range,
                     ylabel=graphs_list[i].axis_title, kind="line", show_blinks=graphs_list[i].show_blinks
                )
            )

    if num_graphs == 1:
        fig, axes = plt.subplots(1, 1, sharex=True, figsize=(4, 4))
    else:
        fig, axes = plt.subplots(num_graphs, 1, sharex=True, figsize=(4*num_graphs, 8))

    # Create and save plot
    for ax, cfg in zip(axes, configs):

        if cfg["kind"] != "step":
            t = cfg["df"][cfg["time_col"]]
            y_raw = cfg["df"][cfg["val_col"]]
            y = cfg["transform"](y_raw)
            t_s = (t - t.iloc[0]) * 1e-9
            ax.step(t_s, y, where="post")
        else:
            t_s = cfg["time"]
            y = cfg["signal"]
            ax.plot(t_s, y)

        ymin, ymax = cfg["ylim"]
        if ymin != -1 or ymax != -1:   # If range = (-1, -1), just fit the range to the data
            ax.set_ylim(cfg["ylim"])
        ax.set_ylabel(cfg["ylabel"])

        # Display blink windows
        if cfg["show_blinks"]:
            signal_blinks = cfg["df"]["blinks"]
            ymin, ymax = ax.get_ylim()
            blink_scaled = signal_blinks * (ymax - ymin) + ymin
            ax.step(time_s, blink_scaled, where="post",
                    color="green", alpha=0.2)

    # Some cosmetic modification
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (s)")
    plt.subplots_adjust(hspace=0.25, left=0.15)

    # Make output directory
    save_folder.mkdir(exist_ok=True)
    plt.savefig(save_folder / output_filename, dpi=300, bbox_inches="tight")
    plt.close()


'''
Function graph() for plotting the timeseries and saving as png. We parameterize:
    a) The folder containing the timeseries
    b) The filename of the timeseries (Should include the '.csv' at the end)
    c) The name of the variable within the timeseries that needs graphing
    d) The minimum value for the y-axis range
    e) The maximum value for the y-axis range
    f) The graph title
    g) The Y-axis label
'''


def graph(timeseries_folder, timeseries_filename, plot_title, y_axis_title, var_name, min_val=-1, max_val=-1):
    csv_path = timeseries_folder / timeseries_filename

    df = pd.read_csv(csv_path)

    # Extract columns
    time_ns = df["timestamp [ns]"]
    var_vals = df[var_name]

    # Convert time to seconds (optional but recommended)
    time_s = (time_ns - time_ns.iloc[0]) * 1e-9

    # Plot
    plt.figure()
    plt.plot(time_s, var_vals)
    plt.xlabel("Time (s)")
    plt.ylabel(y_axis_title)
    plt.title(plot_title)

    if min_val != -1 or max_val != -1:
        plt.ylim(min_val, max_val)  # force y-axis range

    # Make output directory
    output_dir = timeseries_folder / "graphs"
    output_dir.mkdir(exist_ok=True)

    # Clean filename and add extension
    filename = plot_title.replace(" ", "_") + ".png"
    output_path = output_dir / filename

    plt.savefig(output_path, dpi=300, bbox_inches="tight")


'''
Function graph_saccades() for plotting the boxcar graph of all the saccades over an interval and saving as png.
Before running this function, one should filter another timeseries (i.e., gaze.csv --> filtered_gaze.csv) over the
interval in question, since we are using the timestamps from that csv file to generate this plot. We parameterize:
    a) The folder containing the timeseries
    b) The filename of the filtered timeseries (optional; set to "filtered_gaze.csv by default")
Everything else is assumed to be standard, i.e., the file containing the saccades is called "saccades.csv" and exists
within timeseries_folder.
'''


def graph_saccades(timeseries_folder, timeseries_base_name="filtered_gaze.csv"):
    # Load saccade intervals
    saccades_path = timeseries_folder / "saccades.csv"
    saccades_df = pd.read_csv(saccades_path)

    # Load timeseries we are aligning to (for time base)
    ts_path = timeseries_folder / timeseries_base_name
    ts_df = pd.read_csv(ts_path)

    time_window = ts_df["timestamp [ns]"]

    # Initialize boxcar signal
    signal = np.zeros(len(time_window), dtype=int)

    # Fill in 1s for all times that saccades are happening
    for _, row in saccades_df.iterrows():
        saccade_start = row["start timestamp [ns]"]
        saccade_end = row["end timestamp [ns]"]

        active = (time_window >= saccade_start) & (time_window <= saccade_end)
        signal[active] = 1

    # Convert time to seconds for plotting
    time_s = (time_window - time_window.iloc[0]) * 1e-9

    # Plot
    plt.figure()
    plt.step(time_s, signal, where="post")
    plt.ylim(-2, 3)
    plt.yticks([0, 1], ["0", "1"])
    plt.xlabel("Time (s)")
    plt.ylabel("Saccades")
    plt.title("Saccades Over Time")

    # Make output directory
    output_dir = timeseries_folder / "graphs"
    output_dir.mkdir(exist_ok=True)

    # Save png image of the plot
    filename = "Saccades_Over_Time.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def graph_blinks(timeseries_folder, timeseries_base_name="filtered_gaze.csv"):
    # Load saccade intervals
    blinks_path = timeseries_folder / "blinks.csv"
    blinks_df = pd.read_csv(blinks_path)

    # Load timeseries we are aligning to (for time base)
    ts_path = timeseries_folder / timeseries_base_name
    ts_df = pd.read_csv(ts_path)

    time_window = ts_df["timestamp [ns]"]

    # Initialize boxcar signal
    signal = np.zeros(len(time_window), dtype=int)

    # Fill in 1s for all times that blinks are happening
    for _, row in blinks_df.iterrows():
        blink_start = row["start timestamp [ns]"]
        blink_end = row["end timestamp [ns]"]

        active = (time_window >= blink_start) & (time_window <= blink_end)
        signal[active] = 1

    # Convert time to seconds for plotting
    time_s = (time_window - time_window.iloc[0]) * 1e-9

    # Plot
    plt.figure()
    plt.step(time_s, signal, where="post")
    plt.ylim(-2, 3)
    plt.yticks([0, 1], ["0", "1"])
    plt.xlabel("Time (s)")
    plt.ylabel("Blinks")
    plt.title("Blinks Over Time")

    # Make output directory
    output_dir = timeseries_folder / "graphs"
    output_dir.mkdir(exist_ok=True)

    # Save png image of the plot
    filename = "Blinks_Over_Time.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def graph_gaze_on_face(timeseries_folder, y_axis_title, timeseries_filename="filtered_gaze_on_face.csv"):
    csv_path = timeseries_folder / timeseries_filename
    df = pd.read_csv(csv_path)

    time_window = df["timestamp [ns]"]
    signal = df["gaze on face"]

    # Convert time to seconds for plotting
    time_s = (time_window - time_window.iloc[0]) * 1e-9

    # Plot
    plt.figure()
    plt.step(time_s, signal, where="post")
    plt.ylim(-2, 3)
    plt.yticks([0, 1], ["0", "1"])
    plt.xlabel("Time (s)")
    plt.ylabel(y_axis_title)
    title = f"{y_axis_title} Over Time"
    plt.title(title)

    # Make output directory
    output_dir = timeseries_folder / "graphs"
    output_dir.mkdir(exist_ok=True)

    # Save png image of the plot
    filename = title.replace(" ", "_") + ".png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


# Helper function.
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


# Graph of both people's gaze on face data + the multiplication, i.e., both gazes on faces, all stacked together.
# ASSUMES filtered_gaze_on_face exists in both timeseries folders.
def combined_gazes_on_faces(timeseries_folder_1, timeseries_folder_2, output_folder):
    file1_path = timeseries_folder_1 / "filtered_gaze_on_face.csv"
    file2_path = timeseries_folder_2 / "filtered_gaze_on_face.csv"

    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Extract columns
    t1 = df1["timestamp [ns]"]
    s1 = df1["gaze on face"]
    s2 = df2["gaze on face"]

    # Convert to boolean robustly
    b1 = s1.astype(str).str.strip().str.upper() == "TRUE"
    b2 = s2.astype(str).str.strip().str.upper() == "TRUE"

    # Truncate to shorter length (this is needed because sometimes the files differ in length by 1 or 2 rows)
    n = min(len(b1), len(b2))
    t1 = t1.iloc[:n]
    b1 = b1.iloc[:n]
    b2 = b2.iloc[:n]

    # Logical AND
    result = b1 & b2

    # Build output DataFrame
    out_df = pd.DataFrame({
        "timestamp [ns]": t1,
        "gaze on face": result
    })

    csv3_path = output_folder / "mutual_gaze.csv"
    out_df.to_csv(csv3_path, index=False)

    df3 = pd.read_csv(csv3_path)

    time_window = df1["timestamp [ns]"]

    # Convert time to seconds for plotting
    time_s = (time_window - time_window.iloc[0]) * 1e-9

    # Load blink intervals for subject 1
    blinks1_df = pd.read_csv(timeseries_folder_1 / "blinks.csv")

    # Initialize boxcar signal
    signal_blinks1 = np.zeros(len(time_window), dtype=int)

    # Fill in 1s for all times that saccades are happening
    for _, row in blinks1_df.iterrows():
        blink_start = row["start timestamp [ns]"]
        blink_end = row["end timestamp [ns]"]

        active = (time_window >= blink_start) & (time_window <= blink_end)
        signal_blinks1[active] = 1

    # Load blink intervals for subject 2
    blinks2_df = pd.read_csv(timeseries_folder_2 / "blinks.csv")

    # Initialize boxcar signal
    signal_blinks2 = np.zeros(len(time_window), dtype=int)

    # Fill in 1s for all times that saccades are happening
    for _, row in blinks2_df.iterrows():
        blink_start = row["start timestamp [ns]"]
        blink_end = row["end timestamp [ns]"]

        active = (time_window >= blink_start) & (time_window <= blink_end)
        signal_blinks2[active] = 1

    # PLOTTING
    dfs = [df1, df2, df3]
    labels = ["Subject 1's \nGaze on Face", "Subject 2's \nGaze on Face", "Mutual Gaze"]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 6))

    for ax, df, label in zip(axes, dfs, labels):
        time_window = df["timestamp [ns]"]
        var_vals = df["gaze on face"]
        signal = (var_vals.astype(str).str.strip().str.upper() == "TRUE").astype(int)

        time_s = (time_window - time_window.iloc[0]) * 1e-9

        ax.step(time_s, signal, where="post")
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1])
        ax.set_ylabel(label)

    for ax in axes:
        # Remove box around each subplot
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ADD BLINK INTERVALS
    blink1_scaled = (signal_blinks1 * 2) - 0.5
    blink2_scaled = (signal_blinks2 * 2) - 0.5
    axes[0].step(time_s, blink1_scaled, where="post",
            color="green", alpha=0.7)
    axes[1].step(time_s, blink2_scaled, where="post",
                 color="green", alpha=0.7)

    axes[-1].set_xlabel("Time (s)")

    plt.subplots_adjust(hspace=0.25, left=0.15)

    output_path = output_folder / "combined_gaze_on_face.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def combined_gazes_on_faces_interpolate_blinks(timeseries_folder_1, timeseries_folder_2, output_folder, DERIV_THRESH):
    # First we load blinks and do thresholding for glasses1.
    df1 = pd.read_csv(timeseries_folder_1 / "filtered_gaze.csv")
    time_window = df1["timestamp [ns]"]

    # Convert time to seconds for plotting
    time_s = (time_window - time_window.iloc[0]) * 1e-9

    # Load blink intervals
    blinks_df = pd.read_csv(timeseries_folder_1 / "blinks.csv")

    # Initialize boxcar signal
    signal_blinks = np.zeros(len(time_window), dtype=int)

    # Fill in 1s for all times that saccades are happening
    for _, row in blinks_df.iterrows():
        blink_start = row["start timestamp [ns]"]
        blink_end = row["end timestamp [ns]"]

        active = (time_window >= blink_start) & (time_window <= blink_end)
        signal_blinks[active] = 1

    # Take derivative of gaze x for thresholding blinks
    gaze_x = df1["gaze x [px]"]
    dx = np.gradient(gaze_x, time_s)
    abs_dx = np.abs(dx)

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

    refined_blinks1 = np.zeros_like(signal_blinks)

    for s, e in refined_segments:
        refined_blinks1[s:e + 1] = 1

    # Now we load blinks and do thresholding for glasses2.

    df2 = pd.read_csv(timeseries_folder_2 / "filtered_gaze.csv")
    time_window = df2["timestamp [ns]"]

    # Convert time to seconds for plotting
    time_s = (time_window - time_window.iloc[0]) * 1e-9

    # Load blink intervals
    blinks_df = pd.read_csv(timeseries_folder_2 / "blinks.csv")

    # Initialize boxcar signal
    signal_blinks = np.zeros(len(time_window), dtype=int)

    # Fill in 1s for all times that saccades are happening
    for _, row in blinks_df.iterrows():
        blink_start = row["start timestamp [ns]"]
        blink_end = row["end timestamp [ns]"]

        active = (time_window >= blink_start) & (time_window <= blink_end)
        signal_blinks[active] = 1

    # Take derivative of gaze x for thresholding blinks
    gaze_x = df2["gaze x [px]"]
    dx = np.gradient(gaze_x, time_s)
    abs_dx = np.abs(dx)

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

    refined_blinks2 = np.zeros_like(signal_blinks)

    for s, e in refined_segments:
        refined_blinks2[s:e + 1] = 1

    # Load Gaze on Face information
    file1_path = timeseries_folder_1 / "filtered_gaze_on_face.csv"
    file2_path = timeseries_folder_2 / "filtered_gaze_on_face.csv"

    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Extract columns
    t1 = df1["timestamp [ns]"]
    s1 = df1["gaze on face"]
    s2 = df2["gaze on face"]

    # Convert to boolean robustly
    b1 = s1.astype(str).str.strip().str.upper() == "TRUE"
    b2 = s2.astype(str).str.strip().str.upper() == "TRUE"

    # Truncate to shorter length (this is needed because sometimes the files differ in length by 1 or 2 rows)
    n = min(len(b1), len(b2))
    t1 = t1.iloc[:n]
    b1 = b1.iloc[:n]
    b2 = b2.iloc[:n]

    # The following section converts b1 and b2 to numerical form, interpolates the blinks by filling the blink ranges
    # with whatever value was present right before the blink, then converts b1 and b2 back to boolean form.

    # Convert to 0/1 arrays
    b1_num = b1.astype(int).to_numpy()
    b2_num = b2.astype(int).to_numpy()

    blink1 = refined_blinks1[:n]
    blink2 = refined_blinks2[:n]

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

    # Fill blink regions
    b1_filled = fill_blinks_with_prev(b1_num, blink1)
    b2_filled = fill_blinks_with_prev(b2_num, blink2)

    # Convert back to boolean
    b1 = b1_filled.astype(bool)
    b2 = b2_filled.astype(bool)

    # Logical AND
    result = b1 & b2

    # Build output DataFrame
    out_df = pd.DataFrame({
        "timestamp [ns]": t1,
        "gaze on face": result
    })

    csv3_path = output_folder / "mutual_gaze_interpolated_blinks.csv"
    out_df.to_csv(csv3_path, index=False)

    df3 = pd.read_csv(csv3_path)

    time_window = df1["timestamp [ns]"]

    # Convert time to seconds for plotting
    time_s = (time_window - time_window.iloc[0]) * 1e-9

    # Load blink intervals for subject 1
    blinks1_df = pd.read_csv(timeseries_folder_1 / "blinks.csv")

    # Initialize boxcar signal
    signal_blinks1 = np.zeros(len(time_window), dtype=int)

    # Fill in 1s for all times that blinks are happening
    for _, row in blinks1_df.iterrows():
        blink_start = row["start timestamp [ns]"]
        blink_end = row["end timestamp [ns]"]

        active = (time_window >= blink_start) & (time_window <= blink_end)
        signal_blinks1[active] = 1

    # Load blink intervals for subject 2
    blinks2_df = pd.read_csv(timeseries_folder_2 / "blinks.csv")

    # Initialize boxcar signal
    signal_blinks2 = np.zeros(len(time_window), dtype=int)

    # Fill in 1s for all times that blinks are happening
    for _, row in blinks2_df.iterrows():
        blink_start = row["start timestamp [ns]"]
        blink_end = row["end timestamp [ns]"]

        active = (time_window >= blink_start) & (time_window <= blink_end)
        signal_blinks2[active] = 1

    # Define all 3 data plots in terms of names, data streams, and data ranges
    signal3 = (df3["gaze on face"].astype(str).str.strip().str.upper() == "TRUE").astype(int)
    configs = [
        # Gaze on face subject 1
        dict(time=time_s, signal=b1_filled,
             ylim=(-0.5, 1.5), ylabel="Subject 1's \nGaze on Face", kind="step"),

        # Gaze on face subject 2
        dict(time=time_s, signal=b2_filled,
             ylim=(-0.5, 1.5), ylabel="Subject 2's \nGaze on Face", kind="step"),

        # Mutual gaze
        dict(time=time_s, signal=signal3,
             ylim=(-0.5, 1.5), ylabel="Mutual Gaze", kind="step")
    ]

    # Create and save plot

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
    for ax, cfg in zip(axes, configs):

        if "df" in cfg:
            t = cfg["df"][cfg["time_col"]]
            y_raw = cfg["df"][cfg["val_col"]]
            y = cfg["transform"](y_raw)
            t_s = (t - t.iloc[0]) * 1e-9
        else:
            t_s = cfg["time"]
            y = cfg["signal"]

        if cfg["kind"] == "step":
            ax.step(t_s, y, where="post")
        else:
            ax.plot(t_s, y)

        ax.set_ylim(cfg["ylim"])
        ax.set_ylabel(cfg["ylabel"])

    for ax in axes:
        # Remove box around each subplot
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks([0, 1])

    # ADD BLINK INTERVALS
    # Ensure same length for plotting
    m = min(len(time_s), len(refined_blinks1), len(refined_blinks2))

    time_s_plot = time_s.iloc[:m].to_numpy()
    rb1 = refined_blinks1[:m]
    rb2 = refined_blinks2[:m]

    blink1_scaled = (rb1 * 2) - 0.5
    blink2_scaled = (rb2 * 2) - 0.5

    axes[0].step(time_s_plot, blink1_scaled, where="post",
                 color="green", alpha=0.2)
    axes[1].step(time_s_plot, blink2_scaled, where="post",
                 color="green", alpha=0.2)

    axes[-1].set_xlabel("Time (s)")

    plt.subplots_adjust(hspace=0.25, left=0.15)

    output_path = output_folder / "combined_gaze_on_face_interpolated_blinks.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# Graph of multiple streams of eye state information, all stacked together.
# ASSUMES filtered_gaze and filtered_3d_eye_states exist in the timeseries folder.
def combined_x_y_pupils(timeseries_folder):
    df1 = pd.read_csv(timeseries_folder / "filtered_gaze.csv")
    df2 = pd.read_csv(timeseries_folder / "filtered_3d_eye_states.csv")

    time_window = df1["timestamp [ns]"]

    # Convert time to seconds for plotting
    time_s = (time_window - time_window.iloc[0]) * 1e-9

    # Load blink intervals
    blinks_df = pd.read_csv(timeseries_folder / "blinks.csv")

    # Initialize boxcar signal
    signal_blinks = np.zeros(len(time_window), dtype=int)

    # Fill in 1s for all times that blinks are happening
    for _, row in blinks_df.iterrows():
        blink_start = row["start timestamp [ns]"]
        blink_end = row["end timestamp [ns]"]

        active = (time_window >= blink_start) & (time_window <= blink_end)
        signal_blinks[active] = 1

    # PLOTTING
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

    # Define all 3 data plots in terms of names, data streams, and data ranges
    configs = [
        # Gaze x position
        dict(df=df1, time_col="timestamp [ns]", val_col="gaze x [px]",
             transform=lambda s: s.astype(float),
             ylim=(0, 1600), ylabel="Gaze x \nPosition", kind="line"),

        # Gaze y position
        dict(df=df1, time_col="timestamp [ns]", val_col="gaze y [px]",
             transform=lambda s: s.astype(float),
             ylim=(0, 1200), ylabel="Gaze y \nPosition", kind="line"),

        # From DataFrame
        dict(df=df2, time_col="timestamp [ns]", val_col="pupil diameter right [mm]",
             transform=lambda s: s.astype(float),
             ylim=(0, 6), ylabel="Pupil Diameter \n(Right)", kind="line"),
    ]

    # Create and save plot
    for ax, cfg in zip(axes, configs):

        if "df" in cfg:
            t = cfg["df"][cfg["time_col"]]
            y_raw = cfg["df"][cfg["val_col"]]
            y = cfg["transform"](y_raw)
            t_s = (t - t.iloc[0]) * 1e-9
        else:
            t_s = cfg["time"]
            y = cfg["signal"]

        if cfg["kind"] == "step":
            ax.step(t_s, y, where="post")
        else:
            ax.plot(t_s, y)

        ax.set_ylim(cfg["ylim"])
        ax.set_ylabel(cfg["ylabel"])

    for ax in axes:
        # Remove box around each subplot for cosmetic cleanup purposes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[:3]:
        ymin, ymax = ax.get_ylim()
        blink_scaled = signal_blinks * (ymax - ymin) + ymin
        ax.step(time_s, blink_scaled, where="post",
                color="green", alpha=0.7)

    axes[-1].set_xlabel("Time (s)")
    plt.subplots_adjust(hspace=0.25, left=0.15)

    # Make output directory
    output_dir = timeseries_folder / "graphs"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "combined_eye_states.png", dpi=300, bbox_inches="tight")
    plt.close()


def combined_x_y_pupils_interpolate_blinks(timeseries_folder, DERIV_THRESH):
    df1 = pd.read_csv(timeseries_folder / "filtered_gaze.csv")
    df2 = pd.read_csv(timeseries_folder / "filtered_3d_eye_states.csv")

    time_window = df1["timestamp [ns]"]

    # Convert time to seconds for plotting
    time_s = (time_window - time_window.iloc[0]) * 1e-9

    # Load blink intervals
    blinks_df = pd.read_csv(timeseries_folder / "blinks.csv")

    # Initialize boxcar signal
    signal_blinks = np.zeros(len(time_window), dtype=int)

    # Fill in 1s for all times that blinks are happening
    for _, row in blinks_df.iterrows():
        blink_start = row["start timestamp [ns]"]
        blink_end = row["end timestamp [ns]"]

        active = (time_window >= blink_start) & (time_window <= blink_end)
        signal_blinks[active] = 1

    # Take derivative of gaze x for thresholding blinks
    gaze_x = df1["gaze x [px]"]
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

    refined_blinks = np.zeros_like(signal_blinks)

    for s, e in refined_segments:
        refined_blinks[s:e + 1] = 1

    # Interpolate other data streams over the blink segments
    gaze_x_clean = gaze_x.copy()
    gaze_x_clean[refined_blinks == 1] = np.nan

    valid = ~np.isnan(gaze_x_clean)

    spline = UnivariateSpline(
        time_s[valid],
        gaze_x_clean[valid],
        k=1,  # cubic spline
        s=0
    )

    gaze_x_interp = spline(time_s)
    gaze_x_clean[refined_blinks == 1] = gaze_x_interp[refined_blinks == 1]

    gaze_y = df1["gaze y [px]"]
    gaze_y_clean = gaze_y.copy()
    gaze_y_clean[refined_blinks == 1] = np.nan

    valid = ~np.isnan(gaze_y_clean)

    spline = UnivariateSpline(
        time_s[valid],
        gaze_y_clean[valid],
        k=1,  # linear, but turn this number up for quadratic or cubic spline
        s=0
    )

    gaze_y_interp = spline(time_s)
    gaze_y_clean[refined_blinks == 1] = gaze_y_interp[refined_blinks == 1]

    pupil = df2["pupil diameter right [mm]"]
    pupil_clean = pupil.copy()
    pupil_clean[refined_blinks == 1] = np.nan

    valid = ~np.isnan(pupil_clean)

    spline = UnivariateSpline(
        time_s[valid],
        pupil_clean[valid],
        k=1,  # degree of spline
        s=0
    )

    pupil_interp = spline(time_s)
    pupil_clean[refined_blinks == 1] = pupil_interp[refined_blinks == 1]

    # PLOTTING
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

    # Define all 3 data plots in terms of names, data streams, and data ranges
    configs = [
        # Gaze x position
        dict(time=time_s, signal=gaze_x_clean,
             ylim=(0, 1600), ylabel="Gaze x \nPosition", kind="line"),

        # Gaze y position
        dict(time=time_s, signal=gaze_y_clean,
             ylim=(0, 1200), ylabel="Gaze y \nPosition", kind="line"),

        # Pupil Diameter (right)
        dict(time=time_s, signal=pupil_clean,
             ylim=(0, 6), ylabel="Pupil Diameter \n(Right)", kind="line")
    ]

    # Create and save plot
    for ax, cfg in zip(axes, configs):

        if "df" in cfg:
            t = cfg["df"][cfg["time_col"]]
            y_raw = cfg["df"][cfg["val_col"]]
            y = cfg["transform"](y_raw)
            t_s = (t - t.iloc[0]) * 1e-9
        else:
            t_s = cfg["time"]
            y = cfg["signal"]

        if cfg["kind"] == "step":
            ax.step(t_s, y, where="post")
        else:
            ax.plot(t_s, y)

        ax.set_ylim(cfg["ylim"])
        ax.set_ylabel(cfg["ylabel"])

    for ax in axes:
        # Remove box around each subplot for cosmetic cleanup purposes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[:3]:
        ymin, ymax = ax.get_ylim()
        refined_blinks_scaled = refined_blinks * (ymax - ymin) + ymin
        ax.step(time_s, refined_blinks_scaled, where="post", color="green", alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    plt.subplots_adjust(hspace=0.25, left=0.15)

    # Make output directory
    output_dir = timeseries_folder / "graphs"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "combined_eye_states_interpolated_blinks.png", dpi=300, bbox_inches="tight")
    plt.close()


def combined_gyro_x_y_z(timeseries_folder):
    df1 = pd.read_csv(timeseries_folder / "filtered_imu.csv")

    time_window = df1["timestamp [ns]"]

    # Convert time to seconds for plotting
    time_s = (time_window - time_window.iloc[0]) * 1e-9

    # PLOTTING
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

    # Define all 3 data plots in terms of names, data streams, and data ranges
    configs = [
        # Gaze x position
        dict(df=df1, time_col="timestamp [ns]", val_col="gyro x [deg/s]",
             transform=lambda s: s.astype(float),
             ylabel="Gyro x [deg/s]", kind="line"),

        # Gaze y position
        dict(df=df1, time_col="timestamp [ns]", val_col="gyro y [deg/s]",
             transform=lambda s: s.astype(float),
             ylabel="Gyro y [deg/s]", kind="line"),

        # From DataFrame
        dict(df=df1, time_col="timestamp [ns]", val_col="gyro z [deg/s]",
             transform=lambda s: s.astype(float),
             ylabel="Gyro z [deg/s]", kind="line"),
    ]

    # Create and save plot
    for ax, cfg in zip(axes, configs):

        if "df" in cfg:
            t = cfg["df"][cfg["time_col"]]
            y_raw = cfg["df"][cfg["val_col"]]
            y = cfg["transform"](y_raw)
            t_s = (t - t.iloc[0]) * 1e-9
        else:
            t_s = cfg["time"]
            y = cfg["signal"]

        if cfg["kind"] == "step":
            ax.step(t_s, y, where="post")
        else:
            ax.plot(t_s, y)

        # ax.set_ylim(cfg["ylim"])
        ax.set_ylabel(cfg["ylabel"])

    for ax in axes:
        # Remove box around each subplot for cosmetic cleanup purposes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (s)")
    plt.subplots_adjust(hspace=0.25, left=0.15)

    # Make output directory
    output_dir = timeseries_folder / "graphs"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "combined_gyro.png", dpi=300, bbox_inches="tight")
    plt.close()


def combined_roll_pitch_yaw(timeseries_folder):
    df1 = pd.read_csv(timeseries_folder / "filtered_imu.csv")

    time_window = df1["timestamp [ns]"]

    # Convert time to seconds for plotting
    time_s = (time_window - time_window.iloc[0]) * 1e-9

    # PLOTTING
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

    # Define all 3 data plots in terms of names, data streams, and data ranges
    configs = [
        # Gaze x position
        dict(df=df1, time_col="timestamp [ns]", val_col="roll [deg]",
             transform=lambda s: s.astype(float),
             ylabel="Roll [deg]", kind="line"),

        # Gaze y position
        dict(df=df1, time_col="timestamp [ns]", val_col="pitch [deg]",
             transform=lambda s: s.astype(float),
             ylabel="Pitch [deg]", kind="line"),

        # From DataFrame
        dict(df=df1, time_col="timestamp [ns]", val_col="yaw [deg]",
             transform=lambda s: s.astype(float),
             ylabel="Yaw [deg]", kind="line"),
    ]

    # Create and save plot
    for ax, cfg in zip(axes, configs):

        if "df" in cfg:
            t = cfg["df"][cfg["time_col"]]
            y_raw = cfg["df"][cfg["val_col"]]
            y = cfg["transform"](y_raw)
            t_s = (t - t.iloc[0]) * 1e-9
        else:
            t_s = cfg["time"]
            y = cfg["signal"]

        if cfg["kind"] == "step":
            ax.step(t_s, y, where="post")
        else:
            ax.plot(t_s, y)

        # ax.set_ylim(cfg["ylim"])
        ax.set_ylabel(cfg["ylabel"])

    for ax in axes:
        # Remove box around each subplot for cosmetic cleanup purposes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (s)")
    plt.subplots_adjust(hspace=0.25, left=0.15)

    # Make output directory
    output_dir = timeseries_folder / "graphs"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "combined_roll_pitch_yaw.png", dpi=300, bbox_inches="tight")
    plt.close()


def combined_imu_acceleration(timeseries_folder):
    df1 = pd.read_csv(timeseries_folder / "filtered_imu.csv")

    time_window = df1["timestamp [ns]"]

    # Convert time to seconds for plotting
    time_s = (time_window - time_window.iloc[0]) * 1e-9

    # PLOTTING
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

    # Define all 3 data plots in terms of names, data streams, and data ranges
    configs = [
        # Gaze x position
        dict(df=df1, time_col="timestamp [ns]", val_col="acceleration x [g]",
             transform=lambda s: s.astype(float),
             ylabel="Acceleration x [g]", kind="line"),

        # Gaze y position
        dict(df=df1, time_col="timestamp [ns]", val_col="acceleration y [g]",
             transform=lambda s: s.astype(float),
             ylabel="Acceleration y [g]", kind="line"),

        # From DataFrame
        dict(df=df1, time_col="timestamp [ns]", val_col="acceleration z [g]",
             transform=lambda s: s.astype(float),
             ylabel="Acceleration z [g]", kind="line"),
    ]

    # Create and save plot
    for ax, cfg in zip(axes, configs):

        if "df" in cfg:
            t = cfg["df"][cfg["time_col"]]
            y_raw = cfg["df"][cfg["val_col"]]
            y = cfg["transform"](y_raw)
            t_s = (t - t.iloc[0]) * 1e-9
        else:
            t_s = cfg["time"]
            y = cfg["signal"]

        if cfg["kind"] == "step":
            ax.step(t_s, y, where="post")
        else:
            ax.plot(t_s, y)

        # ax.set_ylim(cfg["ylim"])
        ax.set_ylabel(cfg["ylabel"])

    for ax in axes:
        # Remove box around each subplot for cosmetic cleanup purposes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (s)")
    plt.subplots_adjust(hspace=0.25, left=0.15)

    # Make output directory
    output_dir = timeseries_folder / "graphs"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "combined_imu_acceleration.png", dpi=300, bbox_inches="tight")
    plt.close()
