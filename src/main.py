import pandas as pd
from pathlib import Path
import filter_timeseries_between_events
import make_timeseries_graph


'''
This function takes as input the path to the folder containing all the various timeseries' and a start time and end
time, i.e., the events between which we want to look. It filters some of the timeseries data, creating new .csv files
in the timeseries_folder, and creates all of the graphs between those two events.

IMPORTANT: For this code to run, all timeseries data + all face mapping data should be present in the same folder.
'''


def process_timeseries_and_make_graphs(timeseries_folder, start_time, end_time):

    # Filter gaze csv and generate timeseries plots for x and y positions of gaze

    filtered_gaze = filter_timeseries_between_events.get_filtered_series(
        timeseries_folder, "gaze.csv", start_time, end_time
    )

    make_timeseries_graph.graph(
        timeseries_folder, filtered_gaze, "Gaze X over Time", "Gaze X position (px)",
        "gaze x [px]", 0, 1600
    )

    make_timeseries_graph.graph(
        timeseries_folder, filtered_gaze, "Gaze Y over Time", "Gaze Y position (px)",
        "gaze y [px]", 0, 1200
    )

    # Next we do the same but for pupil diameter, for left eye and right eye.

    filtered_eye_states = filter_timeseries_between_events.get_filtered_series(
        timeseries_folder, "3d_eye_states.csv", start_time, end_time
    )

    make_timeseries_graph.graph(
        timeseries_folder, filtered_eye_states, "Pupil Diameter (left) over Time",
        "Pupil Diameter, Left Eye (mm)", "pupil diameter left [mm]"
    )

    make_timeseries_graph.graph(
        timeseries_folder, filtered_eye_states, "Pupil Diameter (right) over Time",
        "Pupil Diameter, Right Eye (mm)", "pupil diameter right [mm]"
    )

    # Next we generate a boxcar graph of the saccades.

    make_timeseries_graph.graph_saccades(timeseries_folder)

    # And we do the same but for blinks.

    make_timeseries_graph.graph_blinks(timeseries_folder)

    # Next we generate a boxcar graph showing when the gaze is on a face.

    filter_timeseries_between_events.get_filtered_series(
        timeseries_folder, "gaze_on_face.csv", start_time, end_time
    )

    make_timeseries_graph.graph_gaze_on_face(timeseries_folder, "Gaze on Face")

    # Filter imu csv
    filter_timeseries_between_events.get_filtered_series(
        timeseries_folder, "imu.csv", start_time, end_time
    )

    return


'''
Multiply two gaze_on_face timeseries together in order to obtain a csv and graph for when both gazes are on faces.
'''


def both_gazes_on_faces(file1_path, file2_path, output_folder):
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

    output_path = output_folder / "both_gazes_on_face.csv"
    out_df.to_csv(output_path, index=False)

    make_timeseries_graph.graph_gaze_on_face(
        output_folder, "Gaze on Face", timeseries_filename="both_gazes_on_face.csv"
    )


def select_events(events_path_1, events_path_2):


    return


# events_file_path1 = Path(r"C:\Users\adamf\OneDrive\Desktop\MAPLab\r34project\Test3\glasses1_timeseries\export\events.csv")
events_file_path2 = Path(r"C:\Users\adamf\OneDrive\Desktop\MAPLab\r34project\Test3\glasses2_timeseries\2026-02-11_00-23-44-b5e2cb4f\events.csv")

''' USE THE FOLLOWING TO DO PROCESSING OVER THE ENTIRE RECORDINGS, AS OPPOSED TO A SUBCLIP

start_event = "recording.begin"
end_event = "recording.end"

# Load CSV
events_df1 = pd.read_csv(events_file_path1)
events_df2 = pd.read_csv(events_file_path2)

# Get timestamps for start and end events
start_time1 = events_df1.loc[events_df1["name"] == start_event, "timestamp [ns]"].iloc[0]
end_time1 = events_df1.loc[events_df1["name"] == end_event, "timestamp [ns]"].iloc[0]
start_time2 = events_df2.loc[events_df2["name"] == start_event, "timestamp [ns]"].iloc[0]
end_time2 = events_df2.loc[events_df2["name"] == end_event, "timestamp [ns]"].iloc[0]

# Window between start_time and end_time should be the range in which both cameras were on and recording.
start_time = max(start_time1, start_time2)
end_time = min(end_time1, end_time2)

'''

start_event = "clip_start"
end_event = "clip_end"

events_df2 = pd.read_csv(events_file_path2)

start_time = events_df2.loc[events_df2["name"] == start_event, "timestamp [ns]"].iloc[0]
end_time = events_df2.loc[events_df2["name"] == end_event, "timestamp [ns]"].iloc[0]

# Process data from glasses1 and make corresponding graphs
timeseries_folder_glasses1 = Path(
    r"C:\Users\adamf\OneDrive\Desktop\MAPLab\r34project\Test3\glasses1_timeseries\2026-02-11_00-23-47-47b67625"
)
process_timeseries_and_make_graphs(timeseries_folder_glasses1, start_time, end_time)

# Process data from glasses2 and make corresponding graphs
timeseries_folder_glasses2 = Path(
    r"C:\Users\adamf\OneDrive\Desktop\MAPLab\r34project\Test3\glasses2_timeseries\2026-02-11_00-23-44-b5e2cb4f"
)
process_timeseries_and_make_graphs(timeseries_folder_glasses2, start_time, end_time)

# Make csv and graph for when both gazes are on faces (both wearers are looking at each other)

gaze_on_face_glasses1 = timeseries_folder_glasses1 / "filtered_gaze_on_face.csv"
gaze_on_face_glasses2 = timeseries_folder_glasses2 / "filtered_gaze_on_face.csv"
main_folder = Path(r"C:\Users\adamf\OneDrive\Desktop\MAPLab\r34project\Test3")

both_gazes_on_faces(gaze_on_face_glasses1, gaze_on_face_glasses2, main_folder)

make_timeseries_graph.combined_gazes_on_faces(timeseries_folder_glasses1, timeseries_folder_glasses2, main_folder)

make_timeseries_graph.combined_x_y_pupils(timeseries_folder_glasses1)
make_timeseries_graph.combined_x_y_pupils(timeseries_folder_glasses2)

make_timeseries_graph.combined_x_y_pupils_interpolate_blinks(timeseries_folder_glasses1, 500)
make_timeseries_graph.combined_x_y_pupils_interpolate_blinks(timeseries_folder_glasses2, 500)

make_timeseries_graph.combined_gazes_on_faces_interpolate_blinks(
    timeseries_folder_glasses1, timeseries_folder_glasses2, main_folder, 500
)

make_timeseries_graph.combined_gyro_x_y_z(timeseries_folder_glasses1)
make_timeseries_graph.combined_gyro_x_y_z(timeseries_folder_glasses2)

make_timeseries_graph.combined_roll_pitch_yaw(timeseries_folder_glasses1)
make_timeseries_graph.combined_roll_pitch_yaw(timeseries_folder_glasses2)

make_timeseries_graph.combined_imu_acceleration(timeseries_folder_glasses1)
make_timeseries_graph.combined_imu_acceleration(timeseries_folder_glasses2)
