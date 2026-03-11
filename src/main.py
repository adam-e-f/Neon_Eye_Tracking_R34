import os

import pandas as pd
from pathlib import Path
import filter_timeseries_between_events
import make_timeseries_graph
from make_timeseries_graph import GraphObj


'''
This function takes as input the path to the folder containing all the various timeseries and a start time and end
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
Merge mutual gaze with mutual gaze interpolated blinks csv. Assumes both csv files exist.
'''


def merge_mutual_gaze_csv(project_folder):
    df1 = pd.read_csv(project_folder / "mutual_gaze.csv")
    df2 = pd.read_csv(project_folder / "mutual_gaze_interpolated_blinks.csv")

    # Extract columns
    t1 = df1["timestamp [ns]"]
    s1 = df1["gaze on face"]
    s2 = df2["gaze on face"]

    s1_num = s1.astype(int).to_numpy()
    s2_num = s2.astype(int).to_numpy()

    # Build output DataFrame
    out_df = pd.DataFrame({
        "timestamp [ns]": t1,
        "gaze on face": s2_num,
        "gaze on face (raw)": s1_num
    })

    os.remove(project_folder / "mutual_gaze.csv")
    os.remove(project_folder / "mutual_gaze_interpolated_blinks.csv")

    output_path = project_folder / "mutual_gaze.csv"
    out_df.to_csv(output_path, index=False)


# events_file_path1 = Path(r"C:\Users\adamf\OneDrive\Desktop\MAPLab\r34project\Test3\glasses1_timeseries\export\events.csv")
events_file_path2 = Path(r"C:\Users\adamf\OneDrive\Desktop\MAPLab\r34project\Test3\glasses2_timeseries\2026-02-11_00-23-44-b5e2cb4f\events.csv")

start_event = "clip_start"
end_event = "clip_end"

events_df2 = pd.read_csv(events_file_path2)

start_time = events_df2.loc[events_df2["name"] == start_event, "timestamp [ns]"].iloc[0]
end_time = events_df2.loc[events_df2["name"] == end_event, "timestamp [ns]"].iloc[0]

# Process data from glasses1 and make corresponding graphs
timeseries_folder_glasses1 = Path(
    r"C:\Users\adamf\OneDrive\Desktop\MAPLab\r34project\Test4\glasses1_timeseries\2026-02-11_00-23-47-47b67625"
)
process_timeseries_and_make_graphs(timeseries_folder_glasses1, start_time, end_time)
filter_timeseries_between_events.combine(timeseries_folder_glasses1, 500)

# Process data from glasses2 and make corresponding graphs
timeseries_folder_glasses2 = Path(
    r"C:\Users\adamf\OneDrive\Desktop\MAPLab\r34project\Test4\glasses2_timeseries\2026-02-11_00-23-44-b5e2cb4f"
)
process_timeseries_and_make_graphs(timeseries_folder_glasses2, start_time, end_time)
filter_timeseries_between_events.combine(timeseries_folder_glasses2, 500)

# Make csv and graph for when both gazes are on faces (both wearers are looking at each other)

gaze_on_face_glasses1 = timeseries_folder_glasses1 / "filtered_gaze_on_face.csv"
gaze_on_face_glasses2 = timeseries_folder_glasses2 / "filtered_gaze_on_face.csv"
main_folder = Path(r"C:\Users\adamf\OneDrive\Desktop\MAPLab\r34project\Test4")

# both_gazes_on_faces(gaze_on_face_glasses1, gaze_on_face_glasses2, main_folder)

make_timeseries_graph.combined_gazes_on_faces(timeseries_folder_glasses1, timeseries_folder_glasses2, main_folder)

make_timeseries_graph.combined_gazes_on_faces_interpolate_blinks(
    timeseries_folder_glasses1, timeseries_folder_glasses2, main_folder, 500
)

merge_mutual_gaze_csv(main_folder)

# GRAPHING - G1 = glasses1, G2 = glasses2
combined_1_path = timeseries_folder_glasses1 / "combined_timeseries.csv"
combined_2_path = timeseries_folder_glasses2 / "combined_timeseries.csv"
imu_1_path = timeseries_folder_glasses1 / "filtered_imu.csv"
imu_2_path = timeseries_folder_glasses2 / "filtered_imu.csv"

gaze_x_G1 = GraphObj(axis_title="Gaze x\nPosition", is_step=False, show_blinks=True,
                     file_path=combined_1_path, column_name="gaze x [px]", range=(0, 1600))
gaze_x_G2 = GraphObj(axis_title="Gaze x\nPosition", is_step=False, show_blinks=True,
                     file_path=combined_2_path, column_name="gaze x [px]", range=(0, 1600))
gaze_y_G1 = GraphObj(axis_title="Gaze y\nPosition", is_step=False, show_blinks=True,
                     file_path=combined_1_path, column_name="gaze y [px]", range=(0, 1200))
gaze_y_G2 = GraphObj(axis_title="Gaze y\nPosition", is_step=False, show_blinks=True,
                     file_path=combined_2_path, column_name="gaze y [px]", range=(0, 1200))

pupil_right_G1 = GraphObj(axis_title="Pupil Diameter\nRight [mm]", is_step=False, show_blinks=True,
                          file_path=combined_1_path, column_name="pupil diameter right [mm]", range=(0, 6))
pupil_right_G2 = GraphObj(axis_title="Pupil Diameter\nRight [mm]", is_step=False, show_blinks=True,
                          file_path=combined_2_path, column_name="pupil diameter right [mm]", range=(0, 6))
pupil_left_G1 = GraphObj(axis_title="Pupil Diameter\nLeft [mm]", is_step=False, show_blinks=True,
                         file_path=combined_1_path, column_name="pupil diameter left [mm]", range=(0, 6))
pupil_left_G2 = GraphObj(axis_title="Pupil Diameter\nLeft [mm]", is_step=False, show_blinks=True,
                         file_path=combined_2_path, column_name="pupil diameter left [mm]", range=(0, 6))

gaze_on_face_G1 = GraphObj(axis_title="Subject 1's\nGaze on Face", is_step=True, show_blinks=True,
                           file_path=combined_1_path, column_name="gaze on face", range=(-0.5, 1.5))
gaze_on_face_G2 = GraphObj(axis_title="Subject 2's\nGaze on Face", is_step=True, show_blinks=True,
                           file_path=combined_2_path, column_name="gaze on face", range=(-0.5, 1.5))

roll_G1 = GraphObj(axis_title="Roll [deg]", is_step=False, show_blinks=False,
                   file_path=imu_1_path, column_name="roll [deg]", range=(-1, -1))
roll_G2 = GraphObj(axis_title="Roll [deg]", is_step=False, show_blinks=False,
                   file_path=imu_2_path, column_name="roll [deg]", range=(-1, -1))
pitch_G1 = GraphObj(axis_title="Pitch [deg]", is_step=False, show_blinks=False,
                    file_path=imu_1_path, column_name="pitch [deg]", range=(-1, -1))
pitch_G2 = GraphObj(axis_title="Pitch [deg]", is_step=False, show_blinks=False,
                    file_path=imu_2_path, column_name="pitch [deg]", range=(-1, -1))
yaw_G1 = GraphObj(axis_title="Yaw [deg]", is_step=False, show_blinks=False,
                  file_path=imu_1_path, column_name="yaw [deg]", range=(-1, -1))
yaw_G2 = GraphObj(axis_title="Yaw [deg]", is_step=False, show_blinks=False,
                  file_path=imu_2_path, column_name="yaw [deg]", range=(-1, -1))

gyro_x_G1 = GraphObj(axis_title="Gyro x\n[deg/s]", is_step=False, show_blinks=False,
                     file_path=imu_1_path, column_name="gyro x [deg/s]", range=(-1, -1))
gyro_x_G2 = GraphObj(axis_title="Gyro x\n[deg/s]", is_step=False, show_blinks=False,
                     file_path=imu_2_path, column_name="gyro x [deg/s]", range=(-1, -1))
gyro_y_G1 = GraphObj(axis_title="Gyro y\n[deg/s]", is_step=False, show_blinks=False,
                     file_path=imu_1_path, column_name="gyro y [deg/s]", range=(-1, -1))
gyro_y_G2 = GraphObj(axis_title="Gyro y\n[deg/s]", is_step=False, show_blinks=False,
                     file_path=imu_2_path, column_name="gyro y [deg/s]", range=(-1, -1))
gyro_z_G1 = GraphObj(axis_title="Gyro z\n[deg/s]", is_step=False, show_blinks=False,
                     file_path=imu_1_path, column_name="gyro z [deg/s]", range=(-1, -1))
gyro_z_G2 = GraphObj(axis_title="Gyro z\n[deg/s]", is_step=False, show_blinks=False,
                     file_path=imu_2_path, column_name="gyro z [deg/s]", range=(-1, -1))

make_timeseries_graph.make_graphs(
    graphs_list=[gaze_x_G1, gaze_y_G1, pupil_right_G1],
    save_folder=timeseries_folder_glasses1/"graphs",
    output_filename="combined_eye_states.png"
)
make_timeseries_graph.make_graphs(
    graphs_list=[gaze_x_G2, gaze_y_G2, pupil_right_G2],
    save_folder=timeseries_folder_glasses2/"graphs",
    output_filename="combined_eye_states.png"
)

make_timeseries_graph.make_graphs(
    graphs_list=[roll_G1, pitch_G1, yaw_G1],
    save_folder=timeseries_folder_glasses1/"graphs",
    output_filename="roll_pitch_yaw.png"
)
make_timeseries_graph.make_graphs(
    graphs_list=[roll_G2, pitch_G2, yaw_G2],
    save_folder=timeseries_folder_glasses2/"graphs",
    output_filename="roll_pitch_yaw.png"
)

make_timeseries_graph.make_graphs(
    graphs_list=[gyro_x_G1, gyro_y_G1, gyro_z_G1],
    save_folder=timeseries_folder_glasses1/"graphs",
    output_filename="gyro_xyz.png"
)
make_timeseries_graph.make_graphs(
    graphs_list=[gyro_x_G2, gyro_y_G2, gyro_z_G2],
    save_folder=timeseries_folder_glasses2/"graphs",
    output_filename="gyro_xyz.png"
)



''' OLD GRAPHING PROCEDURE
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
'''