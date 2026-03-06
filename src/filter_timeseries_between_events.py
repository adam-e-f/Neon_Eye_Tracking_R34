import pandas as pd


'''
Function get_filtered_series() for returning a timeseries csv containing only the rows with timestamps between
the start_time and end_time defined in the function's arguments. We parameterize:
    a) The folder containing the original timeseries (we use this as the destination folder too)
    b) The filename of the original timeseries (Should include the '.csv' at the end)
    c) The start time
    d) The end time
'''


def get_filtered_series(timeseries_folder, timeseries_filename, start_time, end_time, ):
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
