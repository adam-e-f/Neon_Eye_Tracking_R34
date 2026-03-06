import shutil
import requests
import zipfile
import os
from pathlib import Path
from pupilcloud import Api, ApiException
import numpy as np
from numpy.lib import recfunctions as rfn
import subprocess


def download_recording_files(api, recording_id, outdir):
    try:
        response = api.download_recording_zip(recording_id, _preload_content=False)
        # print("Data: %r" % response.read())
    except ApiException as e:
        print("Exception when calling RecordingsApi->download_recording_zip: %s\n" % e)

    zip_path = outdir / f"ts.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    # Write ZIP to disk
    with open(zip_path, "wb") as f:
        while True:
            chunk = response.read(8192)
            if not chunk:
                break
            f.write(chunk)

    # Extract
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(outdir)

    print("Raw data downloaded from cloud. Processing...")

    zip_path.unlink()

    subdir = next(outdir.iterdir())
    recording_dir = outdir / subdir

    # Convert via pl-rec-export
    subprocess.run([
        "pl-rec-export",
        str(recording_dir)
    ], check=True)

    shutil.move(recording_dir / "export", outdir / "export")



API_TOKEN1 = "ChtFqP2ByQZ4b8mpDf9WSW2jTwov7PePQD9SBXKTHxrZ"
API_TOKEN2 = "3jmzyt5AS6nQ99eBAheF6v9apTWxM4qJAovgizfcwk7N"
workspace_ID1 = "e6873f00-ac5a-40f1-909f-2e8a28ab71c2"
workspace_ID2 = "68a49c94-19c5-4d30-a0ca-9e053b7647c8"

HEADERS = {"Authorization": f"Bearer {API_TOKEN2}"}


recording_id_1 = "47b67625-bff6-41aa-899c-ef923341760f"
recording_id_2 = "b5e2cb4f-1ddd-40cd-a480-93fc98e45009"
destination_folder = Path(r"C:\Users\adamf\OneDrive\Desktop\MAPLab\r34project\Test3")

glasses2_ts_folder = destination_folder / "glasses2_timeseries"
glasses2_ts_folder.mkdir(parents=True, exist_ok=True)


# download_timeseries(recording_id_1, glasses1_ts_folder)

# Initialize the API with your API token
api_token = API_TOKEN2
api = Api(api_key=api_token, host=f"https://api.cloud.pupil-labs.com/v2/workspaces/{workspace_ID2}")

download_recording_files(api, recording_id_2, glasses2_ts_folder)
