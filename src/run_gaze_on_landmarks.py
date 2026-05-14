from pathlib import Path
import sys
import json
sys.path.append(str(Path(__file__).parent / "gaze-on-facial-landmarks" / "src"))
sys.path.append(str(Path(__file__).parent / "dynamic-rim-module" / "src"))

from pupil_labs.gaze_on_facial_landmarks.__main__ import run_all

# print(sys.executable)
input_data = json.loads(sys.argv[1])
run_all(input_data)

