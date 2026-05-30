---

# AI-Based Driver Drowsiness & Vehicle Safety Monitoring System

To Develop an AI-Based Driver Drowsiness & Vehicle Safety Monitoring System using Machine Learning and Artificial Intelligence technologies.
The system uses image and video processing to detect driver fatigue symptoms such as eye closure, yawning, and head movement during driving.
It also includes Audio-Based Fatigue Detection where yawning sounds and voice stress patterns are analyzed using audio processing techniques.
Vehicle telemetry and sensor-based text data such as speed, braking patterns, steering angle, and trip duration are analyzed to identify unsafe driving behavior and accident risks.
The system automatically generates safety alerts, fatigue warnings, and trip safety reports to improve driver safety and reduce road accidents.


----
# Data Types Used :-

Image  -  Eye detection, face monitoring, and fatigue identification
Video  -  Driver behavior analysis, yawning detection, and head movement tracking
Audio  -  Yawning sound detection and voice stress analysis
Text  -  Vehicle speed data, braking logs, steering data, trip reports, and safety alerts


--

## Dataset Prep

This dataset is large enough that a first training run should use a reduced, balanced subset.

Recommended starting point:
- Keep about 1000 images per source group.
- Use an 80/10/10 train/val/test split.
- Keep the image and `.txt` label files together.

Prep command:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_nthu_dataset.py --max-per-group 1000
```

With the current four source groups, this gives you 4000 total images.

The script writes a YOLO-style folder layout under `prepared_nthu_dataset/` with matching `images/` and `labels/` splits, plus a selection summary file.

If your source images already contain colored annotation boxes, clean them first:

```powershell
.\.venv\Scripts\python.exe scripts\remove_nthu_box_overlays.py --source-root NTHU_Dataset --output-root cleaned_nthu_dataset
```

Then point the prep script at `cleaned_nthu_dataset/` instead of `NTHU_Dataset/`.
