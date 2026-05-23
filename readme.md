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


---
# File Structure Planned

AI-Driver-Drowsiness-System/
│
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── config.py
├── setup.py
│
├── data/
│   │
│   ├── raw/
│   │   ├── images/
│   │   ├── videos/
│   │   ├── audio/
│   │   └── telemetry/
│   │
│   ├── processed/
│   │   ├── images/
│   │   ├── videos/
│   │   ├── audio/
│   │   └── telemetry/
│   │
│   ├── annotations/
│   │   ├── image_labels/
│   │   ├── video_labels/
│   │   └── audio_labels/
│   │
│   └── reports/
│
├── notebooks/
│   ├── image_model_training.ipynb
│   ├── video_model_training.ipynb
│   ├── audio_model_training.ipynb
│   └── telemetry_analysis.ipynb
│
├── models/
│   │
│   ├── image_models/
│   │   ├── eye_detection/
│   │   ├── yawn_detection/
│   │   └── face_landmark/
│   │
│   ├── video_models/
│   │   ├── head_pose/
│   │   └── behavior_tracking/
│   │
│   ├── audio_models/
│   │   ├── yawn_audio/
│   │   └── voice_stress/
│   │
│   ├── telemetry_models/
│   │   └── driving_behavior/
│   │
│   └── trained/
│       ├── cnn_model.pkl
│       ├── svm_model.pkl
│       ├── random_forest.pkl
│       └── lstm_model.h5
│
├── src/
│   │
│   ├── image_processing/
│   │   ├── eye_detection.py
│   │   ├── yawn_detection.py
│   │   ├── face_detection.py
│   │   └── fatigue_score.py
│   │
│   ├── video_processing/
│   │   ├── frame_extraction.py
│   │   ├── head_movement.py
│   │   ├── driver_monitor.py
│   │   └── behavior_analysis.py
│   │
│   ├── audio_processing/
│   │   ├── audio_preprocessing.py
│   │   ├── yawn_audio_detection.py
│   │   ├── voice_stress_analysis.py
│   │   └── feature_extraction.py
│   │
│   ├── telemetry_processing/
│   │   ├── speed_analysis.py
│   │   ├── steering_analysis.py
│   │   ├── braking_analysis.py
│   │   └── trip_risk_analysis.py
│   │
│   ├── machine_learning/
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── evaluate.py
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   │
│   ├── alerts/
│   │   ├── alarm_system.py
│   │   ├── email_alert.py
│   │   └── report_generator.py
│   │
│   ├── utils/
│   │   ├── logger.py
│   │   ├── helper.py
│   │   ├── visualization.py
│   │   └── constants.py
│   │
│   └── app/
│       ├── dashboard.py
│       ├── live_monitor.py
│       └── gui.py
│
├── tests/
│   ├── test_image.py
│   ├── test_video.py
│   ├── test_audio.py
│   └── test_ml.py
│
├── outputs/
│   ├── predictions/
│   ├── logs/
│   ├── graphs/
│   └── safety_reports/
│
└── docs/
    ├── architecture.png
    ├── project_report.docx
    └── presentation.pptx


---


