#!/usr/bin/env python3

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


DEFAULT_MODEL_PATH = "models/best.pt"
OUTPUT_DIR = Path("ui_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def default_device_value() -> str:
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def run_predict(
    model: YOLO,
    image_bgr: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> np.ndarray:
    results = model.predict(
        image_bgr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    return results[0].plot()


def save_image(image_bgr: np.ndarray, prefix: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = OUTPUT_DIR / f"{prefix}_{timestamp}.jpg"
    cv2.imwrite(str(out_path), image_bgr)
    return out_path


def process_video(
    input_path: Path,
    output_path: Path,
    model: YOLO,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError("Unable to open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = run_predict(model, frame, conf, iou, imgsz, device)
        writer.write(annotated)

        processed += 1
        if total_frames > 0:
            progress.progress(min(processed / total_frames, 1.0))

    cap.release()
    writer.release()
    progress.empty()


def show_webcam(
    model: YOLO,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> None:
    try:
        from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
        import av
    except Exception:
        st.warning(
            "Live webcam needs streamlit-webrtc and av. Install with: "
            "pip install streamlit-webrtc av"
        )
        st.info("Fallback: use the Snapshot tab below.")
        return

    class YOLOVideoProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            self._model = model
            self._conf = conf
            self._iou = iou
            self._imgsz = imgsz
            self._device = device

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            annotated = run_predict(self._model, img, self._conf, self._iou, self._imgsz, self._device)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="yolo-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )


st.set_page_config(page_title="Driver Drowsiness YOLO26", layout="wide")

st.title("Driver Drowsiness YOLO26 - Local UI")

with st.sidebar:
    st.header("Model Settings")
    model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    device = st.text_input("Device", value=default_device_value())
    conf = st.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("IoU", 0.1, 0.9, 0.45, 0.05)
    imgsz = st.selectbox("Image size", [320, 416, 512, 640], index=3)
    save_outputs = st.checkbox("Save outputs", value=True)

model: Optional[YOLO] = None

if model_path:
    if not Path(model_path).exists():
        st.error(f"Model not found: {model_path}")
    else:
        model = load_model(model_path)

if model is None:
    st.stop()

image_tab, video_tab, webcam_tab, snapshot_tab = st.tabs(
    ["Image", "Video", "Live Webcam", "Snapshot"]
)

with image_tab:
    st.subheader("Image inference")
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated = run_predict(model, image_bgr, conf, iou, imgsz, device)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Predictions")
        if save_outputs:
            out_path = save_image(annotated, "image")
            st.success(f"Saved: {out_path}")

with video_tab:
    st.subheader("Video inference")
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.name).suffix) as tmp:
            tmp.write(video_file.read())
            tmp_path = Path(tmp.name)

        out_path = OUTPUT_DIR / f"video_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
        st.info("Processing video. This can take a while.")
        process_video(tmp_path, out_path, model, conf, iou, imgsz, device)
        st.video(str(out_path))
        if save_outputs:
            st.success(f"Saved: {out_path}")
        os.unlink(tmp_path)

with webcam_tab:
    st.subheader("Live webcam")
    show_webcam(model, conf, iou, imgsz, device)

with snapshot_tab:
    st.subheader("Snapshot (single frame)")
    snapshot = st.camera_input("Take a picture")
    if snapshot is not None:
        file_bytes = np.asarray(bytearray(snapshot.getvalue()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated = run_predict(model, image_bgr, conf, iou, imgsz, device)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Predictions")
        if save_outputs:
            out_path = save_image(annotated, "snapshot")
            st.success(f"Saved: {out_path}")
