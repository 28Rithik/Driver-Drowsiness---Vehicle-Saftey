from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import cv2
import os
import time

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

models = {
    "best": YOLO("models/best.pt"),
    "last": YOLO("models/last.pt")
}

selected_model = models["best"]


def predict_frame(frame):
    results = selected_model.predict(
        frame,
        conf=0.25,
        verbose=False
    )

    return results[0].plot()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/select_model", methods=["POST"])
def select_model():
    global selected_model

    model_name = request.form.get("model")

    if model_name in models:
        selected_model = models[model_name]

    return render_template("index.html")


@app.route("/upload_image", methods=["POST"])
def upload_image():

    file = request.files["image"]

    filename = f"{int(time.time())}_{file.filename}"

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    result_path = os.path.join(
        RESULT_FOLDER,
        filename
    )

    file.save(upload_path)

    image = cv2.imread(upload_path)

    annotated = predict_frame(image)

    cv2.imwrite(result_path, annotated)

    return render_template(
        "index.html",
        result_image=result_path
    )


@app.route("/upload_video", methods=["POST"])
def upload_video():

    file = request.files["video"]

    filename = f"{int(time.time())}_{file.filename}"

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    output_path = os.path.join(
        RESULT_FOLDER,
        "processed_" + filename
    )

    file.save(upload_path)

    cap = cv2.VideoCapture(upload_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30,
        (width, height)
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        annotated = predict_frame(frame)

        writer.write(annotated)

    cap.release()
    writer.release()

    return render_template(
        "index.html",
        result_video=output_path
    )


camera = cv2.VideoCapture(0)


def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        annotated = predict_frame(frame)

        ret, buffer = cv2.imencode(
            ".jpg",
            annotated
        )

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )