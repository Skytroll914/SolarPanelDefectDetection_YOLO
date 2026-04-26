############################
# Coded by Richard, Ting, Li, Zeng
############################

import gradio as gr
from ultralytics import YOLO
import numpy as np
import cv2
import tempfile
import os

# ===============================
# Load trained model
# ===============================
# Make sure best.pt is in the same folder as this script
model = YOLO("best.pt")

# ===============================
# Inference functions
# ===============================

def detect_image(image, confidence):
    """Run detection on a single uploaded image."""
    if image is None:
        return None

    results = model.predict(source=image, conf=confidence, iou=0.5, imgsz=640)
    annotated = results[0].plot()
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def detect_video(video_path, confidence):
    """Run detection on every frame of an uploaded video and return annotated video."""
    if video_path is None:
        return None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Write output to a temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=confidence, iou=0.5, imgsz=640, verbose=False)
        annotated = results[0].plot()
        writer.write(annotated)

    cap.release()
    writer.release()
    return out_path


def detect_webcam(frame, confidence):
    """Run detection on a single webcam frame (called per streaming frame)."""
    if frame is None:
        return None

    results = model.predict(source=frame, conf=confidence, iou=0.5, imgsz=640, verbose=False)
    annotated = results[0].plot()
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# ===============================
# Gradio UI
# ===============================
with gr.Blocks(title="Solar Panel Defect Detection") as demo:
    gr.Markdown(
        """
        # 🔆 Solar Panel Defect Detection
        Detect defects on solar panels using a fine-tuned **YOLO26-nano** model.
        Choose an input mode below.
        """
    )

    # Shared confidence slider above the tabs
    confidence = gr.Slider(
        minimum=0.1, maximum=0.9, value=0.25, step=0.05,
        label="Confidence Threshold"
    )

    with gr.Tabs():

        # ── Tab 1: Image ──────────────────────────────────────
        with gr.Tab("📷 Image"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="numpy", label="Upload Image")
                    img_btn = gr.Button("Detect Defects", variant="primary")
                with gr.Column():
                    img_output = gr.Image(type="numpy", label="Detection Result")

            img_btn.click(fn=detect_image, inputs=[img_input, confidence], outputs=img_output)

        # ── Tab 2: Video ──────────────────────────────────────
        with gr.Tab("🎬 Video"):
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Upload Video")
                    vid_btn = gr.Button("Detect Defects", variant="primary")
                with gr.Column():
                    vid_output = gr.Video(label="Annotated Video")

            vid_btn.click(fn=detect_video, inputs=[vid_input, confidence], outputs=vid_output)

        # ── Tab 3: Webcam ─────────────────────────────────────
        with gr.Tab("📹 Webcam"):
            gr.Markdown("Allow browser camera access, then detections will appear live on the right.")
            with gr.Row():
                with gr.Column():
                    cam_input = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam Feed")
                with gr.Column():
                    cam_output = gr.Image(type="numpy", label="Live Detection")

            cam_input.stream(fn=detect_webcam, inputs=[cam_input, confidence], outputs=cam_output)

    gr.Markdown("*Model: YOLO26-nano fine-tuned on solar panel defect dataset*")

# ===============================
# Launch
# ===============================
if __name__ == "__main__":
    demo.launch()
