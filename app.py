"""Streamlit dashboard for recorded and live pose-based gait recognition."""
from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path
from time import perf_counter

import numpy as np
import streamlit as st

st.set_page_config(page_title="Gait Recognition", page_icon="🚶", layout="wide")
st.title("🚶 Pose-Based Gait Recognition")
st.caption("Research demonstration only. Gait is biometric data: obtain consent and do not use this for surveillance or safety-critical decisions.")

MODEL_PATH = Path("models_artifacts/gait_model.h5")


@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        return None, []
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH), json.loads(Path("models_artifacts/class_names.json").read_text())


model, names = load_assets()


def classify(sequence: np.ndarray) -> tuple[str, float]:
    if model is None:
        return "Model unavailable", 0.0
    scores = model.predict(sequence[None], verbose=0)[0]
    index = int(scores.argmax())
    return names[index], float(scores[index])


def recorded_video() -> None:
    st.subheader("Recorded video")
    upload = st.file_uploader("Upload a clear, single-person walking video", type=["mp4", "avi", "mov"])
    if not upload:
        return
    st.video(upload)
    if not st.button("Analyze gait", type="primary"):
        return
    if model is None:
        st.error("No trained model found. Generate data and train the model first.")
        return
    suffix = Path(upload.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(upload.getbuffer()); temp_path = tmp.name
    try:
        from preprocessing.video_processor import video_to_sequence
        with st.spinner("Extracting pose landmarks and running the model…"):
            sequence, preview = video_to_sequence(temp_path, model.input_shape[1])
            scores = model.predict(sequence[None], verbose=0)[0]
        index = int(scores.argmax())
        a, b = st.columns(2)
        a.metric("Predicted class", names[index]); b.metric("Model score", f"{scores[index]:.1%}")
        if preview is not None:
            st.image(preview, caption="Last detected pose frame")
        st.bar_chart({names[i]: float(scores[i]) for i in range(len(names))})
    except Exception as exc:
        st.error(str(exc))
    finally:
        Path(temp_path).unlink(missing_ok=True)


def live_camera() -> None:
    st.subheader("Live multi-person camera")
    st.caption("Uses browser camera access, OpenCV person detection, persistent IDs, per-person pose buffers, and the trained classifier once 30 poses are collected.")
    if model is None:
        st.warning("Live tracking works after a trained model is available; recognition is currently disabled.")
    threshold = st.slider("Known-person confidence threshold", .0, 1.0, .65, .01)
    target_id = st.number_input("Target tracking ID (0 = no target)", min_value=0, value=0, step=1)
    try:
        import av
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
    except ImportError:
        st.info("Install the optional live-camera dependency with: pip install streamlit-webrtc av")
        return
    from preprocessing.normalization import normalize_landmarks
    from preprocessing.pose_extractor import PoseDetector
    from preprocessing.realtime import GaitPerception, OpenCVPersonDetector
    from preprocessing.performance import PerformanceMonitor

    class LiveProcessor:
        def __init__(self):
            self.perception = GaitPerception(classifier=classify if model is not None else None,
                                             sequence_length=30, unknown_threshold=threshold)
            self.detector = OpenCVPersonDetector()
            self.pose = PoseDetector()
            self.lock = threading.Lock()
            self.last_timestamp = perf_counter()
            self.performance = PerformanceMonitor()

        def recv(self, frame):
            bgr = frame.to_ndarray(format="bgr24")
            now = perf_counter()
            with self.performance.measure("detection"):
                boxes = self.detector.detect(bgr)
            tracks = self.perception.update_detections(boxes, now)
            for track in tracks:
                x, y, w, h = track.bbox
                crop = bgr[max(0, y):min(bgr.shape[0], y + h), max(0, x):min(bgr.shape[1], x + w)]
                if crop.size:
                    with self.performance.measure("pose"):
                        result = self.pose.detect(crop[:, :, ::-1])
                    if result.pose_landmarks:
                        raw = np.asarray([[p.x, p.y, p.z] for p in result.pose_landmarks[0]], np.float32)
                        hips = raw[[23, 24], :2].mean(axis=0)
                        position = np.asarray((x + hips[0] * w, y + hips[1] * h))
                        self.perception.add_pose(track, normalize_landmarks(raw), position, now)
                direction, speed = track.movement()
                color = (0, 220, 0) if track.track_id == target_id else (255, 180, 0)
                cv2 = __import__("cv2")
                cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 2)
                text = f"ID {track.track_id} | {track.label} {track.confidence:.0%}"
                cv2.putText(bgr, text, (x, max(20, y - 22)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
                cv2.putText(bgr, f"{direction} {speed:.0f}px/s", (x, max(40, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
            elapsed = max(now - self.last_timestamp, 1e-6); self.last_timestamp = now
            cv2 = __import__("cv2")
            metrics = self.performance.summary()
            cv2.putText(bgr, f"FPS {1 / elapsed:.1f} | tracks {len(tracks)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
            cv2.putText(bgr, f"Detect {metrics.get('detection_ms', 0):.0f}ms Pose {metrics.get('pose_ms', 0):.0f}ms Mem {metrics['memory_mb']:.0f}MB", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 255, 0), 1)
            return av.VideoFrame.from_ndarray(bgr, format="bgr24")

    webrtc_streamer(key="gait-live-camera", mode=WebRtcMode.SENDRECV,
                    video_processor_factory=LiveProcessor,
                    media_stream_constraints={"video": True, "audio": False}, async_processing=True)
    st.caption("Green box = selected target. Speed is image-space pixels/second; camera calibration is required for metres/second.")


tab_live, tab_video, tab_metrics = st.tabs(["Live camera", "Recorded video", "Model performance"])
with tab_live:
    live_camera()
with tab_video:
    recorded_video()
with tab_metrics:
    metrics = Path("outputs/metrics.json")
    if metrics.exists():
        data = json.loads(metrics.read_text()); st.metric("Held-out accuracy", f"{data['accuracy']:.1%}")
        for image in ["outputs/confusion_matrix.png", "outputs/accuracy_curve.png", "outputs/loss_curve.png"]:
            if Path(image).exists(): st.image(image)
    else:
        st.caption("Run evaluation after training to display metrics and plots.")
