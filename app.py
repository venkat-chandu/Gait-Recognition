"""Streamlit demo for the pose-based gait-recognition pipeline."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Gait Recognition", page_icon="🚶", layout="wide")
st.title("🚶 Pose-Based Gait Recognition")
st.caption("Educational demonstration — scores are not calibrated confidence estimates, and predictions are only meaningful for identities represented in training data.")

model_path = Path("models_artifacts/gait_model.keras")
if not model_path.exists():
    st.info("No trained model found. Generate data and train the model first (see README).")
    st.stop()

@st.cache_resource
def load_assets():
    import tensorflow as tf
    return tf.keras.models.load_model(model_path), json.loads(Path("models_artifacts/class_names.json").read_text())

model, names = load_assets()
upload = st.file_uploader("Upload a clear, single-person walking video", type=["mp4", "avi", "mov"])
if upload:
    st.video(upload)
    if st.button("Analyze gait", type="primary"):
        suffix = Path(upload.name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(upload.getbuffer()); temp_path = tmp.name
        try:
            from preprocessing.video_processor import video_to_sequence
            with st.spinner("Extracting pose landmarks and running the model…"):
                sequence, preview = video_to_sequence(temp_path, model.input_shape[1])
                scores = model.predict(sequence[None], verbose=0)[0]
            index = int(scores.argmax())
            a, b = st.columns(2); a.metric("Predicted class", names[index]); b.metric("Model score", f"{scores[index]:.1%}")
            if preview is not None: st.image(preview, caption="Last detected pose frame")
            st.bar_chart({names[i]: float(scores[i]) for i in range(len(names))})
        except Exception as exc:
            st.error(str(exc))
        finally:
            Path(temp_path).unlink(missing_ok=True)

st.divider(); st.subheader("Model performance")
metrics = Path("outputs/metrics.json")
if metrics.exists():
    data = json.loads(metrics.read_text()); st.metric("Held-out accuracy", f"{data['accuracy']:.1%}")
    for image in ["outputs/confusion_matrix.png", "outputs/accuracy_curve.png", "outputs/loss_curve.png"]:
        if Path(image).exists(): st.image(image)
else:
    st.caption("Run evaluation after training to display metrics and plots.")
