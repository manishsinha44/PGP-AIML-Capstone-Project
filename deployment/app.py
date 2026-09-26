"""
Streamlit app: Pneumonia (Chest X-Ray) Classification — Deployment Demo
========================================================================
Upload a chest X-ray DICOM (.dcm) file and see the predicted class
(Normal / Lung Opacity / No Lung Opacity / Not Normal) with its probability.

Uses the exact same preprocessing (src/preprocessing.py) that the training
notebook used, so there is no train/serve skew.

Run locally:
    streamlit run app.py

Run in a container / GitHub Codespaces:
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
"""

import json
import os
import sys

import numpy as np
import streamlit as st

# Make the shared `src` package importable whether this file is run from the
# repo root or from inside deployment/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for p in (_REPO_ROOT, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src import preprocessing as pp  # noqa: E402

MODEL_DIR = os.path.join(_THIS_DIR, "model_artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")


@st.cache_resource
def load_model_and_config():
    import tensorflow as tf

    if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    return model, config


def main():
    st.set_page_config(page_title="Pneumonia X-Ray Classifier", page_icon="🫁", layout="centered")
    st.title("🫁 Chest X-Ray Pneumonia Classifier")
    st.caption(
        "Decision-support demo — classifies a chest X-ray as Normal, Lung Opacity, or "
        "No Lung Opacity / Not Normal. This is a research/education prototype, not a "
        "certified diagnostic device; a qualified radiologist should confirm any result."
    )

    model, config = load_model_and_config()
    if model is None:
        st.error(
            "No trained model found at `deployment/model_artifacts/`. Run the training "
            "notebook's 'Model Deployment' section first — it exports `best_model.keras` "
            "and `model_config.json` into this folder."
        )
        st.stop()

    with st.sidebar:
        st.subheader("Model in use")
        st.write(f"**Architecture:** {config['model_name']}")
        st.write(f"**Test accuracy:** {config['test_accuracy']:.3f}")
        st.write(f"**Test macro-F1:** {config['test_macro_f1']:.3f}")
        st.write(f"**Input size:** {config['image_size']}×{config['image_size']}×{config['channels']}")

    uploaded_file = st.file_uploader("Upload a chest X-ray (.dcm)", type=["dcm"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        try:
            x = pp.load_and_prepare_from_bytes(
                file_bytes, size=config["image_size"], channels=config["channels"]
            )
        except Exception as e:
            st.error(f"Could not read this file as a DICOM image: {e}")
            st.stop()

        display_img = x[..., 0] if config["channels"] == 3 else x[..., 0]
        st.image(display_img, caption="Uploaded X-ray (preprocessed)", use_container_width=True, clamp=True)

        with st.spinner("Running inference..."):
            probs = model.predict(x[np.newaxis, ...], verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        pred_class = pp.IDX_TO_CLASS[pred_idx]

        st.subheader("Prediction")
        st.markdown(f"### **{pred_class}**  ({probs[pred_idx] * 100:.1f}% confidence)")

        st.write("Class probabilities:")
        for i, cls in enumerate(pp.CLASS_NAMES):
            st.progress(float(probs[i]), text=f"{cls}: {probs[i] * 100:.1f}%")

        if pred_class == "Lung Opacity":
            st.warning(
                "Predicted **Lung Opacity** — flag for radiologist priority review. "
                "This tool is a second opinion, not a standalone diagnosis."
            )


if __name__ == "__main__":
    main()
