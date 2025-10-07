import os
from pathlib import Path
from typing import Optional

import streamlit as st

from model_loader import load_bundle, DEFAULT_ZIP_ENV_KEY
from inference import predict_from_video, CLASS_LABELS  # noqa: F401  (kept for future use)
from video_utils import save_uploaded_video

APP_TITLE = "Interview Confidence Classifier"
APP_DESC = """
Upload a short interview response video and predict **Confidence / Moderate / Not Confident**.
"""

@st.cache_resource(show_spinner=True)
def _load_bundle_cached(zip_path: Optional[str] = None):
    return load_bundle(zip_path)

def _safe_rerun():
    """Cross-version rerun helper (Streamlit >=1.27 uses st.rerun)."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.warning("Please refresh the page to reload the model.")

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🎥", layout="centered")
    st.title("🎥 " + APP_TITLE)
    st.markdown(APP_DESC)

    with st.expander("⚙️ Model settings", expanded=False):
        st.write("Provide the path to your **zipped** Kaggle model bundle.")
        default_zip = os.getenv(DEFAULT_ZIP_ENV_KEY, "")
        zip_path_in = st.text_input(
            "Model ZIP path",
            value=default_zip,
            placeholder="e.g., C:/models/confidence_model.zip",
            help=f"You can also set the {DEFAULT_ZIP_ENV_KEY} environment variable."
        )
        if st.button("Load / Reload Model"):
            # Clear ONLY our cached loader
            try:
                _load_bundle_cached.clear()
            except Exception:
                pass
            st.session_state["model_ready"] = False
            _safe_rerun()

    # Try load once per session
    bundle = None
    try:
        # If the input box is empty, let load_bundle() fall back to the env var.
        path_arg = zip_path_in.strip() or None
        bundle = _load_bundle_cached(path_arg)
        st.success(f"Model loaded on device: **{bundle.device}**")
        st.session_state["model_ready"] = True
    except Exception as e:
        load_err = str(e)
        st.warning("Model not loaded yet. Set the ZIP path above, then click **Load / Reload Model**.")
        st.caption(f"Details: {load_err}")

    st.divider()
    st.subheader("1) Upload your video")
    up = st.file_uploader(
        "Choose a video file (MP4, MOV, MKV, AVI)",
        type=["mp4", "mov", "mkv", "avi"]
    )
    temp_dir = Path("uploads")
    saved_path = None

    if up is not None:
        saved_path = save_uploaded_video(up.read(), up.name, temp_dir)
        st.video(str(saved_path))

    st.subheader("2) Run prediction")
    infer_btn = st.button(
        "🔎 Predict Confidence",
        disabled=(up is None or not st.session_state.get("model_ready", False))
    )

    if infer_btn and saved_path and bundle:
        with st.spinner("Analyzing video and running model …"):
            try:
                label, proba, extras = predict_from_video(bundle, saved_path)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        st.success(f"**Prediction:** {label}")
        st.progress(max(0.0, min(1.0, proba.get(label, 0.0))))

        with st.expander("🔢 Class probabilities", expanded=True):
            for cls, p in proba.items():
                st.write(f"{cls}: {p:.3f}")
                st.progress(max(0.0, min(1.0, p)))

        with st.expander("🧾 Details / Debug info"):
            st.json(extras)

        st.info("Tip: You can now try another file, or change the model ZIP path and reload.")

    st.divider()
    st.caption("Built with Streamlit. Replace the logic in **inference.py** to hook up your real pipeline.")

if __name__ == "__main__":
    main()
