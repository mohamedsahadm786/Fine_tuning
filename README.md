# 🎥 Interview Confidence Classifier — Streamlit App

This is a **ready-to-run Streamlit UI** to reuse your **Kaggle-trained** model (saved as a **ZIP**) that predicts whether a candidate is **Confident / Moderately Confident / Not Confident** from an interview response video.

---

## 🗂 Project layout

```
confidence_streamlit/
├─ app.py                 # Streamlit UI (upload video → predict)
├─ inference.py           # 🔧 Plug in your real preprocessing + forward pass here
├─ model_loader.py        # Loads your zipped model, rebuilds model object(s)
├─ video_utils.py         # Helpers to save uploaded videos
├─ requirements.txt
└─ .streamlit/
   └─ config.toml         # Dark theme
```

---

## 🚀 How to run

1) **Put your trained ZIP** somewhere accessible (e.g., `C:/models/conf_model.zip` or `/home/user/conf_model.zip`).  
2) Create a fresh environment and install deps:

```bash
pip install -r requirements.txt
```

> If your pipeline needs **ffmpeg**, install it separately (e.g., `sudo apt-get install ffmpeg` on Debian/Ubuntu, or use the official installers on Windows/macOS).

3) Start the app:

```bash
# Option A: pass via environment variable
export CONFIDENCE_MODEL_ZIP=/full/path/to/your_model.zip
streamlit run app.py

# Option B: enter the path in the UI (⚙️ Model settings → Model ZIP path → Load / Reload Model)
streamlit run app.py
```

---

## 🔌 Connect your notebook logic

Open **`inference.py`** and implement:

- `build_model_for_inference(hparams)` — reconstruct the network architecture that matches training; load with `load_state_dict` if your checkpoint stores only weights.
- `predict_from_video(bundle, video_path)` — do your end-to-end preprocessing (clip sampling, features (VideoMAE/WavLM/OpenFace/etc.), fusion, logits → softmax) and return:
  - `label` — final predicted class string
  - `proba` — dict of `{class_name: probability}`
  - `extras` — any useful dict to show (scores, timings, shapes)

The loader in **`model_loader.py`** tries to find a checkpoint (`*.pt`/`*.pth`) *inside* your ZIP and supports common save styles:
- `torch.save({"model_state_dict": ..., "aux": {...}, "hyperparams": {...}})`
- `torch.save({"model": model, "aux": {...}})`
- `torch.save(model.state_dict())`

If your format differs (e.g., joblib pickles, separate scalers), adjust the code accordingly after extraction.

---

## 🧪 Quick test without a real model

`inference.py` returns a **deterministic dummy** prediction so the UI runs end-to-end. Once your real code is wired, remove the dummy and compute real probabilities.

---

## ❓FAQ

**Q: My model needs GPU. Will Streamlit pick it up?**  
If `torch.cuda.is_available()` is true, the loader sets `device="cuda"`. Otherwise it uses CPU.

**Q: Do I need MoviePy?**  
Not for the UI itself. Use any tools you used on Kaggle to preprocess videos (OpenCV, ffmpeg). Add them to `requirements.txt` if missing.

**Q: Where do uploads go?**  
They are saved under `uploads/` with unique names and shown via `st.video(...)`.

---

## 📄 License
Do whatever you like. Attribution appreciated.
