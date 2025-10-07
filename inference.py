# inference.py
# ===== Notebook-faithful inference (from Prediction.ipynb) =====
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, List

import os, json, glob, tempfile, subprocess
import numpy as np
import torch
import torch.nn as nn
import cv2
from scipy.io import wavfile

# ---- transformers: VideoMAE + WavLM ----
from transformers import VideoMAEModel
try:
    # newer transformers
    from transformers import VideoMAEImageProcessor as VideoMAEFeatureExtractor
except Exception:
    from transformers import VideoMAEFeatureExtractor
from transformers import WavLMModel, AutoFeatureExtractor

# ---------- constants (match notebook) ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEOMAE_CKPT = "MCG-NJU/videomae-base"
WAVLM_CKPT    = "microsoft/wavlm-base-plus"

CLIP_SECONDS    = 10
FRAMES_PER_CLIP = 16
MIN_TAIL_SEC    = 3.0
TARGET_AUDIO_SR = 16000

# Default class labels used for fallback when no class_map is provided in the
# model bundle. Importing this constant allows ``app.py`` to reference
# ``inference.CLASS_LABELS`` without raising ImportError. The ordering of
# these labels corresponds to the class indices 0, 1 and 2 as used in
# ``predict_from_video``.
CLASS_LABELS = ["Not confident", "Moderately confident", "Highly confident"]

# Prefer bundled ffmpeg (works on Windows)
try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_BIN = "ffmpeg"  # requires ffmpeg on PATH

# --------------- helpers: video duration & clips ---------------
def get_video_duration_cv2(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if total_frames <= 0:
        return 0.0
    return float(total_frames) / float(fps if fps > 1e-6 else 25.0)

def split_into_segments(duration: float) -> List[Tuple[float, float]]:
    n_full = int(duration // CLIP_SECONDS)
    starts = [i * CLIP_SECONDS for i in range(n_full)]
    rem = duration - n_full * CLIP_SECONDS
    segs = [(s, CLIP_SECONDS) for s in starts]
    if rem > 0:
        segs.append((n_full * CLIP_SECONDS, rem))
    return segs

def select_clips(video_path: str) -> List[Dict[str, float]]:
    """
    Notebook logic:
    - split video into 10s chunks + a tail
    - drop segments shorter than MIN_TAIL_SEC
    - if there’s a tail and >5 total segments, prefer full 10s segments
    - if base has >=5, take 5 uniformly; else keep all
    - no extension/padding flag (extend=False in notebook)
    """
    dur = float(get_video_duration_cv2(video_path))
    segs = split_into_segments(dur)
    segs = [(s, d) for (s, d) in segs if d >= MIN_TAIL_SEC]
    if not segs:
        return []

    fulls = [(s, d) for (s, d) in segs if d >= CLIP_SECONDS - 1e-3]
    tail  = [(s, d) for (s, d) in segs if d <  CLIP_SECONDS - 1e-3]

    base = fulls if (len(segs) > 5 and tail) else segs

    clips: List[Dict[str, float]] = []
    if len(base) >= 5:
        idxs = np.round(np.linspace(0, len(base) - 1, 5)).astype(int).tolist()
        for i in idxs:
            s, d = base[i]
            clips.append({"start": float(s), "duration": float(d), "extend": False})
    else:
        for s, d in base:
            clips.append({"start": float(s), "duration": float(d), "extend": False})
    return clips

def load_frames(video_path: str, start: float, duration: float, num_frames: int,
                extend: bool = False) -> List[np.ndarray]:
    """Sample frames uniformly over the *actual* clip duration (no artificial extension)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_dur = total_frames / max(fps, 1e-6)

    target_dur = CLIP_SECONDS if extend else duration  # extend is always False in notebook
    times = np.linspace(start, start + target_dur, num_frames, endpoint=False)

    frames: List[np.ndarray] = []
    for t in times:
        t_clamped = min(max(t, 0.0), max(video_dur - 1e-3, 0.0))
        frame_idx = int(round(t_clamped * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx, 0))
        ok, frame_bgr = cap.read()
        if not ok:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
            continue
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def extract_audio(video_path: str, start: float, duration: float, extend: bool = False,
                  target_sr: int = TARGET_AUDIO_SR) -> np.ndarray:
    start = max(0.0, float(start))
    duration = max(0.0, float(duration))
    want_n = int(round(target_sr * duration))
    if want_n <= 0:
        return np.zeros(0, dtype=np.float32)

    tmpdir = tempfile.mkdtemp(prefix="aud_")
    wav_path = os.path.join(tmpdir, "cut.wav")
    try:
        cmd = [
            FFMPEG_BIN, "-hide_banner", "-loglevel", "error",
            "-ss", f"{start}", "-t", f"{duration}",
            "-i", video_path, "-vn", "-ac", "1", "-ar", str(target_sr),
            "-f", "wav", wav_path
        ]
        subprocess.run(cmd, check=True)

        sr, audio = wavfile.read(wav_path)
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # normalize if int16
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32768.0

        # pad/truncate to exact length (notebook pads only when shorter)
        if len(audio) < want_n:
            audio = np.pad(audio, (0, want_n - len(audio)), mode="constant")
        elif len(audio) > want_n:
            audio = audio[:want_n]
        return audio
    finally:
        try:
            if os.path.exists(wav_path): os.remove(wav_path)
            os.rmdir(tmpdir)
        except Exception:
            pass

# --------------- encoders (same as notebook) ---------------
print("[load] Loading encoders (first time may download weights)…")
videomae_processor = VideoMAEFeatureExtractor.from_pretrained(VIDEOMAE_CKPT)
videomae_model     = VideoMAEModel.from_pretrained(VIDEOMAE_CKPT).to(DEVICE).eval()

wavlm_processor    = AutoFeatureExtractor.from_pretrained(WAVLM_CKPT)
wavlm_model        = WavLMModel.from_pretrained(WAVLM_CKPT).to(DEVICE).eval()

@torch.no_grad()
def videomae_embedding(frames: List[np.ndarray]) -> torch.Tensor:
    inputs = videomae_processor([frames], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out = videomae_model(**inputs)
    # mean pool over temporal tokens -> [768]
    last = out.last_hidden_state
    return last.mean(dim=1).squeeze(0)

@torch.no_grad()
def wavlm_embedding(audio: np.ndarray) -> torch.Tensor:
    inputs = wavlm_processor(audio, sampling_rate=TARGET_AUDIO_SR, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out = wavlm_model(**inputs)
    last = out.last_hidden_state
    return last.mean(dim=1).squeeze(0)  # [768]

def compute_clip_embeddings(video_path: str) -> np.ndarray:
    specs = select_clips(video_path)
    if not specs:
        raise RuntimeError("No usable clips found for the video (too short?)")
    clips: List[torch.Tensor] = []
    for spec in specs:
        s, d, ext = spec["start"], spec["duration"], spec["extend"]
        frames = load_frames(video_path, s, d, FRAMES_PER_CLIP, ext)
        v_emb  = videomae_embedding(frames)
        a_emb  = wavlm_embedding(extract_audio(video_path, s, d, ext))
        h = torch.cat([v_emb, a_emb], dim=0)  # [1536]
        clips.append(h)
    X = torch.stack(clips, dim=0).cpu().numpy().astype("float32")  # [N, 1536]
    return X

# --------------- model (matches notebook) ---------------
class ClipAttentionClassifier(nn.Module):
    def __init__(self, in_dim, attn_dim, hidden_dim, num_classes=3, dropout=0.30):
        super().__init__()
        self.attn_W  = nn.Linear(in_dim, attn_dim)
        self.attn_v  = nn.Linear(attn_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: [B, N, D], lengths: [B]
        B, N, D = x.shape
        e = self.attn_v(torch.tanh(self.attn_W(x))).squeeze(-1)  # [B, N]
        mask = (torch.arange(N, device=x.device).unsqueeze(0) < lengths.unsqueeze(1))
        e_masked = e.masked_fill(~mask, -1e9)
        w = torch.softmax(e_masked, dim=1)                       # [B, N]
        pooled = torch.sum(w.unsqueeze(-1) * x, dim=1)           # [B, D]
        logits = self.classifier(pooled)                         # [B, C]
        return logits

# --------------- hooks used by model_loader.py ---------------
def build_model_for_inference(hparams: Any):
    """
    Called by model_loader when your checkpoint only contains weights.
    Matches the notebook hyperparams:
      in_dim, attn_dim, hidden_dim, num_classes, dropout
    If hparams is None, fall back to defaults (1536, 256, 512, 3, 0.30).
    """
    if hparams is None:
        in_dim, attn_dim, hidden_dim, num_classes, dropout = 1536, 256, 512, 3, 0.30
    else:
        in_dim     = int(hparams.get("in_dim", 1536))
        attn_dim   = int(hparams.get("attn_dim", 256))
        hidden_dim = int(hparams.get("hidden_dim", 512))
        num_classes= int(hparams.get("num_classes", 3))
        dropout    = float(hparams.get("dropout", 0.30))
    return ClipAttentionClassifier(in_dim, attn_dim, hidden_dim, num_classes, dropout)

@torch.inference_mode()
def predict_from_video(bundle, video_path: Path) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
    """
    Notebook-faithful prediction:
      X = compute_clip_embeddings(video)  # [N, 1536]
      logits = model(attn_pool(X))
      softmax -> probs; label via class_map.json if present
    """
    # 1) features
    X = compute_clip_embeddings(str(video_path))
    N, D = X.shape

    # 2) optional scaler saved in bundle.aux (if you added one later)
    scaler = getattr(bundle, "aux", {}).get("scaler", None)
    if scaler is not None:
        X = scaler.transform(X)

    # 3) forward
    x = torch.from_numpy(X).float().to(bundle.device)[None, :, :]  # [1, N, D]
    lengths = torch.tensor([N], dtype=torch.long, device=bundle.device)
    logits = bundle.model(x, lengths).squeeze(0)                   # [C]
    probs = torch.softmax(logits, dim=0).cpu().numpy()

    # 4) class names — prefer class_map.json inside the extracted zip
        # ---- class names: prefer ones loaded by the loader (from class_map.json)
    class_names = getattr(bundle, "aux", {}).get("class_names")
    if not class_names:
        try:
            cm_path = Path(bundle.root_dir) / "class_map.json"
            if cm_path.exists():
                import json
                data = json.loads(cm_path.read_text())
                class_names = [data[str(i)] for i in range(len(probs))]
        except Exception:
            pass
    if not class_names:
        class_names = ["Not confident", "Moderately confident", "Highly confident"]

    proba = {name: float(p) for name, p in zip(class_names, probs)}
    label = class_names[int(np.argmax(probs))]
 

    extras = {
        "n_clips": int(N),
        "feature_dim": int(D),
        "frames_per_clip": FRAMES_PER_CLIP,
        "seconds_per_clip": CLIP_SECONDS,
        "device": bundle.device,
    }
    return label, proba, extras
