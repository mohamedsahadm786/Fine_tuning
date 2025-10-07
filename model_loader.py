import os
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any

import torch

try:
    import joblib  # optional, for scaler.pkl
except Exception:
    joblib = None

DEFAULT_ZIP_ENV_KEY = "CONFIDENCE_MODEL_ZIP"  # env var to point to your model zip


class ModelBundle:
    """
    Holds everything needed for inference.
    """
    def __init__(self, model, aux: Optional[Dict[str, Any]], device: str, root_dir: Path):
        self.model = model
        self.aux = aux or {}
        self.device = device
        self.root_dir = root_dir


def _extract_zip(zip_path: Path, extract_to: Path) -> Path:
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    # If a single top-level folder exists, use it as root
    entries = [p for p in extract_to.iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extract_to


def _auto_find_checkpoint(root: Path) -> Optional[Path]:
    """
    Attempt to locate a PyTorch checkpoint within the extracted ZIP contents.

    The original notebook simply selected the first ``*.pt`` or ``*.pth`` file
    at the top level of the extracted directory. To mirror that behaviour,
    this helper first looks for common filenames (``model.pt``, ``model.pth``,
    etc.) in the root. If none are found, it enumerates all ``*.pt`` and
    ``*.pth`` files in the root directory (not recursively) and returns the
    lexicographically first one. Only if no top‑level checkpoint is present
    does it fall back to a recursive search.
    """
    # 1) Look for well‑known filenames in the root (non‑recursive)
    common = [
        "model.pt", "model.pth", "checkpoint.pt", "checkpoint.pth", "weights.pt",
        "artifacts/model.pt", "artifacts/checkpoint.pt",
    ]
    for name in common:
        p = root / name
        if p.exists():
            return p

    # 2) Any *.pt / *.pth files in the root directory (non‑recursive)
    candidates: list[Path] = []
    candidates += list(root.glob("*.pt"))
    candidates += list(root.glob("*.pth"))
    if candidates:
        # Sort to ensure deterministic selection
        return sorted(candidates)[0]

    # 3) Fallback: search recursively. Use rglob to find the first match.
    for p in root.rglob("*.pt"):
        return p
    for p in root.rglob("*.pth"):
        return p
    return None


def _load_aux_files(root: Path) -> Dict[str, Any]:
    aux: Dict[str, Any] = {}

    # class_map.json -> class_names in correct index order
    cm_path = root / "class_map.json"
    if cm_path.exists():
        try:
            import json
            data = json.loads(cm_path.read_text())
            # Expect {"0": "class_0", "1": "class_1", ...}
            class_names = [data[str(i)] for i in range(len(data))]
            aux["class_names"] = class_names
        except Exception:
            pass

    # optional scaler.pkl (sklearn StandardScaler) if you saved one
    if joblib is not None:
        for name in ["scaler.pkl", "standard_scaler.pkl"]:
            sp = root / name
            if sp.exists():
                try:
                    aux["scaler"] = joblib.load(sp)
                    break
                except Exception:
                    pass

    return aux


def load_bundle(zip_path: Optional[str] = None, device_preference: Optional[str] = None) -> ModelBundle:
    """
    Load your zipped Kaggle-trained pipeline (model + any aux files).
    Robust to different checkpoint dict formats.
    """
    if zip_path is None:
        zip_env = os.getenv(DEFAULT_ZIP_ENV_KEY)
        if not zip_env:
            raise FileNotFoundError(
                "Model zip path not provided. Set CONFIDENCE_MODEL_ZIP env var or pass zip_path explicitly."
            )
        zip_path = zip_env

    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Model zip not found at: {zip_path}")

    workdir = zip_path.parent / ("extracted_" + zip_path.stem)
    root = _extract_zip(zip_path, workdir)

    # Find a checkpoint file
    ckpt_path = _auto_find_checkpoint(root)
    if ckpt_path is None:
        raise FileNotFoundError(f"Could not locate a model checkpoint (.pt/.pth) under {root}")

    # Choose device
    device = "cuda" if (device_preference in (None, "cuda") and torch.cuda.is_available()) else "cpu"

    # Read checkpoint
    obj = torch.load(ckpt_path, map_location="cpu")  # load on CPU, then move to device
    from inference import build_model_for_inference  # defined in inference.py

    aux_from_zip = _load_aux_files(root)  # class_map.json, scaler.pkl, etc.
    aux_from_ckpt: Dict[str, Any] = {}

    # Unify dict formats:
    #  1) {"model_state_dict": ..., "hyperparams": ..., "aux": ...}
    #  2) {"state_dict": ..., "in_dim": ..., ...}
    #  3) {"model": model, "aux": ...}
    #  4) raw state_dict
    if isinstance(obj, dict):
        # pick weights
        state_dict = None
        if "model_state_dict" in obj:
            state_dict = obj["model_state_dict"]
        elif "state_dict" in obj:
            state_dict = obj["state_dict"]

        # pick hyperparams (possibly nested)
        hparams = obj.get("hyperparams")
        if hparams is None:
            # Allow flat keys
            hint_keys = ["in_dim", "attn_dim", "hidden_dim", "num_classes", "dropout"]
            if any(k in obj for k in hint_keys):
                hparams = {k: obj.get(k) for k in hint_keys}

        # aux from checkpoint, if present
        if "aux" in obj and isinstance(obj["aux"], dict):
            aux_from_ckpt = obj["aux"]

        if state_dict is not None:
            model = build_model_for_inference(hparams)
            # first try strict load; if keys are prefixed (e.g., "module."), fallback to non-strict
            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception:
                model.load_state_dict(state_dict, strict=False)
            model.to(device).eval()
        elif "model" in obj and hasattr(obj["model"], "eval"):
            model = obj["model"]
            model.to(device).eval()
        else:
            # dict but no known keys → treat the dict itself as a state_dict
            model = build_model_for_inference(None)
            try:
                model.load_state_dict(obj, strict=True)
            except Exception:
                model.load_state_dict(obj, strict=False)
            model.to(device).eval()
    else:
        # torch.save(model) directly
        model = obj
        model.to(device).eval()

    # Merge aux (zip files take precedence for class_names/scaler)
    aux = {}
    aux.update(aux_from_ckpt)
    aux.update(aux_from_zip)

    return ModelBundle(model=model, aux=aux, device=device, root_dir=root)
