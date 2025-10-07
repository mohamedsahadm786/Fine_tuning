from pathlib import Path
import uuid

def save_uploaded_video(uploaded_file_bytes: bytes, original_name: str, dest_dir: Path) -> Path:
    """Save uploaded video bytes to dest_dir with a safe unique name."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = "".join(Path(original_name).suffixes) or ".mp4"
    safe_name = f"upload_{uuid.uuid4().hex}{suffix}"
    out_path = dest_dir / safe_name
    out_path.write_bytes(uploaded_file_bytes)
    return out_path
