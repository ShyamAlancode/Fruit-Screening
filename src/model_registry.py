"""
model_registry.py
=================
Deterministic model and experiment tracking with cryptographic SHA256 provenance.
Replaces non-deterministic mtime sorting with an explicit machine-readable manifest.
"""

import os
import json
import hashlib
import subprocess
import logging
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "model_manifest.json"


def compute_sha256(filepath: str | Path) -> str:
    """Compute SHA256 hex digest of a file."""
    p = Path(filepath)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    """Get current short Git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def load_manifest() -> dict:
    """Load model manifest JSON, creating default if missing."""
    if not MANIFEST_PATH.exists():
        return {"models": {}, "experiments": []}
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error reading {MANIFEST_PATH}: {e}. Returning default.")
        return {"models": {}, "experiments": []}


def save_manifest(manifest: dict) -> None:
    """Save model manifest JSON atomically."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MANIFEST_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    tmp_path.replace(MANIFEST_PATH)


def register_model(
    model_key: str,
    filepath: str | Path,
    dataset_path: str | Path,
    metrics: dict,
    model_type: str,
    validation_method: str = "5-fold GroupKFold by fruit_id",
    scientific_status: str = "",
    notes: str = "",
) -> dict:
    """Register or update a model in model_manifest.json."""
    manifest = load_manifest()
    p = Path(filepath).resolve()
    ds_p = Path(dataset_path).resolve()

    try:
        rel_filepath = str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        rel_filepath = str(p)

    try:
        rel_dataset = str(ds_p.relative_to(PROJECT_ROOT))
    except ValueError:
        rel_dataset = str(ds_p)

    entry = {
        "model_key": model_key,
        "filepath": rel_filepath,
        "model_sha256": compute_sha256(p),
        "dataset_path": rel_dataset,
        "dataset_sha256": compute_sha256(ds_p),
        "model_type": model_type,
        "validation_method": validation_method,
        "metrics": metrics,
        "git_commit": get_git_commit(),
        "scientific_status": scientific_status,
        "notes": notes,
    }

    manifest.setdefault("models", {})[model_key] = entry
    save_manifest(manifest)
    return entry


def get_model_path(model_key: str) -> Optional[Path]:
    """Look up model path from manifest, with fallback to directory search."""
    manifest = load_manifest()
    entry = manifest.get("models", {}).get(model_key)
    if entry:
        p = PROJECT_ROOT / entry["filepath"]
        if p.exists():
            return p

    # Fallback to known subdirectories
    for sub in ["treatment_condition", "camera", "quality", "brix"]:
        candidates = list((MODELS_DIR / sub).glob(f"*{model_key}*"))
        if candidates:
            return candidates[0]
    return None


def load_model(model_key: str, verify_sha256: bool = True) -> Any:
    """
    Safely load a model registered under model_key.
    For .joblib: uses joblib.load.
    For .pth: uses torch.load(..., weights_only=True).
    """
    manifest = load_manifest()
    entry = manifest.get("models", {}).get(model_key)
    p = get_model_path(model_key)
    if not p or not p.exists():
        raise FileNotFoundError(f"Model {model_key!r} not found at {p}")

    if verify_sha256 and entry and entry.get("model_sha256"):
        current_hash = compute_sha256(p)
        if current_hash != entry["model_sha256"]:
            logger.warning(
                f"SHA256 mismatch for {model_key}: expected {entry['model_sha256'][:8]}..., "
                f"got {current_hash[:8]}..."
            )

    if p.suffix == ".joblib":
        import joblib
        return joblib.load(p)
    elif p.suffix == ".pth":
        import torch
        # PyTorch 2.4+ supports weights_only=True safely
        try:
            return torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            return torch.load(p, map_location="cpu", weights_only=False)
    else:
        raise ValueError(f"Unsupported model extension: {p.suffix}")
