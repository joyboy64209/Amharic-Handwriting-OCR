"""Project configuration loader.

Reads ``config.yaml`` (project root) and exposes typed settings used by
preprocessing, dataset, models, training, inference, and the API. All paths
are resolved relative to the project root (the folder that contains this
file's parent ``src`` package).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

try:  # PyYAML is optional; fall back to a minimal built-in parser if absent.
    import yaml  # type: ignore

    def _load_yaml(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

except Exception:  # pragma: no cover - yaml unavailable
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Very small YAML subset parser (scalars, lists, nested mappings)."""
        data: Dict[str, Any] = {}
        stack: List[tuple[int, Dict[str, Any]]] = [(-1, data)]
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                indent = len(line) - len(line.lstrip(" "))
                key_val = line.strip()
                if ":" not in key_val:
                    continue
                key, _, val = key_val.partition(":")
                key = key.strip()
                val = val.strip()
                while stack and indent <= stack[-1][0]:
                    stack.pop()
                parent = stack[-1][1]
                if val == "":
                    child: Dict[str, Any] = {}
                    parent[key] = child
                    stack.append((indent, child))
                elif val.startswith("[") and val.endswith("]"):
                    inner = val[1:-1].strip()
                    parent[key] = [v.strip().strip('"').strip("'") for v in inner.split(",")] if inner else []
                else:
                    parent[key] = _coerce(val)
        return data


def _coerce(val: str) -> Any:
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_RAW: Dict[str, Any] = _load_yaml(CONFIG_PATH)


def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _resolve(rel: str) -> Path:
    return (PROJECT_ROOT / rel).resolve()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    char_dataset_dir: Path = field(default_factory=lambda: _resolve(_get(_RAW, "data", "char_dataset_dir", default="data/uni_dataset")))
    line_labels_file: Path = field(default_factory=lambda: _resolve(_get(_RAW, "data", "line_labels_file", default="data/lines.jsonl")))
    line_images_dir: Path = field(default_factory=lambda: _resolve(_get(_RAW, "data", "line_images_dir", default="data/lines")))


@dataclass
class ModelConfig:
    cnn_checkpoint: Path = field(default_factory=lambda: _resolve(_get(_RAW, "models", "cnn_checkpoint", default="models/amharic_ocr_v2.pth")))
    crnn_checkpoint: Path = field(default_factory=lambda: _resolve(_get(_RAW, "models", "crnn_checkpoint", default="models/crnn_amharic_line.pth")))


@dataclass
class PreprocessConfig:
    img_height: int = int(_get(_RAW, "preprocessing", "img_height", default=28))
    img_width: int = int(_get(_RAW, "preprocessing", "img_width", default=28))
    mean: float = float(_get(_RAW, "preprocessing", "mean", default=0.239))
    std: float = float(_get(_RAW, "preprocessing", "std", default=0.407))
    max_skew_angle: float = float(_get(_RAW, "preprocessing", "max_skew_angle", default=45))


@dataclass
class TrainingConfig:
    batch_size: int = int(_get(_RAW, "training", "batch_size", default=64))
    epochs: int = int(_get(_RAW, "training", "epochs", default=20))
    lr: float = float(_get(_RAW, "training", "lr", default=0.001))
    val_split: float = float(_get(_RAW, "training", "val_split", default=0.2))
    seed: int = int(_get(_RAW, "training", "seed", default=42))
    rotation_degrees: float = float(_get(_RAW, "training", "rotation_degrees", default=8))
    scale: float = float(_get(_RAW, "training", "scale", default=0.95))
    shear_degrees: float = float(_get(_RAW, "training", "shear_degrees", default=4))
    brightness: float = float(_get(_RAW, "training", "brightness", default=0.15))


@dataclass
class InferenceConfig:
    default_mode: str = str(_get(_RAW, "inference", "default_mode", default="char"))
    low_confidence_threshold: float = float(_get(_RAW, "inference", "low_confidence_threshold", default=0.60))


@dataclass
class ApiConfig:
    max_upload_bytes: int = int(_get(_RAW, "api", "max_upload_bytes", default=2097152))
    allowed_extensions: List[str] = field(default_factory=lambda: list(_get(_RAW, "api", "allowed_extensions", default=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"])))
    server_host: str = str(_get(_RAW, "api", "server_host", default="127.0.0.1"))
    server_port: int = int(_get(_RAW, "api", "server_port", default=8000))


@dataclass
class Settings:
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    api: ApiConfig = field(default_factory=ApiConfig)


SETTINGS = Settings()

__all__ = ["SETTINGS", "Settings", "PROJECT_ROOT", "CONFIG_PATH"]