"""Config + path helpers. Everything is relative to the repo root unless an absolute path is given."""
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None) -> dict:
    p = Path(path or os.environ.get("CSU_CONFIG") or ROOT / "config" / "config.yaml")
    cfg = yaml.safe_load(p.read_text())
    cfg.setdefault("catalog", dict(cfg["products"]))  # full product definitions; cfg["products"] = what gets modelled
    return cfg


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def paths(cfg: dict) -> dict:
    d = _abs(cfg["paths"]["data"])
    out = {
        "data": d,
        "raw": d / "raw", "bronze": d / "bronze", "silver": d / "silver", "gold": d / "gold",
        "landing": d / "stream" / "landing", "processed": d / "stream" / "processed",
        "quarantine": d / "quarantine",
        "models": _abs(cfg["paths"]["models"]),
        "reports": _abs(cfg["paths"]["reports"]),
    }
    out["monitoring"] = out["reports"] / "monitoring"
    for k, v in out.items():
        v.mkdir(parents=True, exist_ok=True)
    return out


def prefix(cfg, product):
    return cfg["products"][product]["prefix"]
