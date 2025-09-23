import json
from pathlib import Path
from typing import Any, Dict

class DotDict(dict):
    """Dictionary with attribute-style access (read-only convenience)."""
    def __getattr__(self, item):
        v = self.get(item)
        if isinstance(v, dict):
            return DotDict(v)
        return v
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        cfg = json.load(f)
    return cfg


def as_dotdict(cfg: Dict[str, Any]) -> DotDict:
    return DotDict(cfg)


def merge_override(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge override into base (override wins)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge_override(base[k], v)
        else:
            base[k] = v
    return base


def save_effective(cfg: Dict[str, Any], out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
