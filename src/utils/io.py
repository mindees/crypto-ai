"""Atomic IO helpers for JSON, YAML, and Parquet."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, payload: Any, *, indent: int = 2, sort_keys: bool = False) -> None:
    text = json.dumps(payload, indent=indent, sort_keys=sort_keys, default=str)
    _atomic_write_bytes(Path(path), text.encode("utf-8") + b"\n")


def read_yaml(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_yaml(path: str | Path, payload: Any) -> None:
    text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    _atomic_write_bytes(Path(path), text.encode("utf-8"))


def repo_root() -> Path:
    """Return the repo root by walking up from this file until a marker is found."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "config.yaml").exists():
            return parent
    return here.parents[2]
