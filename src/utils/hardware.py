"""Hardware detection: GPUs, Kaggle dual-T4, MirroredStrategy, mixed precision.

Run directly: ``python -m src.utils.hardware``
"""
from __future__ import annotations

import json
import os
import platform
import sys
from typing import Any


def _detect_environment() -> str:
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_URL_BASE" in os.environ:
        return "kaggle"
    if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return "github_actions"
    return "local"


def _safe_get_tf_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "tensorflow_installed": False,
        "tensorflow_version": None,
        "keras_version": None,
        "physical_gpus": [],
        "logical_gpus": [],
        "gpu_count": 0,
        "gpu_names": [],
        "mixed_precision_policy": None,
        "would_use_mirrored_strategy": False,
        "is_kaggle_dual_t4": False,
    }
    try:
        import tensorflow as tf
    except ImportError as exc:
        info["error"] = f"tensorflow import failed: {exc}"
        return info

    info["tensorflow_installed"] = True
    info["tensorflow_version"] = tf.__version__
    try:
        info["keras_version"] = tf.keras.__version__
    except AttributeError:
        info["keras_version"] = "unknown"

    physical = tf.config.list_physical_devices("GPU")
    info["physical_gpus"] = [d.name for d in physical]
    info["gpu_count"] = len(physical)

    for device in physical:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except (RuntimeError, ValueError):
            pass

    gpu_details: list[str] = []
    for device in physical:
        try:
            details = tf.config.experimental.get_device_details(device)
            name = details.get("device_name", "unknown")
        except Exception:
            name = "unknown"
        gpu_details.append(name)
    info["gpu_names"] = gpu_details

    logical = tf.config.list_logical_devices("GPU")
    info["logical_gpus"] = [d.name for d in logical]

    env = _detect_environment()
    is_dual_t4 = (
        env == "kaggle"
        and len(physical) >= 2
        and all("T4" in name for name in gpu_details if name)
    )
    info["is_kaggle_dual_t4"] = is_dual_t4
    info["would_use_mirrored_strategy"] = len(physical) >= 2

    try:
        policy = tf.keras.mixed_precision.global_policy()
        info["mixed_precision_policy"] = str(policy.name)
    except Exception:
        info["mixed_precision_policy"] = None

    return info


def detect_hardware() -> dict[str, Any]:
    report: dict[str, Any] = {
        "environment": _detect_environment(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "executable": sys.executable,
    }
    report.update(_safe_get_tf_info())
    return report


def main() -> None:
    report = detect_hardware()
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
