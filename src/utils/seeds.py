"""Deterministic seeding across Python, NumPy, and TensorFlow."""
from __future__ import annotations

import os
import random


def set_global_seed(seed: int = 42, *, deterministic_tf_ops: bool = True) -> None:
    """Set seeds for Python, NumPy, and TensorFlow if available.

    Call once at the very start of any training/evaluation entrypoint.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np  # noqa: WPS433
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf  # noqa: WPS433
        tf.random.set_seed(seed)
        if deterministic_tf_ops:
            os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
            os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    except ImportError:
        pass
