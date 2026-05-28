"""Roll back production to a previous model.

Sets the target model to ``production``, marks the currently-production model
as ``rolled_back``, and repoints ``artifacts/production/current_model.json``.
Artifacts are never deleted or overwritten.

CLI::

    python -m src.models.rollback --model-id <previous_model_id>
"""
from __future__ import annotations

import argparse
import sys

from src.models import registry
from src.utils.io import repo_root
from src.utils.logging import get_logger

_log = get_logger("models.rollback")


def rollback_to(model_id: str, *, root=None) -> bool:
    root = root or repo_root()
    target = registry.get_model(model_id, root=root)
    if target is None:
        _log.error("model_id %s not in registry", model_id)
        return False

    current = registry.get_production(root)
    if current and current["model_id"] != model_id:
        registry.set_status(current["model_id"], "rolled_back", root=root,
                            note=f"rolled back in favour of {model_id}")
    registry.set_status(model_id, "production", root=root,
                        promotion_decision="approved",
                        note="restored via rollback")
    registry.set_current_model_pointer(model_id, root=root)
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.models.rollback")
    p.add_argument("--model-id", required=True)
    args = p.parse_args(argv)

    root = repo_root()
    ok = rollback_to(args.model_id, root=root)
    if not ok:
        print(f"rollback failed: {args.model_id} not found in registry")
        return 2
    prod = registry.get_production(root)
    print(f"rolled back. production is now: {prod['model_id'] if prod else '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
