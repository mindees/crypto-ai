"""Disabled-by-default stubs for paid sources.

Per the spec's "free by default" rule, these adapters exist as placeholders
so the codebase has a clear extension point if a user later supplies their
own paid API keys. They never run unless the user explicitly enables them
in ``configs/config.yaml`` (``sources.enable_paid_sources: true``).

Currently scaffolded:

* Glassnode (on-chain metrics)
* CryptoQuant (exchange flow, miner data)
* CoinGlass (liquidations, heatmaps)
* Coinalyze (derivatives, liquidations)
* Amberdata (institutional crypto data)

CLI is informational only — running it always reports each source as
disabled unless the config opts in.
"""
from __future__ import annotations

import argparse
import sys

from src.ingest._adapter_base import AdapterResult, print_result, update_source_registry
from src.utils.io import read_yaml, repo_root
from src.utils.logging import get_logger

_log = get_logger("ingest.paid_stubs")

PAID_SOURCES: tuple[tuple[str, str, list[str]], ...] = (
    (
        "glassnode",
        "glassnode.com",
        ["On-chain metrics (active addrs, SOPR, MVRV, etc.).",
         "Requires paid API tier for full coverage; free tier is limited."],
    ),
    (
        "cryptoquant",
        "cryptoquant.com",
        ["Exchange flow, miner outflows, stablecoin metrics.",
         "Mostly paid."],
    ),
    (
        "coinglass",
        "coinglass.com",
        ["Liquidation heatmaps, OI by exchange, options OI.",
         "Free tier insufficient for full historical liquidation data."],
    ),
    (
        "coinalyze",
        "coinalyze.net",
        ["Per-exchange OI, liquidations, funding aggregates.",
         "Free tier limited; historical depth shallow without subscription."],
    ),
    (
        "amberdata",
        "amberdata.io",
        ["Institutional crypto market + DeFi data.",
         "No free tier of meaningful depth."],
    ),
)


def _enabled() -> bool:
    cfg_path = repo_root() / "configs" / "config.yaml"
    try:
        cfg = read_yaml(cfg_path)
    except Exception:  # noqa: BLE001
        return False
    return bool((cfg.get("sources") or {}).get("enable_paid_sources"))


def fetch() -> list[AdapterResult]:
    enabled = _enabled()
    results: list[AdapterResult] = []
    for name, host, limitations in PAID_SOURCES:
        res = AdapterResult(
            name=f"paid_{name}",
            source=host,
            available=False,
            known_limitations=list(limitations),
        )
        if enabled:
            res.reason = (
                f"{name} adapter stub is not implemented; paid sources are disabled "
                "in this build by design."
            )
        else:
            res.reason = (
                f"{name} is disabled: sources.enable_paid_sources=false (spec default). "
                "Bring your own API key + implement adapter to use."
            )
        results.append(res)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.ingest.paid_stubs")
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args(argv)

    results = fetch()
    if not args.no_persist:
        update_source_registry(results)
    for res in results:
        print_result(res)
    # Intentional: stubs are not "errors" — exit 0 even when nothing ran.
    return 0


if __name__ == "__main__":
    sys.exit(main())
