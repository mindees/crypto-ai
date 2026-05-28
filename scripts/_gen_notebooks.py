"""Generate the 4 Phase-8 notebooks as valid nbformat-4 JSON.

Throwaway builder: run once to (re)create notebooks/. Each notebook imports
the real project modules from src/ (no duplicated logic), detects GPUs, and
supports resume via existing checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


DISCLAIMER = (
    "> **Decision-support only. Not financial advice.** Markets are noisy and "
    "adversarial. A model that fails to beat baselines after fees/slippage must "
    "not be used for trading.\n"
)

GPU_CHECK = '''\
# --- GPU / environment check ---
import json
from src.utils.hardware import detect_hardware
report = detect_hardware()
print(json.dumps(report, indent=2, default=str))

if report.get("would_use_mirrored_strategy"):
    print("\\n>>> 2+ GPUs detected — training will use tf.distribute.MirroredStrategy().")
elif report.get("gpu_count", 0) == 1:
    print("\\n>>> single GPU — standard single-device training.")
else:
    print("\\n>>> no GPU — CPU smoke only. Enable the accelerator for real training.")
'''

CONFIG_BLOCK = '''\
# --- single visible config block (override here, no need to edit src/) ---
ASSETS       = ["BTCUSDT", "ETHUSDT"]
TIMEFRAMES   = ["1h", "4h"]           # train_default per configs/config.yaml
MARKET       = "spot"
START_YEAR   = 2022                    # bound ingestion so the run stays Kaggle/Colab-sized; None = full history
SAMPLE       = False                   # True = last 500 bars (≈2-min smoke); False = full window
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 25
BATCH_SIZE    = 256
RESUME        = True                   # resume from latest checkpoint if present
print({"assets": ASSETS, "timeframes": TIMEFRAMES, "market": MARKET,
       "start_year": START_YEAR, "sample": SAMPLE, "phase1": PHASE1_EPOCHS,
       "phase2": PHASE2_EPOCHS, "batch_size": BATCH_SIZE, "resume": RESUME})
'''

INGEST = '''\
# --- ingest free data (Internet required) ---
# OHLCV bounded by START_YEAR to keep the run quick; set START_YEAR=None above
# for the deepest verified history. Context adapters are resilient (non-fatal).
import subprocess, sys
bin_cmd = [sys.executable, "-m", "src.ingest.binance_bulk",
           "--symbols", *ASSETS, "--market-types", MARKET, "--timeframes", *TIMEFRAMES]
if START_YEAR:
    bin_cmd += ["--start-year", str(START_YEAR)]
print(" ".join(bin_cmd))
subprocess.run(bin_cmd, check=True)
subprocess.run([sys.executable, "-m", "src.ingest.sentiment"], check=False)
subprocess.run([sys.executable, "-m", "src.ingest.coingecko"], check=False)
subprocess.run([sys.executable, "-m", "src.ingest.onchain", "--btc-only"], check=False)
subprocess.run([sys.executable, "-m", "src.ingest.derivatives", "--symbols", *ASSETS], check=False)
'''

FEATURES_LABELS = '''\
# --- build causal features + labels ---
import subprocess, sys
subprocess.run([sys.executable, "-m", "src.features.build_matrix",
                "--symbols", *ASSETS, "--timeframes", *TIMEFRAMES,
                "--market", MARKET, "--sample", "true" if SAMPLE else "false"], check=True)
subprocess.run([sys.executable, "-m", "src.labels.labeling",
                "--symbols", *ASSETS, "--timeframes", *TIMEFRAMES,
                "--market", MARKET, "--sample", "true" if SAMPLE else "false"], check=True)
'''

RESUME_CELL = '''\
# --- resume support: detect the latest run dir with checkpoints ---
from pathlib import Path
runs = sorted((Path("artifacts") / "runs").glob("*")) if (Path("artifacts") / "runs").exists() else []
resumable = [r for r in runs if (r / "phase1_best.keras").exists() or (r / "model.keras").exists()]
if RESUME and resumable:
    LATEST_RUN = resumable[-1].name
    print(f"resumable run found: {LATEST_RUN}")
    print("training will continue from its best checkpoint where supported.")
else:
    LATEST_RUN = None
    print("no prior checkpoint — training from scratch.")
'''

BUILD_DATASET = '''\
# --- build the sequence-windowed dataset (train-only scaler, purged splits) ---
import subprocess, sys
cmd = [sys.executable, "-m", "src.datasets.build_dataset",
       "--symbols", *ASSETS, "--timeframes", *TIMEFRAMES,
       "--market", MARKET, "--sample", "true" if SAMPLE else "false"]
print(" ".join(cmd))
subprocess.run(cmd, check=True)
'''

EVALUATE_BACKTEST = '''\
# --- honest evaluation + event-driven backtest vs baselines ---
import subprocess, sys
subprocess.run([sys.executable, "-m", "src.models.evaluate", "--latest",
                "--timeframe", TIMEFRAMES[0], "--sample", "true" if SAMPLE else "false"], check=False)
subprocess.run([sys.executable, "-m", "src.backtest.engine", "--latest",
                "--timeframes", *TIMEFRAMES, "--sample", "true" if SAMPLE else "false"], check=False)
print("Eval + backtest reports written under reports/. "
      "Check whether the model BEATS baselines before trusting it.")
'''

REGISTER = '''\
# --- register the trained run as a candidate; dry-run the promotion gates ---
import subprocess, sys
subprocess.run([sys.executable, "-m", "src.models.registry", "--sync"], check=False)
subprocess.run([sys.executable, "-m", "src.models.registry", "--list"], check=False)
subprocess.run([sys.executable, "-m", "src.models.promote", "--latest", "--dry-run"], check=False)
'''


def standard_train_cell(strategy_note: str) -> str:
    return f'''\
# --- train (uses MirroredStrategy automatically when 2+ GPUs present) ---
# {strategy_note}
import subprocess, sys
cmd = [sys.executable, "-m", "src.models.train",
       "--symbols", *ASSETS, "--timeframe", TIMEFRAMES[0],
       "--epochs", str(PHASE2_EPOCHS), "--batch-size", str(BATCH_SIZE),
       "--sample", "true" if SAMPLE else "false"]
print(" ".join(cmd))
subprocess.run(cmd, check=True)
'''


def plantguard_train_cell() -> str:
    return '''\
# --- PlantGuard-style two-phase training (calls the real src module) ---
# Phase 1: supervised head warmup. Phase 2: fine-tune last encoder blocks at lower LR.
# Produces: training curves, confusion matrices, classification report,
# prediction demo, reload test — all under artifacts/runs/<id>_plantguard/.
import subprocess, sys
cmd = [sys.executable, "-m", "src.models.train_like_plantguard",
       "--symbols", *ASSETS, "--timeframes", TIMEFRAMES[0],
       "--phase1-epochs", str(PHASE1_EPOCHS), "--phase2-epochs", str(PHASE2_EPOCHS),
       "--batch-size", str(BATCH_SIZE), "--sample", "true" if SAMPLE else "false"]
print(" ".join(cmd))
subprocess.run(cmd, check=True)
'''


# ---------------------------------------------------------------------------
# Environment-setup cells (platform-specific)
# ---------------------------------------------------------------------------

KAGGLE_SETUP = '''\
# --- Kaggle setup ---
# REQUIRES the notebook's "Internet" toggle = ON (Settings → Internet) so we can
# git-clone the public repo and pip-install. Select "GPU T4 x2" for dual-T4 →
# the project auto-detects and uses MirroredStrategy.
import os, sys, subprocess

REPO_URL = "https://github.com/mindees/crypto-ai.git"
BASE = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
REPO_DIR = os.path.join(BASE, "crypto-ai")
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, os.getcwd())   # make `import src...` work in this kernel
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=False)
print("cwd:", os.getcwd())
assert os.path.isdir("src"), "repo not cloned correctly — is Internet enabled?"
'''

KAGGLE_PULL_DATASET = '''\
# --- pull the latest prebuilt dataset from Kaggle (if configured) ---
# If you publish processed parquet as a Kaggle dataset, attach it to this
# notebook and symlink it into data/. Otherwise the build step below
# regenerates features/labels from raw (requires the raw parquet to be present).
import os
from pathlib import Path
KAGGLE_INPUT = Path("/kaggle/input")
if KAGGLE_INPUT.exists():
    print("Kaggle input datasets:", [p.name for p in KAGGLE_INPUT.iterdir()])
else:
    print("not on Kaggle or no input datasets attached.")
'''

KAGGLE_EXPORT = '''\
# --- export artifacts to /kaggle/working (downloadable / dataset-versionable) ---
import shutil
from pathlib import Path
out = Path("/kaggle/working/artifacts") if Path("/kaggle/working").exists() else Path("artifacts_export")
out.mkdir(parents=True, exist_ok=True)
src_runs = Path("artifacts/runs")
if src_runs.exists():
    latest = sorted(src_runs.glob("*"))[-1]
    shutil.copytree(latest, out / latest.name, dirs_exist_ok=True)
    print(f"exported {latest.name} -> {out}")
print("Also copy reports/ for the eval + backtest summaries.")
'''

COLAB_SETUP = '''\
# --- Colab setup ---
import sys, subprocess, os
# Optional: mount Google Drive for persistent checkpoints across disconnects.
try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    DRIVE_DIR = "/content/drive/MyDrive/mindees-crypto-ai"
    os.makedirs(DRIVE_DIR, exist_ok=True)
    print("Drive mounted at", DRIVE_DIR)
except Exception as e:
    DRIVE_DIR = None
    print("not on Colab / Drive not mounted:", e)

# Clone the public repo + install requirements.
REPO_URL = "https://github.com/mindees/crypto-ai.git"
BASE = "/content" if os.path.isdir("/content") else os.getcwd()
REPO_DIR = os.path.join(BASE, "crypto-ai")
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, os.getcwd())   # make `import src...` work in this kernel
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=False)
print("cwd:", os.getcwd())
assert os.path.isdir("src"), "repo not cloned correctly — check connectivity."
'''

COLAB_CHECKPOINT = '''\
# --- checkpoints to Drive (survive Colab disconnects) ---
import shutil
from pathlib import Path
if DRIVE_DIR:
    runs = Path("artifacts/runs")
    if runs.exists():
        for r in sorted(runs.glob("*")):
            dest = Path(DRIVE_DIR) / "artifacts" / "runs" / r.name
            shutil.copytree(r, dest, dirs_exist_ok=True)
        print("checkpoints synced to Drive.")
else:
    print("no Drive — checkpoints stay in the ephemeral runtime.")
'''


def build_standard(platform: str) -> dict:
    if platform == "kaggle":
        setup = [md("## 1. Kaggle setup"), code(KAGGLE_SETUP),
                 md("## 2. Pull dataset"), code(KAGGLE_PULL_DATASET)]
        export = [md("## 9. Export artifacts"), code(KAGGLE_EXPORT)]
        title = "# Kaggle — BTC/ETH multi-task training\n"
        note = "Kaggle dual-T4 -> MirroredStrategy is auto-selected by src/models/train.py."
    else:
        setup = [md("## 1. Colab setup (+ optional Drive mount)"), code(COLAB_SETUP)]
        export = [md("## 9. Checkpoints to Drive / export"), code(COLAB_CHECKPOINT)]
        title = "# Colab — BTC/ETH multi-task training\n"
        note = "Colab T4 -> single-GPU training; mount Drive to survive disconnects."

    cells = [
        md(title + "\n" + DISCLAIMER),
        *setup,
        md("## 3. GPU / environment check"), code(GPU_CHECK),
        md("## 4. Configuration"), code(CONFIG_BLOCK),
        md("## 5. Resume support"), code(RESUME_CELL),
        md("## 6. Ingest free data"), code(INGEST),
        md("## 7. Build features + labels"), code(FEATURES_LABELS),
        md("## 8. Build dataset"), code(BUILD_DATASET),
        md("## 9. Train"), code(standard_train_cell(note)),
        md("## 10. Evaluate + backtest + register"),
        code(EVALUATE_BACKTEST), code(REGISTER),
        *export,
    ]
    return notebook(cells)


def build_plantguard(platform: str) -> dict:
    if platform == "kaggle":
        setup = [md("## Kaggle setup"), code(KAGGLE_SETUP), code(KAGGLE_PULL_DATASET)]
        export = [md("## Save / export"), code(KAGGLE_EXPORT)]
        title = "# Kaggle — PlantGuard-style BTC/ETH training\n"
    else:
        setup = [md("## Colab setup"), code(COLAB_SETUP)]
        export = [md("## Save / export to Drive"), code(COLAB_CHECKPOINT)]
        title = "# Colab — PlantGuard-style BTC/ETH training\n"

    intro = (title + "\n" + DISCLAIMER +
             "\nAdapted from the Plant Guard image-classification workflow for "
             "**causal BTC/ETH time series**. The 16 sections below mirror that "
             "polished flow: GPU check -> config -> dataset validation -> label/"
             "sample visualization -> two-phase training -> curves -> confusion "
             "matrices -> classification report -> prediction demo -> save -> "
             "reload -> deployment smoke test. All logic lives in `src/` — the "
             "notebook only orchestrates.\n")

    cells = [
        md(intro),
        *setup,
        md("## 1. Imports & GPU check"), code(GPU_CHECK),
        md("## 2. Configuration"), code(CONFIG_BLOCK),
        md("## 3. Resume support"), code(RESUME_CELL),
        md("## 3b. Ingest free data + build features/labels\n"
           "Internet required. OHLCV is bounded by START_YEAR; context adapters "
           "(sentiment/coingecko/onchain/derivatives) are resilient."),
        code(INGEST), code(FEATURES_LABELS),
        md("## 4. Dataset validation + build\n"
           "Builds windows, prints per-combo row counts, feature counts, and "
           "the train/val/test split sizes."),
        code(BUILD_DATASET),
        md("## 5. Label & sample-window visualization\n"
           "The two-phase trainer emits label-distribution context and sample "
           "charts; see artifacts after the run. (Heavy chart code lives in "
           "`src/models/train_like_plantguard.py`.)"),
        md("## 6–9. Two-phase training (Phase 1 warmup + Phase 2 fine-tune)\n"
           "Produces training curves, confusion matrices (direction/regime/cycle), "
           "and the classification report."),
        code(plantguard_train_cell()),
        md("## 10–13. Curves, confusion matrices, classification report, prediction demo\n"
           "All saved under `artifacts/runs/<id>_plantguard/`:\n"
           "`training_curves.png`, `confusion_*.png`, `classification_report.json`, "
           "`prediction_demo.json`."),
        code('''\
from pathlib import Path
runs = sorted((Path("artifacts") / "runs").glob("*_plantguard"))
if runs:
    latest = runs[-1]
    print("artifacts in", latest.name)
    for f in sorted(latest.iterdir()):
        print("  ", f.name)
'''),
        md("## 14–15. Save final model + reload test\n"
           "The trainer already saves `model.keras` and runs a reload + sanity "
           "prediction. Confirm it printed `reload + sanity prediction: OK`."),
        md("## 16. Deployment smoke test"),
        code('''\
import subprocess, sys
subprocess.run([sys.executable, "-m", "src.models.predict", "--latest",
                "--symbols", *ASSETS, "--timeframes", *TIMEFRAMES], check=False)
subprocess.run([sys.executable, "-m", "src.serve.api", "--smoke-test"], check=False)
subprocess.run([sys.executable, "-m", "src.models.registry", "--sync"], check=False)
print("deployment smoke test complete.")
'''),
        md("## Evaluate + backtest + register"),
        code(EVALUATE_BACKTEST), code(REGISTER),
        *export,
    ]
    return notebook(cells)


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    targets = {
        "kaggle_train.ipynb": build_standard("kaggle"),
        "colab_train.ipynb": build_standard("colab"),
        "kaggle_train_plantguard_style.ipynb": build_plantguard("kaggle"),
        "colab_train_plantguard_style.ipynb": build_plantguard("colab"),
    }
    for name, nb in targets.items():
        path = NB_DIR / name
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
