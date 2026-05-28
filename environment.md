# Environment notes

This document records environment assumptions and observed deviations from the
project spec. Update it whenever the install path, supported Python version,
or platform-specific setup changes.

## Python version

The spec requests **Python 3.11**. The current development host has only
**Python 3.13.7**, and TensorFlow 2.21.0 ships a `cp313` wheel for Windows, so
the project pins **TensorFlow 2.21.0** and runs on Python 3.13.

If Python 3.11 becomes available later, recreate the venv with 3.11 and pin
TF 2.18 instead — both pins are known-good combinations.

## Virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Or use the venv interpreter directly without activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.utils.hardware
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

## GPU

| Environment | GPU | Notes |
| --- | --- | --- |
| Local (Windows) | usually none | TF 2.11+ on Windows runs CPU only; use WSL2 for GPU |
| Kaggle | 2× T4 (when enabled) | the project auto-detects and uses `MirroredStrategy` |
| Colab | T4 (when enabled) | single-GPU path |
| GitHub Actions | none | CPU only; never trains heavy models |

## TA-Lib (optional)

The indicator layer abstracts over TA-Lib and `pandas-ta-classic`. TA-Lib
needs a native C library on the host, which is often painful on Windows. The
spec explicitly tolerates this — if `import talib` fails the codebase falls
back to `pandas-ta-classic` and, if that also fails, to minimal manual
implementations.

To try installing TA-Lib on Windows:

```powershell
# (one option) install a prebuilt wheel matching your Python ABI, then:
pip install -r requirements-optional.txt
```

If it fails, leave it uninstalled and continue — Phase 0 does not need it.

## Optional API keys (none required by Phase 0)

These belong in `.env` (which is gitignored) once Phase 1+ uses them:

```text
KAGGLE_USERNAME
KAGGLE_KEY
FRED_API_KEY        # only if sources.enable_fred = true
ETHERSCAN_API_KEY   # only if sources.enable_etherscan = true
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
DISCORD_WEBHOOK_URL
SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS / SMTP_FROM / SMTP_TO
```

## Phase 0 gate commands

```powershell
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
python -m src.utils.hardware
pytest tests/ -q
```
