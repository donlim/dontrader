import os

# repo root = .../trading_bot/..
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Allow overriding to an external disk via env vars
LOG_ROOT        = os.environ.get("TB_LOG_ROOT",        os.path.join(REPO_ROOT, "data", "logs"))
ARTIFACTS_ROOT  = os.environ.get("TB_ARTIFACTS_ROOT",  os.path.join(REPO_ROOT, "data", "artifacts"))
DATASETS_DIR    = os.path.join(ARTIFACTS_ROOT, "datasets")
WEIGHTS_DIR     = os.path.join(ARTIFACTS_ROOT, "weights")

# Ensure the directories exist
for _d in (LOG_ROOT, ARTIFACTS_ROOT, DATASETS_DIR, WEIGHTS_DIR):
    os.makedirs(_d, exist_ok=True)