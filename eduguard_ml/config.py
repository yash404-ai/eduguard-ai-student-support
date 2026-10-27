"""Central configuration. Override paths with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("EDUGUARD_DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.environ.get("EDUGUARD_MODELS_DIR", ROOT / "models"))

SEED = 42
SPLIT = {"train": 0.70, "validation": 0.15, "test": 0.15}
ALGORITHMS = ["logistic_regression", "random_forest", "xgboost"]
SELECTION_METRIC = "macro_f1"          # computed on the VALIDATION set only

# Small grids on purpose: they run on a laptop / 1 CPU. Extend if you have more compute.
GRIDS = {
    "logistic_regression": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "random_forest": [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": l}
        for n in (200,) for d in (None, 12) for l in (1, 3)
    ],
    "xgboost": [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr}
        for n in (150, 300) for d in (3, 5) for lr in (0.05, 0.1)
    ],
}
