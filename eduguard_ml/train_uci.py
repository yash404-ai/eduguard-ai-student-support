"""UCI 'Predict Students' Dropout and Academic Success' (file: data__1_.csv, 4424 rows).
Target: Dropout / Enrolled / Graduate. Prediction moment: end of 2nd semester of year 1."""
import pandas as pd
from . import config
from .pipeline import run_experiment

NOMINAL = ["Marital status", "Application mode", "Course", "Previous qualification", "Nacionality",
           "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]


def load(path=None):
    path = path or config.DATA_DIR / "data__1_.csv"
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def main():
    df = load()
    y = df.pop("Target")
    cat = NOMINAL
    for c in cat:
        df[c] = df[c].astype(str)  # numeric codes, but nominal -> one-hot
    num = [c for c in df.columns if c not in cat]
    return run_experiment(
        dataset_id="uci_dropout", title="UCI Predict Students' Dropout and Academic Success",
        X=df, y_labels=y, num_cols=num, cat_cols=cat, class_order=["Dropout", "Enrolled", "Graduate"],
        target_description="Target: Dropout / Enrolled / Graduate (dataset's own labels)",
        provenance={"source": "UCI ML Repository - Predict Students' Dropout and Academic Success (Realinho et al., 2021)",
                    "file": "data__1_.csv", "rows": int(len(df)), "missing_values": 0},
        intended_use="Early-warning triage for staff review at the end of the 1st year. Not for automatic decisions.",
        limitations=["Single Portuguese higher-education institution; may not transfer to Indian schools.",
                     "Includes 1st- and 2nd-semester academic results, so predictions are only valid after semester 2.",
                     "'Enrolled' is a still-in-progress outcome, not a final outcome.",
                     "Probabilities are not calibrated; do not treat them as exact risk percentages."])


if __name__ == "__main__":
    main()
