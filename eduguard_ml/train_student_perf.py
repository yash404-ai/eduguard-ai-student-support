"""UCI Student Performance (student-mat.csv + student-por.csv), Cortez & Silva 2008.
Target: fail (final grade G3 < 10) vs pass (G3 >= 10) on the 0-20 Portuguese scale.
Early-warning design: G1, G2 (earlier period grades) and G3 are EXCLUDED, so the model uses only background and behaviour."""
import pandas as pd
from . import config
from .pipeline import run_experiment

KEY = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]  # from student-merge.R
CAT = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup",
       "paid", "activities", "nursery", "higher", "internet", "romantic", "subject"]
NUM = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]


def load(d=None):
    d = d or config.DATA_DIR
    mat = pd.read_csv(d / "student-mat.csv", sep=";"); mat["subject"] = "math"
    por = pd.read_csv(d / "student-por.csv", sep=";"); por["subject"] = "portuguese"
    return pd.concat([mat, por], ignore_index=True)


def main():
    df = load()
    # ~382 students appear in both files (see student-merge.R). Group them so one student never lands in two splits.
    groups = df[KEY].astype(str).agg("|".join, axis=1).factorize()[0]
    y = (df["G3"] >= 10).map({True: "pass", False: "fail"})
    return run_experiment(
        dataset_id="student_performance", title="Student Performance (Math + Portuguese) - fail risk",
        X=df, y_labels=y, groups=groups, num_cols=NUM, cat_cols=CAT, class_order=["fail", "pass"],
        target_description="fail = final grade G3 < 10 (of 20); pass = G3 >= 10. Academic failure, NOT dropout.",
        provenance={"source": "UCI Student Performance (Cortez & Silva, 2008), Portuguese secondary schools",
                    "files": ["student-mat.csv", "student-por.csv"], "rows": int(len(df)),
                    "duplicate_students_grouped": int(len(df) - len(set(groups))),
                    "excluded_for_leakage": ["G1", "G2", "G3"]},
        intended_use="Early-warning support from background/behaviour before any grades exist.",
        limitations=["~1,044 rows from two schools in Portugal; metrics are noisy.",
                     "Grades G1/G2 are deliberately excluded, which lowers accuracy but avoids near-duplicate-of-target features.",
                     "Contains sensitive attributes (alcohol use, romantic status, parents' education/jobs); use only to direct support, never to penalise students.",
                     "Pass/fail is not dropout."])


if __name__ == "__main__":
    main()
