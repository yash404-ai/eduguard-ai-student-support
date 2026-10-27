"""UCI xAPI-Edu-Data (480 rows). Target = Class (L/M/H) = ACADEMIC PERFORMANCE band, NOT dropout."""
import pandas as pd
from . import config
from .pipeline import run_experiment


def load(path=None):
    return pd.read_csv(path or config.DATA_DIR / "xAPI-Edu-Data.csv")


def main():
    df = load()
    y = df.pop("Class")
    cat = ["gender", "NationalITy", "PlaceofBirth", "StageID", "GradeID", "SectionID", "Topic", "Semester",
           "Relation", "ParentAnsweringSurvey", "ParentschoolSatisfaction", "StudentAbsenceDays"]
    num = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]
    return run_experiment(
        dataset_id="xapi_performance", title="xAPI-Edu-Data academic performance",
        X=df, y_labels=y, num_cols=num, cat_cols=cat, class_order=["L", "M", "H"],
        target_description="Class: L (low), M (medium), H (high) academic performance band. This is NOT a dropout label.",
        provenance={"source": "UCI / Kaggle xAPI-Edu-Data (Amrieh et al., 2016), Kalboard 360 LMS", "file": "xAPI-Edu-Data.csv",
                    "rows": int(len(df)), "missing_values": 0},
        intended_use="Predict performance band from engagement behaviour. Do not present as dropout prediction.",
        limitations=["Only 480 rows: test set is ~72 students, so metrics are noisy.",
                     "Performance band is derived from marks, not dropout.",
                     "Engagement counts may be partly a consequence of performance (possible reverse causality).",
                     "Single LMS, mostly Middle-Eastern students."])


if __name__ == "__main__":
    main()
