"""OULAD experiment: predict WITHDRAWAL during a course presentation.

Target (binary): Withdrawn (final_result == 'Withdrawn')  vs  Continued (Pass/Distinction/Fail).
Prediction time: day CUTOFF of the presentation (default 60). Only students still registered at that day are
scored (anyone who unregistered earlier is excluded, since their outcome is already known), and every feature uses
information with date <= CUTOFF. Nothing from final_result, later assessments, or post-cutoff clicks is used."""
import os
import numpy as np
import pandas as pd
from . import config
from .pipeline import run_experiment

CUTOFF = int(os.environ.get("OULAD_CUTOFF_DAY", 60))
CAT = ["code_module", "code_presentation", "gender", "region", "highest_education", "imd_band", "age_band", "disability"]
NUM = ["num_of_prev_attempts", "studied_credits", "date_registration", "clicks_total", "clicks_active_days",
       "clicks_distinct_sites", "clicks_last14", "clicks_prev14", "days_since_last_click", "clicks_before_start",
       "n_assess_due", "n_assess_submitted", "assess_submit_rate", "assess_mean_score", "assess_weighted_score",
       "n_assess_late"]


def vle_features(path, cutoff):
    agg = []
    for ch in pd.read_csv(path, usecols=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
                          dtype={"code_module": "category", "code_presentation": "category", "id_student": "int32",
                                 "id_site": "int32", "date": "int16", "sum_click": "int16"}, chunksize=2_000_000):
        ch = ch[ch["date"] <= cutoff]
        ch["code_module"] = ch["code_module"].astype(str); ch["code_presentation"] = ch["code_presentation"].astype(str)
        ch["pre"] = np.where(ch["date"] < 0, ch["sum_click"], 0)
        ch["l14"] = np.where(ch["date"] > cutoff - 14, ch["sum_click"], 0)
        ch["p14"] = np.where((ch["date"] > cutoff - 28) & (ch["date"] <= cutoff - 14), ch["sum_click"], 0)
        agg.append(ch)
    v = pd.concat(agg, ignore_index=True)
    k = ["code_module", "code_presentation", "id_student"]
    g = v.groupby(k)
    f = pd.DataFrame({"clicks_total": g["sum_click"].sum(), "clicks_active_days": g["date"].nunique(),
                      "clicks_distinct_sites": g["id_site"].nunique(), "clicks_last14": g["l14"].sum(),
                      "clicks_prev14": g["p14"].sum(), "clicks_before_start": g["pre"].sum(),
                      "last_click": g["date"].max()}).reset_index()
    f["days_since_last_click"] = cutoff - f["last_click"]
    return f.drop(columns="last_click")


def build_dataset(d=None, cutoff=CUTOFF):
    d = d or config.DATA_DIR
    si = pd.read_csv(d / "studentInfo.csv")
    reg = pd.read_csv(d / "studentRegistration.csv", na_values=["?"])
    k = ["code_module", "code_presentation", "id_student"]
    df = si.merge(reg, on=k, how="left")
    # keep only students registered by the cutoff day and NOT yet unregistered (outcome unknown at that time)
    still = df["date_unregistration"].isna() | (df["date_unregistration"] > cutoff)
    df = df[still].copy()
    df["withdrawn"] = np.where(df["final_result"] == "Withdrawn", "Withdrawn", "Continued")
    df["imd_band"] = df["imd_band"].replace("?", np.nan)
    df = df.merge(vle_features(d / "studentVle.csv", cutoff), on=k, how="left")
    for c in ["clicks_total", "clicks_active_days", "clicks_distinct_sites", "clicks_last14", "clicks_prev14", "clicks_before_start"]:
        df[c] = df[c].fillna(0)
    df["days_since_last_click"] = df["days_since_last_click"].fillna(cutoff + 1)  # never active
    # assessments due and submitted by cutoff
    a = pd.read_csv(d / "assessments.csv", na_values=["?"])
    sa = pd.read_csv(d / "studentAssessment.csv", na_values=["?"])
    a = a[(a["date"] <= cutoff)]
    due = a.groupby(["code_module", "code_presentation"]).agg(n_assess_due=("id_assessment", "count")).reset_index()
    s = sa.merge(a, on="id_assessment")
    s = s[s["date_submitted"] <= cutoff]
    s["late"] = (s["date_submitted"] > s["date"]).astype(int)
    s["w"] = s["score"] * s["weight"]
    # id_student comes from studentAssessment; code_module/presentation from assessments
    sgp = s.groupby(["code_module", "code_presentation", "id_student"]).agg(
        n_assess_submitted=("id_assessment", "count"), assess_mean_score=("score", "mean"),
        wsum=("w", "sum"), wt=("weight", "sum"), n_assess_late=("late", "sum")).reset_index()
    sgp["assess_weighted_score"] = sgp["wsum"] / sgp["wt"].replace(0, np.nan)
    df = df.merge(due, on=["code_module", "code_presentation"], how="left").merge(
        sgp.drop(columns=["wsum", "wt"]), on=k, how="left")
    df["n_assess_due"] = df["n_assess_due"].fillna(0)
    df["n_assess_submitted"] = df["n_assess_submitted"].fillna(0)
    df["n_assess_late"] = df["n_assess_late"].fillna(0)
    df["assess_submit_rate"] = np.where(df["n_assess_due"] > 0, df["n_assess_submitted"] / df["n_assess_due"].replace(0, 1), np.nan)
    return df


def main():
    df = build_dataset()
    groups = df["id_student"].to_numpy()  # same student in several presentations stays in one split
    print(f"OULAD cutoff day {CUTOFF}: {len(df)} still-registered student-presentations; withdrawn rate {(df.withdrawn=='Withdrawn').mean():.3f}")
    return run_experiment(
        dataset_id="oulad_withdrawal", title=f"OULAD withdrawal prediction at day {CUTOFF}",
        X=df, y_labels=df["withdrawn"], groups=groups, num_cols=NUM, cat_cols=CAT, class_order=["Continued", "Withdrawn"],
        target_description=f"Withdrawn (final_result = Withdrawn) vs Continued, for students still registered on day {CUTOFF}.",
        provenance={"source": "Open University Learning Analytics Dataset (Kuzilek et al., 2017)",
                    "files": ["studentInfo", "studentRegistration", "studentVle", "assessments", "studentAssessment"],
                    "prediction_day": CUTOFF, "rows": int(len(df)),
                    "leakage_controls": ["students who unregistered on/before cutoff removed", "VLE clicks with date > cutoff dropped",
                                         "only assessments due and submitted by cutoff", "final_result/score_total never used as a feature"]},
        intended_use=f"Identify students at risk of withdrawing from an online course around week {CUTOFF//7}; staff review only.",
        limitations=["Online distance-learning UK university; not equivalent to the UCI 3-class dropout or Indian school contexts.",
                     "Withdrawal can occur after cutoff for reasons the logs cannot show.",
                     "Same student may appear in several presentations (grouped in splits).",
                     "Probabilities are uncalibrated; the base rate differs per module and cutoff."],
        extra_meta={"prediction_day": CUTOFF})


if __name__ == "__main__":
    main()
