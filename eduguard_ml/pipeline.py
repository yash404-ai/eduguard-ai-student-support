"""Shared training / evaluation machinery used by every dataset experiment."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, log_loss, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from . import config


# --------------------------------------------------------------------------- splits
def three_way_split(y, groups=None, seed=config.SEED):
    """Stratified 70/15/15 split. `groups` keeps all rows of one student in one split.
    Uses 20 stratified folds (5% each): 3 -> test, 3 -> validation, 14 -> train."""
    y = np.asarray(y)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold = np.zeros(len(y), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        fold[idx] = k
    test = np.where(fold < 3)[0]
    val = np.where((fold >= 3) & (fold < 6))[0]
    train = np.where(fold >= 6)[0]
    return train, val, test


# --------------------------------------------------------------------------- models
def build_preprocessor(num_cols, cat_cols, scale):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:  # only needed for Logistic Regression
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [("num", Pipeline(num_steps), list(num_cols)),
         ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
          list(cat_cols))],
        verbose_feature_names_out=True)


def build_estimator(algo, params, n_classes, seed=config.SEED):
    if algo == "logistic_regression":
        return LogisticRegression(C=params["C"], max_iter=3000, class_weight="balanced", random_state=seed)
    if algo == "random_forest":
        return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=1, random_state=seed, **params)
    if algo == "xgboost":
        kw = dict(subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=1,
                  random_state=seed, eval_metric="mlogloss" if n_classes > 2 else "logloss", **params)
        return XGBClassifier(**kw)
    raise ValueError(algo)


def build_pipeline(algo, params, num_cols, cat_cols, n_classes):
    return Pipeline([("prep", build_preprocessor(num_cols, cat_cols, scale=(algo == "logistic_regression"))),
                     ("clf", build_estimator(algo, params, n_classes))])


def fit_pipeline(pipe, algo, X, y):
    """Class imbalance: balanced class weights (LR/RF) or balanced sample weights (XGBoost)."""
    if algo == "xgboost":
        pipe.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))
    else:
        pipe.fit(X, y)
    return pipe


# --------------------------------------------------------------------------- metrics
def evaluate(pipe, X, y, class_names):
    proba = pipe.predict_proba(X)
    pred = proba.argmax(1)
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    m = {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y, proba, labels=labels)),
        "per_class": {class_names[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                       "f1": float(f[i]), "support": int(s[i])} for i in labels},
        "confusion_matrix": {"labels": list(class_names),
                             "matrix": confusion_matrix(y, pred, labels=labels).tolist()},
    }
    if len(class_names) == 2:
        m["roc_auc"] = float(roc_auc_score(y, proba[:, 1]))
        m["pr_auc"] = float(average_precision_score(y, proba[:, 1]))
    else:
        m["roc_auc_ovr_macro"] = float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=labels))
    return m


# --------------------------------------------------------------------------- experiment runner
def run_experiment(*, dataset_id, title, X, y_labels, num_cols, cat_cols, target_description,
                   provenance, limitations, intended_use, groups=None, class_order=None,
                   out_dir=None, extra_meta=None, seed=config.SEED):
    """Train LR / RF / XGBoost with identical splits, tune on validation, test once."""
    out_dir = (out_dir or config.MODELS_DIR) / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    X = X[list(num_cols) + list(cat_cols)].reset_index(drop=True)
    classes = [str(c) for c in (class_order if class_order else sorted(set(map(str, y_labels))))]
    y = pd.Series(np.asarray(y_labels).astype(str)).map({c: i for i, c in enumerate(classes)}).to_numpy()
    assert not np.isnan(y.astype(float)).any(), "label outside class_order"
    y = y.astype(int)
    tr, va, te = three_way_split(y, groups, seed)
    Xtr, Xva, Xte = X.iloc[tr], X.iloc[va], X.iloc[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    version = f"{dataset_id}-v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    schema = feature_schema(num_cols, cat_cols, Xtr)
    bg = Xtr.sample(n=min(100, len(Xtr)), random_state=seed)  # SHAP background (train only)

    print(f"\n=== {dataset_id}: {len(X)} rows | train {len(tr)} / val {len(va)} / test {len(te)} | classes {classes}", flush=True)
    results = {}
    for algo in config.ALGORITHMS:
        tuning, best = [], None
        for params in config.GRIDS[algo]:
            t0 = time.time()
            pipe = fit_pipeline(build_pipeline(algo, params, num_cols, cat_cols, len(classes)), algo, Xtr, ytr)
            vm = evaluate(pipe, Xva, yva, classes)
            tuning.append({"params": params, "val_macro_f1": vm["macro_f1"], "val_accuracy": vm["accuracy"],
                           "val_log_loss": vm["log_loss"], "fit_seconds": round(time.time() - t0, 1)})
            if best is None or vm[config.SELECTION_METRIC] > best[1][config.SELECTION_METRIC]:
                best = (params, vm, pipe)
        params, vm, pipe = best
        tm = evaluate(pipe, Xte, yte, classes)  # test set touched ONCE per algorithm, after selection
        print(f"  {algo:20s} best={params}  val macroF1={vm['macro_f1']:.4f} | test acc={tm['accuracy']:.4f} macroF1={tm['macro_f1']:.4f}", flush=True)
        meta = {
            "model_id": f"{version}-{algo}", "version": version, "dataset_id": dataset_id, "title": title,
            "algorithm": algo, "target": target_description, "classes": classes,
            "class_mapping": {c: int(i) for i, c in enumerate(classes)},
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "seed": seed, "split": {"strategy": "stratified" + ("-grouped" if groups is not None else ""),
                                    **config.SPLIT, "n_train": len(tr), "n_validation": len(va), "n_test": len(te)},
            "selection_metric": f"validation {config.SELECTION_METRIC}",
            "hyperparameters": params, "tuning_log": tuning,
            "class_imbalance_handling": "balanced class/sample weights",
            "validation_metrics": vm, "test_metrics": tm,
            "feature_schema": schema, "provenance": provenance, "intended_use": intended_use,
            "limitations": limitations, "status": "validated",
            "explanation_note": "SHAP values describe how the model uses features; they are NOT causal explanations.",
            **(extra_meta or {}),
        }
        joblib.dump({"pipeline": pipe, "meta": meta, "background": bg}, out_dir / f"{algo}.joblib", compress=3)
        (out_dir / f"{algo}.json").write_text(json.dumps(meta, indent=2))
        results[algo] = meta

    best_algo = max(results, key=lambda a: results[a]["validation_metrics"][config.SELECTION_METRIC])
    report = {"dataset_id": dataset_id, "version": version, "title": title, "selected_default": best_algo,
              "models": {a: m for a, m in results.items()}}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    update_registry(dataset_id, version, best_algo, out_dir)
    return report


def feature_schema(num_cols, cat_cols, Xtr):
    s = {}
    for c in num_cols:
        s[c] = {"type": "numeric", "min": float(Xtr[c].min()), "max": float(Xtr[c].max()),
                "integer": bool(pd.api.types.is_integer_dtype(Xtr[c]))}
    for c in cat_cols:
        s[c] = {"type": "categorical", "allowed": sorted(map(str, Xtr[c].dropna().unique()))}
    return s


def update_registry(dataset_id, version, selected_algo, out_dir):
    reg_path = config.MODELS_DIR / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {"datasets": {}, "selected": {}}
    reg["datasets"][dataset_id] = {"version": version, "dir": str(out_dir.relative_to(config.MODELS_DIR)),
                                   "algorithms": config.ALGORITHMS}
    reg["selected"][dataset_id] = {"algorithm": selected_algo, "version": version,
                                   "reason": "best validation macro-F1 (auto)"}
    reg_path.write_text(json.dumps(reg, indent=2))
