"""Production prediction code (no training logic here). Loads validated artifacts from MODELS_DIR."""
from __future__ import annotations

import json
import math
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from . import config


class ModelNotFound(Exception): ...
class InputValidationError(ValueError):
    def __init__(self, errors): super().__init__("; ".join(errors)); self.errors = errors


def _reg_path(): return config.MODELS_DIR / "registry.json"


def registry():
    if not _reg_path().exists():
        raise ModelNotFound("No trained models found. Run: python scripts/train_all.py")
    return json.loads(_reg_path().read_text())


def list_datasets():
    reg = registry(); out = []
    for ds, info in reg["datasets"].items():
        rep = json.loads((config.MODELS_DIR / info["dir"] / "report.json").read_text())
        out.append({"dataset_id": ds, "title": rep["title"], "version": rep["version"],
                    "selected": reg["selected"].get(ds), "models": list(rep["models"])})
    return out


def model_report(dataset_id):
    reg = registry()
    if dataset_id not in reg["datasets"]:
        raise ModelNotFound(f"Unknown dataset '{dataset_id}'")
    rep = json.loads((config.MODELS_DIR / reg["datasets"][dataset_id]["dir"] / "report.json").read_text())
    rep["selected"] = reg["selected"].get(dataset_id)
    return rep


def select_model(dataset_id, algorithm, reason="manual selection"):
    """Explicitly choose which validated algorithm the API serves for a dataset."""
    reg = registry()
    if dataset_id not in reg["datasets"]:
        raise ModelNotFound(f"Unknown dataset '{dataset_id}'")
    if algorithm not in reg["datasets"][dataset_id]["algorithms"]:
        raise ModelNotFound(f"Unknown algorithm '{algorithm}'")
    path = config.MODELS_DIR / reg["datasets"][dataset_id]["dir"] / f"{algorithm}.joblib"
    if not path.exists():
        raise ModelNotFound(f"Artifact missing: {path.name}")
    if json.loads(path.with_suffix(".json").read_text()).get("status") != "validated":
        raise ModelNotFound("Only validated models can be selected")
    reg["selected"][dataset_id] = {"algorithm": algorithm, "version": reg["datasets"][dataset_id]["version"], "reason": reason}
    _reg_path().write_text(json.dumps(reg, indent=2))
    return reg["selected"][dataset_id]


@lru_cache(maxsize=16)
def _load(path: str):
    return joblib.load(path)


def load_artifact(dataset_id, algorithm=None):
    reg = registry()
    if dataset_id not in reg["datasets"]:
        raise ModelNotFound(f"Unknown dataset '{dataset_id}'")
    algorithm = algorithm or reg["selected"][dataset_id]["algorithm"]
    path = config.MODELS_DIR / reg["datasets"][dataset_id]["dir"] / f"{algorithm}.joblib"
    if not path.exists():
        raise ModelNotFound(f"Model artifact not found for {dataset_id}/{algorithm}")
    return _load(str(path))


def validate_input(record: dict, schema: dict) -> pd.DataFrame:
    """Strict: every feature required, no invented defaults, no unknown fields, values checked against training schema."""
    errors = []
    unknown = sorted(set(record) - set(schema))
    if unknown: errors.append(f"unknown fields: {unknown}")
    missing = sorted(set(schema) - set(record))
    if missing: errors.append(f"missing required fields: {missing}")
    row = {}
    for name, spec in schema.items():
        if name not in record: continue
        v = record[name]
        if v is None or (isinstance(v, float) and math.isnan(v)) or v == "":
            errors.append(f"{name}: value is empty"); continue
        if spec["type"] == "numeric":
            if isinstance(v, bool) or not isinstance(v, (int, float, str)):
                errors.append(f"{name}: must be a number"); continue
            try: x = float(v)
            except ValueError: errors.append(f"{name}: must be a number, got {v!r}"); continue
            if not math.isfinite(x): errors.append(f"{name}: must be finite"); continue
            if x < spec["min"] or x > spec["max"]:
                errors.append(f"{name}: {x} is outside the range seen in training [{spec['min']}, {spec['max']}]"); continue
            row[name] = x
        else:
            sv = str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)  # 2.0 -> "2"
            if sv not in spec["allowed"]:
                errors.append(f"{name}: {v!r} not one of {spec['allowed']}"); continue
            row[name] = sv
    if errors: raise InputValidationError(errors)
    return pd.DataFrame([row])[list(schema)]


def _orig_feature(transformed_name, cat_cols):
    kind, rest = transformed_name.split("__", 1)
    if kind == "num": return rest
    return max((c for c in cat_cols if rest.startswith(c + "_")), key=len)


def explain(art, X: pd.DataFrame, class_index: int, top_k=8):
    """Real SHAP values for the predicted class, summed back from one-hot columns to original features."""
    import shap
    pipe, bg = art["pipeline"], art["background"]
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    Z = prep.transform(X); B = prep.transform(bg)
    names = list(prep.get_feature_names_out())
    cats = [c for c, s in art["meta"]["feature_schema"].items() if s["type"] == "categorical"]
    algo = art["meta"]["algorithm"]
    explainer = shap.LinearExplainer(clf, B) if algo == "logistic_regression" else shap.TreeExplainer(clf, B, model_output="probability" if algo == "random_forest" else "raw")
    sv = explainer.shap_values(Z)
    sv = np.asarray(sv) if not isinstance(sv, list) else np.stack(sv, axis=-1)
    if sv.ndim == 3:   # (samples, features, classes)
        vals = sv[0, :, class_index]
    elif sv.ndim == 2 and len(art["meta"]["classes"]) == 2:  # binary: values refer to positive class
        vals = sv[0] if class_index == 1 else -sv[0]
    else:
        vals = sv[0]
    agg = {}
    for n, v in zip(names, vals):
        f = _orig_feature(n, cats); agg[f] = agg.get(f, 0.0) + float(v)
    top = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:top_k]
    return [{"feature": f, "value": X.iloc[0][f].item() if hasattr(X.iloc[0][f], "item") else X.iloc[0][f],
             "shap_value": round(v, 5), "direction": "raises" if v > 0 else "lowers"} for f, v in top]


def predict(dataset_id, record: dict, algorithm=None, with_explanation=True):
    art = load_artifact(dataset_id, algorithm)
    meta = art["meta"]
    X = validate_input(record, meta["feature_schema"])
    proba = art["pipeline"].predict_proba(X)[0]
    idx = int(np.argmax(proba))
    out = {"dataset_id": dataset_id, "model_id": meta["model_id"], "algorithm": meta["algorithm"],
           "model_version": meta["version"], "target": meta["target"],
           "predicted_class": meta["classes"][idx],
           "probabilities": {c: round(float(p), 4) for c, p in zip(meta["classes"], proba)},
           "notice": "Model output supports staff review only. It is not a diagnosis and must not be used alone to deny any student support or opportunity. Probabilities are uncalibrated estimates."}
    if with_explanation:
        try:
            out["explanation"] = {"method": "SHAP (" + ("LinearExplainer" if meta["algorithm"] == "logistic_regression" else "TreeExplainer") + ") for the predicted class",
                                  "top_features": explain(art, X, idx),
                                  "note": meta["explanation_note"]}
        except Exception as e:  # never fail a prediction because the explanation failed
            out["explanation"] = {"error": f"explanation unavailable ({type(e).__name__})"}
    return out
