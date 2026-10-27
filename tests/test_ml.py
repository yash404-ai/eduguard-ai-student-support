import copy, json, os
import pytest
from fastapi.testclient import TestClient
from eduguard_ml import config, serving, train_uci, train_xapi, pipeline
from eduguard_ml.api import app

client = TestClient(app)
DS = "xapi_performance"


def sample_row():
    schema = serving.load_artifact(DS)["meta"]["feature_schema"]
    row = {}
    for name, spec in schema.items():
        if spec["type"] == "categorical":
            row[name] = spec["allowed"][0]
        elif spec.get("integer"):
            row[name] = int(spec["min"])
        else:
            row[name] = spec["min"]
    return row


def test_registry_has_all_datasets():
    ids = {d["dataset_id"] for d in serving.list_datasets()}
    assert {"uci_dropout", "xapi_performance", "student_performance", "oulad_withdrawal"} <= ids


def test_prediction_page_and_automatic_model_context_are_served():
    page = client.get("/")
    assert page.status_code == 200
    assert 'src="/assets/student_prediction.js?v=2"' in page.text
    assert "Spot dropout risk early." in page.text  # unrelated original page retained
    assert "AI Model Lab" not in page.text
    script = client.get("/assets/student_prediction.js")
    assert script.status_code == 200
    assert 'student-snapshot-form' in script.text
    assert 'const DATASET_ID = "student_performance"' in script.text
    assert 'id="eg-dataset"' not in script.text
    assert 'mother' not in script.text.lower() and 'father' not in script.text.lower()
    report = client.get("/api/ml/datasets/student_performance/report")
    assert report.status_code == 200
    assert len(report.json()["models"][report.json()["selected"]["algorithm"]]["feature_schema"]) == 31


@pytest.mark.parametrize("algo", config.ALGORITHMS)
def test_artifact_loads_and_predicts(algo):
    r = serving.predict(DS, sample_row(), algo)
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-3
    assert r["algorithm"] == algo and "top_features" in r["explanation"]


def test_reported_metrics_are_real_and_separate():
    rep = serving.model_report("uci_dropout")
    for m in rep["models"].values():
        assert m["validation_metrics"] != m["test_metrics"]
        assert 0 < m["test_metrics"]["accuracy"] < 1 and "confusion_matrix" in m["test_metrics"]


def test_split_is_stratified_disjoint_and_grouped():
    import numpy as np
    y = np.array([0, 1] * 200); g = np.repeat(np.arange(200), 2)
    tr, va, te = pipeline.three_way_split(y, g)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    assert not ({g[i] for i in tr} & {g[i] for i in te})


def test_missing_field_rejected_not_invented():
    row = sample_row(); row.pop("raisedhands")
    with pytest.raises(serving.InputValidationError) as e:
        serving.predict(DS, row)
    assert "raisedhands" in str(e.value)


def test_out_of_range_and_bad_category_rejected():
    row = sample_row(); row["raisedhands"] = 5000; row["gender"] = "X"
    with pytest.raises(serving.InputValidationError) as e:
        serving.predict(DS, row)
    assert "raisedhands" in str(e.value) and "gender" in str(e.value)


def test_api_predict_ok_and_422():
    ok = client.post(f"/api/ml/predict/{DS}", json=sample_row())
    assert ok.status_code == 200 and ok.json()["predicted_class"] in {"L", "M", "H"}
    bad = client.post(f"/api/ml/predict/{DS}", json={"gender": "M"})
    assert bad.status_code == 422
    unknown_algorithm = client.post(f"/api/ml/predict/{DS}?algorithm=../../outside", json=sample_row())
    assert unknown_algorithm.status_code == 404
    assert client.post("/api/ml/predict/nope", json={}).status_code == 404


def test_select_model_validated_only():
    original = serving.registry()["selected"][DS]["algorithm"]
    try:
        assert serving.select_model(DS, "random_forest")["algorithm"] == "random_forest"
        assert serving.predict(DS, sample_row(), with_explanation=False)["algorithm"] == "random_forest"
        with pytest.raises(serving.ModelNotFound):
            serving.select_model(DS, "does_not_exist")
    finally:
        serving.select_model(DS, original, "restore")


def test_intervention_endpoint_requires_review():
    r = client.post(f"/api/ml/interventions/{DS}", json=sample_row())
    assert r.status_code == 200 and r.json()["intervention"]["status"] == "pending_staff_review"


def test_training_pipeline_runs_end_to_end(tmp_path, monkeypatch):
    """Retrains xAPI into a temp dir with a tiny grid to prove the training code itself works."""
    if not (config.DATA_DIR / "xAPI-Edu-Data.csv").exists():
        pytest.skip("Raw xAPI training CSV was not included; bundled models remain available for inference")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "GRIDS", {"logistic_regression": [{"C": 1.0}], "random_forest": [{"n_estimators": 20, "max_depth": 5, "min_samples_leaf": 1}],
                                           "xgboost": [{"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1}]})
    rep = train_xapi.main()
    assert set(rep["models"]) == set(config.ALGORITHMS)
    assert (tmp_path / "xapi_performance" / "xgboost.joblib").exists()
