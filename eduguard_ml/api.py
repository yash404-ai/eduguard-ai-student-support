"""FastAPI service. Run: uvicorn eduguard_ml.api:app --reload   (or mount `router` inside your existing FastAPI app)."""
from pathlib import Path

from fastapi import APIRouter, Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import interventions, serving

router = APIRouter(prefix="/api/ml", tags=["ml"])


class SelectRequest(BaseModel):
    algorithm: str
    reason: str = "manual selection"


@router.get("/datasets")
def datasets():
    try: return serving.list_datasets()
    except serving.ModelNotFound as e: raise HTTPException(404, str(e))


@router.get("/datasets/{dataset_id}/report")
def report(dataset_id: str):
    """Everything the AI Model Lab needs: per-algorithm validation + test metrics, confusion matrices, features, provenance."""
    try: return serving.model_report(dataset_id)
    except serving.ModelNotFound as e: raise HTTPException(404, str(e))


@router.get("/datasets/{dataset_id}/schema")
def schema(dataset_id: str):
    try: return serving.load_artifact(dataset_id)["meta"]["feature_schema"]
    except serving.ModelNotFound as e: raise HTTPException(404, str(e))


@router.put("/datasets/{dataset_id}/selected")
def select(dataset_id: str, body: SelectRequest):
    try: return serving.select_model(dataset_id, body.algorithm, body.reason)
    except serving.ModelNotFound as e: raise HTTPException(404, str(e))


@router.post("/predict/{dataset_id}")
def predict(dataset_id: str, record: dict = Body(...), algorithm: str | None = Query(None), explain: bool = True):
    try: return serving.predict(dataset_id, record, algorithm, explain)
    except serving.ModelNotFound as e: raise HTTPException(404, str(e))
    except serving.InputValidationError as e: raise HTTPException(422, {"errors": e.errors})


@router.post("/interventions/{dataset_id}")
def intervention_suggestions(dataset_id: str, record: dict = Body(...), algorithm: str | None = Query(None)):
    """Prediction + supportive suggestions that staff must review before any ticket is created."""
    try:
        pred = serving.predict(dataset_id, record, algorithm, with_explanation=False)
        return {"prediction": pred, "intervention": interventions.suggest(dataset_id, record, pred)}
    except serving.ModelNotFound as e: raise HTTPException(404, str(e))
    except serving.InputValidationError as e: raise HTTPException(422, {"errors": e.errors})


app = FastAPI(title="EduGuard AI - ML service")
app.include_router(router)

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="frontend-assets")


@app.get("/", include_in_schema=False)
def frontend():
    """Serve the supplied EduGuard UI with its API integration."""
    page = FRONTEND_DIR / "EduGuard_AI.html"
    if not page.is_file():
        raise HTTPException(404, "Frontend file is missing")
    return FileResponse(page, media_type="text/html")


@app.get("/health")
def health(): return {"status": "ok"}
