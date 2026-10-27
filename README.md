# EduGuard AI — Student Support Project

EduGuard is an educational decision-support prototype intended to help students and educators notice academic changes and respond with supportive next steps. It combines a web interface with a Python/FastAPI service and bundled machine-learning artifacts.

> **Demo and safety note:** This project is not a validated school-management system. Its current Student Performance page shows a snapshot of marks and attendance entered by the user. It does **not** generate an ML prediction or risk label from those short-form inputs. The trained models exposed by the backend require dataset-specific feature sets that are not the same as that page's simplified form. Do not use this demo to punish, exclude, rank, or deny services to a student.

## What the project is for

Students can find it difficult to tell whether their academic performance is improving, steady, or declining. EduGuard demonstrates how a school-support tool could present academic information clearly and help a teacher or student decide what to discuss next.

The intended social benefit is **earlier, more understandable support**: surface a change in recorded marks, prompt a useful conversation about missed learning, and suggest practical follow-up such as reviewing difficult topics or checking progress with a teacher. This is a project goal, not a measured outcome; the prototype has not been evaluated in a real school or shown to improve student outcomes.

## What is implemented

- A browser-based EduGuard interface using the project's existing HTML/CSS design.
- A focused Student Performance page with optional student reference and age, first/latest entered marks, attendance, and optional fee-status display.
- A responsive chart of the two marks supplied, a marks-earned-versus-remaining graphic, a results table, and suggestions grounded only in those entered values.
- Validation for mark and attendance ranges, and a Clear action. The snapshot values stay in the current page and are not sent to the model API by this page enhancement.
- A FastAPI ML service with dataset, schema, model-report, model-selection, prediction, and staff-review intervention endpoints.
- Bundled Logistic Regression, Random Forest, and XGBoost artifacts for four datasets. Each model report contains its target, evaluation metrics, feature schema, provenance, and limitations.

The backend APIs can make dataset-specific predictions when given the **complete, valid feature record** required by that dataset's schema. The simplified Student Performance page deliberately does not call those endpoints with partial or invented values.

## What is different about EduGuard

1. **Supportive rather than punitive:** the goal is to help people decide where to offer support, not automate a disciplinary decision.
2. **Shows evidence, not invented history:** the chart contains only the two values entered; it does not create missing semesters, a class average, or a future grade.
3. **Makes limitations visible:** it distinguishes a user-entered snapshot from an ML prediction and explains when a reliable model result is unavailable.
4. **Keeps dataset targets separate:** the backend includes datasets with different populations and outcomes. A dropout label, a day-60 withdrawal label, and a school pass/fail label are not interchangeable.
5. **Stores context with the artifacts:** schema, data provenance, validation/test metrics, and limitations are included with each model report.

These are design choices in a prototype; they do not establish that the models are unbiased, accurate for a particular school, or suitable for consequential decisions.

## Important model compatibility limitation

The bundled `student_performance` model predicts a historical pass/fail category from **31 background and behavior inputs**. It intentionally excludes semester marks and does not accept attendance percentage or fee-payment status. The project also has no student-ID lookup, student-record database, or class roster.

Consequently, the current simplified Prediction page provides a transparent academic snapshot and practical prompts, but does **not** produce a trained-model prediction, a Low/Medium/High risk classification, a student lookup, or a class-average comparison. To add one of those capabilities responsibly, the project would need a validated model trained on the permitted input fields or an authorized data source, plus appropriate school-defined criteria. No such source or compatible model is included here.

Fee status is displayed only as optional context; this page does not use it to infer ability. Predictions from the backend's other dataset-specific models should only be interpreted within their documented population and target.

## Backend datasets and targets

The model metadata and reports under `models/` contain the exact feature schemas, provenance, evaluation metrics, and limitations for each bundled artifact.

| Dataset ID | Documented target | Important context |
|---|---|---|
| `uci_dropout` | Dropout / Enrolled / Graduate | UCI higher-education outcomes; not a school pass/fail target. |
| `xapi_performance` | Low / Medium / High academic-performance band | xAPI educational dataset; these labels are not dropout categories. |
| `student_performance` | Fail if final grade G3 < 10/20; otherwise pass | Portuguese secondary-school data. G1/G2/G3 are excluded from model features. This is academic failure, not dropout. |
| `oulad_withdrawal` | Withdrawn vs. continued at the documented day-60 prediction point | Open University Learning Analytics Dataset (OULAD); a different population and target. |

The datasets are not directly comparable. They describe different populations, features, and outcomes. Do not infer that reported test metrics apply to a new school or cohort. Raw source CSVs are not included in this repository; the pre-trained artifacts are included for local inference through their dataset-specific APIs.

## Project structure

```text
eduguard_ml/
├── README.md
├── requirements.txt
├── eduguard_ml/              # FastAPI API, serving, validation, and interventions
├── frontend/
│   ├── EduGuard_AI.html      # Original site with the Prediction script reference
│   └── student_prediction.js # Prediction-page-specific UI logic
├── models/                   # Registry, trained artifacts, schemas, reports
├── scripts/                  # Optional training entry points
├── tests/                    # API, validation, model, and page regression tests
└── .gitignore
```

## Run locally on Windows

Use Python 3.12 (recommended for the bundled dependencies). Open PowerShell in the repository root—the folder containing `requirements.txt`—and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn eduguard_ml.api:app --host 127.0.0.1 --port 8000
```

Then open these addresses in Chrome:

- **EduGuard:** http://127.0.0.1:8000/#predict
- **API health:** http://127.0.0.1:8000/health
- **Interactive API documentation:** http://127.0.0.1:8000/docs

Keep the PowerShell window open while using the site. Stop the local server with `Ctrl+C`.

If PowerShell blocks virtual-environment activation, skip activation and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn eduguard_ml.api:app --host 127.0.0.1 --port 8000
```

### Run locally on macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn eduguard_ml.api:app --host 127.0.0.1 --port 8000
```

## API overview

- `GET /api/ml/datasets` — list registered datasets and selected models.
- `GET /api/ml/datasets/{dataset_id}/schema` — retrieve required feature names, ranges, and categories.
- `GET /api/ml/datasets/{dataset_id}/report` — retrieve model metrics, target, provenance, and limitations.
- `PUT /api/ml/datasets/{dataset_id}/selected` — select a registered algorithm for a dataset.
- `POST /api/ml/predict/{dataset_id}` — predict from a complete schema-valid record.
- `POST /api/ml/interventions/{dataset_id}` — return a prediction and supportive suggestions marked for staff review.

Prediction endpoints validate their records against the bundled model schemas. The intervention response is advisory and should be reviewed by staff before use.

## Tests

From the repository root:

```powershell
python -m pytest -q
```

The optional retraining test is skipped when its raw source CSV is not present. Packaged model artifacts remain available for inference without those CSVs.

## Retraining (optional)

Training requires the source CSVs expected by the scripts. Place them in the configured data directory (default `data/`) and consult `eduguard_ml/config.py` and the relevant `eduguard_ml/train_*.py` loader for exact filenames and requirements. Raw datasets are intentionally excluded from Git. Training writes new artifacts and updates the model registry; back up the current `models/` directory before retraining if you need to preserve it.

## Publish this folder to GitHub

1. Extract `EduGuard_AI_GitHub_Ready.zip` and open a terminal in the extracted `EduGuard_AI_GitHub` folder.
2. On GitHub, create an empty repository named `eduguard-ai-student-support`. Choose its visibility deliberately: a public repository can be viewed by anyone. Do not have GitHub add a second README or license during creation.
3. In PowerShell, run the following commands, replacing `<YOUR_GITHUB_USERNAME>` with your account name. GitHub will handle authentication through its normal browser or credential flow; do not put a password or access token in these commands.

```powershell
git init -b main
git add .
git commit -m "Add EduGuard student support project"
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/eduguard-ai-student-support.git
git push -u origin main
```

Before making a repository public, review the file list and confirm there are no personal student records, credentials, or raw datasets. This package includes bundled model artifacts and their metadata, but excludes local Python environments and caches.

## Privacy, security, and responsible use

- This is a local educational prototype, not a production service. The UI is not a substitute for server-side authentication, authorization, secure storage, or school data-governance review.
- Do not enter real student personal data into a public demo or expose the API to the internet.
- The simplified Prediction page does not send its entered values to the ML API or save them. The API does not provide student-record lookup or roster services.
- Treat model outputs as uncertain associations, not causes or facts about an individual. A human educator should review support decisions.
- Before any real deployment, conduct consent, privacy, security, subgroup-performance, and fairness reviews, and confirm legal authority to use the data.

## Data and model provenance

Dataset provenance and model evaluation details are recorded in the corresponding JSON files under `models/`. Preserve those records when sharing or presenting this project. If an artifact is replaced or retrained, update its provenance, metrics, schema, and limitations, and follow the dataset provider's attribution and use conditions.

## License

No project license has been selected. Until the project owner adds one, GitHub's default copyright rules apply to the project code; the bundled datasets and model artifacts may also have their own terms. Check those terms before redistributing or making the repository public.

## Development transparency

AI assistance was used to help integrate the Prediction page and prepare this repository documentation. Review the code, be prepared to explain the project, and follow your teacher's rules for acknowledging assistance.
