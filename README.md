# quantic_ml_project
## Malware/Clearware Identifier Web App

This project hosts a lightweight Flask service that exposes batch inference endpoints and helpers for evaluating malware/clearware classifiers. It is intended to provide an API-backed demo for the pretrained pipeline under `artifacts/`.

## Running Locally
1. Create and enter a virtual environment (python3.10+) and install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   Use `requirements-dev.txt` when you want to explore the notebooks or evaluate models, as it pulls in the extra tooling (notebooks, dataset helpers, etc.) beyond the runtime-only dependencies in `requirements.txt`.
2. Make sure the pretrained pipeline artifact exists at `artifacts/model.joblib` (this is loaded by `app/services/inference_service.py`).
3. (Optional) Set `CORS_ORIGINS` to a comma-separated list of origins for `/api/*` routes; if omitted, no origins are allowed.
4. Start the server:
   ```
   python wsgi.py
   ```
   The Flask app listens on `http://127.0.0.1:5000` by default.

## Tests
- Run `pytest tests` from the project root to exercise the route and inference tests.

## Data
- The labeled dataset lives in `data/data.csv`. We pull it from the Brazilian Malware Dataset repository (`https://github.com/fabriciojoc/brazilian-malware-dataset`). Clone that project or download its CSV and copy it into `data/data.csv` before running training/evaluation.

## Model evaluation
- Install the extra tooling before running evaluation (e.g., `pip install -r requirements-dev.txt`).
- Ensure the labeled dataset sits at `data/data.csv`; the evaluation script loads this file to build features via `features.make_xy`.
- Execute the evaluation workflow:
  ```
  python ml/evaluate.py
  ```
  The script cross-validates each candidate model, prints the CV leaderboard, selects the best model, and reports hold-out test metrics (AUC, accuracy, confusion matrix).

## Training workflow
- Prepare `data/data.csv` and install dev dependencies (`pip install -r requirements-dev.txt`) if not already done.
- Run `python ml/train.py` to rebuild the preprocessing/XGBoost pipeline, tune via `RandomizedSearchCV`, and serialize the updated model/metadata artifacts.
