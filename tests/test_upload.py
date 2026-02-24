import io
import sys
from pathlib import Path

import pytest
from marshmallow import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_app
from app.controllers import upload_controller


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _post_file(client, filename, content):
    payload = content.encode() if isinstance(content, str) else content
    return client.post(
        "/api/upload",
        data={"file": (io.BytesIO(payload), filename)},
        content_type="multipart/form-data",
    )


def _patch_schema_to_accept(monkeypatch):
    def _load(payload):
        return {"rows": payload["rows"]}

    monkeypatch.setattr(upload_controller.batch_schema, "load", _load)


def test_upload_rejects_missing_file(client):
    resp = client.post("/api/upload")
    assert resp.status_code == 400
    assert resp.get_json()["message"] == "No file was uploaded"


def test_upload_rejects_non_csv(client):
    resp = _post_file(client, "sample.txt", "not,csv")
    assert resp.status_code == 415
    data = resp.get_json()
    assert data["message"] == "Invalid file type"
    assert data["errors"]["file"] == "Only csv files are supported"


def test_upload_controller_rejects_empty_file(client):
    resp = _post_file(client, "empty.csv", "")
    assert resp.status_code == 400
    assert resp.get_json()["message"] == "Uploaded file is empty"


def test_upload_validation_error(client, monkeypatch):
    def _load(payload):
        raise ValidationError({"rows": ["too many rows"]})

    monkeypatch.setattr(upload_controller.batch_schema, "load", _load)
    resp = _post_file(client, "payload.csv", "Label\n1\n")
    data = resp.get_json()
    assert resp.status_code == 422
    assert data["message"] == "Validation failed"
    assert "rows" in data["errors"]


def test_upload_prediction_failure(client, monkeypatch):
    _patch_schema_to_accept(monkeypatch)

    def _predict(_rows):
        raise RuntimeError("model offline")

    monkeypatch.setattr(upload_controller, "predict_rows", _predict)
    resp = _post_file(client, "data.csv", "Label\n0\n")
    data = resp.get_json()
    assert resp.status_code == 500
    assert data["message"] == "Prediction failed"
    assert "server" in data["errors"]


def test_upload_returns_evaluation(client, monkeypatch):
    _patch_schema_to_accept(monkeypatch)

    predictions = [
        {"row_index": 0, "prediction": 1, "probability": 0.9},
        {"row_index": 1, "prediction": 0, "probability": 0.1},
    ]

    monkeypatch.setattr(upload_controller, "predict_rows", lambda rows: predictions)
    captured = {}

    def _build_evaluation(y_true, preds):
        captured["y_true"] = list(y_true)
        captured["predictions"] = preds
        return {
            "available": True,
            "accuracy": 1.0,
            "confusion_matrix": {"tn": 1, "fp": 0, "fn": 0, "tp": 1},
            "auc": 1.0,
        }

    monkeypatch.setattr(upload_controller, "build_evaluation", _build_evaluation)
    resp = _post_file(client, "labeled.csv", "Label\n1\n0\n")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data["success"] is True
    assert data["data"]["results"] == predictions
    assert data["data"]["evaluation"]["available"] is True
    assert captured["y_true"] == [1, 0]
    assert captured["predictions"] == predictions


def test_upload_skips_evaluation_without_label(client, monkeypatch):
    _patch_schema_to_accept(monkeypatch)

    predictions = [{"row_index": 0, "prediction": 0, "probability": 0.25}]
    monkeypatch.setattr(upload_controller, "predict_rows", lambda rows: predictions)

    def _fail_build(_y, _p):
        raise AssertionError("Evaluation should not run without labels")

    monkeypatch.setattr(upload_controller, "build_evaluation", _fail_build)
    resp = _post_file(client, "nolabel.csv", "Size\n123\n")
    data = resp.get_json()

    assert resp.status_code == 200
    assert data["data"]["results"] == predictions
    assert data["data"]["evaluation"] == {
        "available": False,
        "message": "No label column found. Predictions returned without evaluation",
    }
