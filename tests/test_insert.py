import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_app


BASE_ROW = {
    "BaseOfCode": 1024,
    "BaseOfData": 2048,
    "Characteristics": 4096,
    "DllCharacteristics": 256,
    "Entropy": 5.5,
    "FileAlignment": 512,
    "FirstSeenDate": "2024-01-01",
    "Identify": "sample",
    "ImageBase": 4194304,
    "ImportedDlls": "kernel32.dll",
    "ImportedSymbols": "CreateFileW CloseHandle",
    "Machine": 34404,
    "Magic": None,
    "NumberOfRvaAndSizes": 6,
    "NumberOfSections": 4,
    "NumberOfSymbols": 0,
    "PE_TYPE": None,
    "PointerToSymbolTable": 0,
    "SHA1": None,
    "Size": 8192,
    "SizeOfCode": 4096,
    "SizeOfHeaders": 1024,
    "SizeOfImage": 8192,
    "SizeOfInitializedData": 1024,
    "SizeOfOptionalHeader": None,
    "SizeOfUninitializedData": 512,
    "TimeDateStamp": 1660000000,
    "Label": None,
}


@pytest.fixture
def client():
    app = create_app()
    app.testing = True
    return app.test_client()


def make_row(**overrides):
    row = BASE_ROW.copy()
    row.update(overrides)
    return row


def test_insert_success_returns_predictions(client, monkeypatch):
    expected = [
        {"row_index": 0, "prediction": 0},
        {"row_index": 1, "prediction": 1},
    ]

    def fake_predict(rows):
        return expected

    monkeypatch.setattr("app.controllers.insert_controller.predict_rows", fake_predict)

    response = client.post("/api/insert", json={"rows": [make_row(), make_row()]})

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["data"]["count"] == 2
    assert data["data"]["results"] == expected


def test_insert_returns_validation_error_for_bad_rows(client):
    payload = {"rows": [{"Machine": 1}]}
    response = client.post("/api/insert", json=payload)
    assert response.status_code == 422
    data = response.get_json()
    assert data["success"] is False
    assert isinstance(data["errors"], dict)
    assert data["errors"]


def test_insert_requires_json(client):
    response = client.post(
        "/api/insert", data="not-json", headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "Invalid JSON" in data["message"]


def test_predict_rows_failure_returns_server_error(client, monkeypatch):
    def fake_predict(rows):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.controllers.insert_controller.predict_rows", fake_predict)
    response = client.post("/api/insert", json={"rows": [make_row()]})

    assert response.status_code == 500
    data = response.get_json()
    assert data["success"] is False
    assert data["message"] == "Prediction failed"
    assert data["errors"]["server"] == ["boom"]
