import csv
import io
from pathlib import Path

import pytest

from app import create_app


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Integration tests require artifacts/model.joblib",
)

CSV_HEADERS = [
    "BaseOfCode",
    "BaseOfData",
    "Characteristics",
    "DllCharacteristics",
    "Entropy",
    "FileAlignment",
    "FirstSeenDate",
    "Identify",
    "ImageBase",
    "ImportedDlls",
    "ImportedSymbols",
    "Machine",
    "Magic",
    "NumberOfRvaAndSizes",
    "NumberOfSections",
    "NumberOfSymbols",
    "PE_TYPE",
    "PointerToSymbolTable",
    "SHA1",
    "Size",
    "SizeOfCode",
    "SizeOfHeaders",
    "SizeOfImage",
    "SizeOfInitializedData",
    "SizeOfOptionalHeader",
    "SizeOfUninitializedData",
    "TimeDateStamp",
    "Label",
]

BASE_ROW = {
    "BaseOfCode": 4096,
    "BaseOfData": 69632,
    "Characteristics": 783,
    "DllCharacteristics": 0,
    "Entropy": 5.981248597142612,
    "FileAlignment": 512,
    "FirstSeenDate": "1970-01-01",
    "Identify": "powerbasic/win 8.00",
    "ImageBase": 4194304,
    "ImportedDlls": "comdlg32.dll gdi32.dll kernel32.dll ole32.dll oleaut32.dll user32.dll comctl32.dll libnodave.dll",
    "ImportedSymbols": "printdlga getopenfilenamea getsavefilenamea bitblt createcompatiblebitmap createcompatibledc createfontindirecta createsolidbrush deletedc deleteobject getdevicecaps getstockobject gettextmetricsa movetoex selectobject setbkcolor setbkmode settextalign settextcolor closehandle createfilea enumresourcenamesa exitprocess getcommandlinea getcurrentdirectorya getlasterror getmodulehandlea getstartupinfoa getversionexa globalalloc globalfree multibytetowidechar readfile setcurrentdirectorya seterrormode setfilepointer setlasterror sleep tlsalloc tlsfree tlsgetvalue tlssetvalue widechartomultibyte writefile rtlmovememory clsidfromprogid cocreateinstance coinitialize couninitialize progidfromclsid getactiveobject safearraycreate sysallocstringbytelen sysfreestring sysstringbytelen variantclear variantcopy checkradiobutton clienttoscreen createdialogindirectparama createdialogparama createwindowexa destroyicon destroywindow dialogboxindirectparama dispatchmessagea enablewindow fillrect getclientrect getdc getdlgitem getmenu getmenuiteminfoa getsyscolor getsyscolorbrush getwindowlonga getwindowrect getwindowtexta getwindowtextlengtha isdialogmessagea iswindow loadimagea mapdialogrect peekmessagea postmessagea redrawwindow releasedc screentoclient sendmessagea setfocus setwindowlonga setwindowpos setwindowtexta showwindow systemparametersinfoa translatemessage dialogboxparama getfocus getwindow imagelist_replaceicon imagelist_remove imagelist_geticon imagelist_loadimagea davestrerror davestringcopy davenewinterface davenewconnection daveareaname daveblockname davegets32 davegetfloat davegetfloatat daveput32 daveputfloat daveconnectplc davereadbytes davewritebytes daveinitadapter davedisconnectplc davedisconnectadapter davegetname davefree setport closeport",
    "Machine": 332,
    "Magic": 267,
    "NumberOfRvaAndSizes": 16,
    "NumberOfSections": 5,
    "NumberOfSymbols": 0,
    "PE_TYPE": 267,
    "PointerToSymbolTable": 0,
    "SHA1": "b0068836a40e6a43c6b546fcb709237e5aa223d1",
    "Size": 76288,
    "SizeOfCode": 64855,
    "SizeOfHeaders": 1024,
    "SizeOfImage": 86016,
    "SizeOfInitializedData": 2560,
    "SizeOfOptionalHeader": 224,
    "SizeOfUninitializedData": 1500,
    "TimeDateStamp": 12345,
    "Label": 0,
}


@pytest.fixture(scope="session")
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def make_row(**overrides):
    row = BASE_ROW.copy()
    row.update(overrides)
    return row


def _rows_to_csv(rows: list[dict[str, object]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_HEADERS)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {col: "" if row.get(col) is None else row[col] for col in CSV_HEADERS}
        )
    return output.getvalue()


def _assert_prediction_format(results: list[dict[str, object]]):
    assert isinstance(results, list)
    assert results, "Predictions list should not be empty"
    for idx, entry in enumerate(results):
        assert entry["row_index"] == idx
        prob_value = entry["probability"]
        assert isinstance(prob_value, (float, int))
        prob = float(prob_value)
        assert 0.0 <= prob <= 1.0
        assert isinstance(entry["prediction"], int)


def test_insert_route_integration(client):
    response = client.post("/api/insert", json={"rows": [make_row()]})
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["data"]["count"] == 1
    _assert_prediction_format(data["data"]["results"])


def test_upload_route_integration(client):
    rows = [make_row(Label=1), make_row(Label=0, Entropy=4.2, TimeDateStamp=1665000000)]
    csv_text = _rows_to_csv(rows)
    payload = {
        "file": (io.BytesIO(csv_text.encode("utf-8")), "batch.csv"),
    }
    response = client.post(
        "/api/upload",
        data=payload,
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["data"]["count"] == len(rows)
    _assert_prediction_format(data["data"]["results"])

    evaluation = data["data"]["evaluation"]
    assert evaluation["available"] is True
    assert 0.0 <= evaluation["accuracy"] <= 1.0
    assert 0.0 <= evaluation["auc"] <= 1.0
    cm = evaluation["confusion_matrix"]
    for key in ("tn", "fp", "fn", "tp"):
        assert isinstance(cm[key], int)
