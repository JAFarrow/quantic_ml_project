from flask import request
import io
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from marshmallow import ValidationError

from ..services import predict_rows
from ..models import PredictionBatchSchema
from ..helpers import build_evaluation

batch_schema = PredictionBatchSchema()

def handle_upload(req: request):
    file = req.files['file']
    
    if not file or file.filename == "":
        return {
            "success": False,
            "message": "No file selected",
            "errors": {
                "file": "Empty filename"
            }
        }, 400

    if not file.filename.lower().endswith(".csv"):
        return {
            "success": False,
            "message": "Invalid file type",
            "errors": {
                "file": "Only csv files are supported"
            }
        }, 415

    try:
        raw = file.read()

        if not raw:
            return {
                "success": False,
                "message": "Uploaded file is empty",
                "errors": {
                    "file": "Empty file content"
                }
            }, 400

        text = raw.decode("utf-8")
        df = pd.read_csv(io.StringIO(text))
        df = df.astype(object).where(pd.notnull(df), None)

        y_true = df["Label"].tolist() if "Label" in df.columns else None

        rows = df.to_dict(orient="records")

        validated = batch_schema.load({ "rows": rows })

        predictions = predict_rows(validated["rows"])

        if y_true is not None:
            evaluation = build_evaluation(y_true, predictions)
        else:
            evaluation = {
            "available": False,
            "message": "No label column found. Predictions returned without evaluation",
        }

        return {
            "success": True,
            "message": "Inference successful",
            "data": {
                "count": len(predictions),
                "results": predictions,
                "evaluation": evaluation
            }
        }, 200

    except UnicodeDecodeError:
        return {
            "success": False,
            "message": "Could not decode file",
            "errors": {"file": "CSV must be UTF-8 encoded"},
        }, 400

    except ParserError as e:
        return {
                "success": False,
                "message": "Invalid CSV format",
                "errors": {"file": str(e)},
            }, 422

    except ValidationError as e:
        return {
                "success": False,
                "message": "Validation failed",
                "errors": e.messages,
            }, 422

    except Exception as e:
        return {
            "success": False,
            "message": "Prediction failed",
            "errors": {"server": [str(e)]}
        }, 500
