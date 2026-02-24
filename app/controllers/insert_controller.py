from flask import request
from marshmallow import ValidationError

from ..services import predict_rows
from ..models import PredictionBatchSchema

batch_schema = PredictionBatchSchema()

def handle_insert(req: request):
    payload = req.get_json()
    try:
        validated = batch_schema.load(payload)
    except ValidationError as err:
        return {
            "success": False,
            "message": "Validation failed",
            "errors": err.messages
        }, 422

    rows = validated["rows"]

    try:
        predictions = predict_rows(rows)
    except Exception as e:
        return {
            "success": False,
            "message": "Prediction failed",
            "errors": {"server": [str(e)]}
        }, 500

    return {
        "success": True,
        "message": "Inference succesful",
        "data": {
            "count": len(predictions),
            "results": predictions,
        }
    }, 200
