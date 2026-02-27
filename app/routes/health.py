from pathlib import Path

from flask import Blueprint

from ..services.inference_service import MODEL_PATH

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    return {"success": True, "message": "Service is healthy"}, 200
