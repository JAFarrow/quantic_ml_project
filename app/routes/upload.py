from flask import Blueprint
from flask import request, jsonify
from marshmallow import ValidationError

from ..controllers import handle_upload

upload_bp = Blueprint("upload", __name__)


@upload_bp.route("/upload", methods=["POST"])
def upload_post():
    if "file" not in request.files:
        return {
            "success": False,
            "message": "No file was uploaded",
            "errors": {"request": "Missing file field"},
        }, 400
    return handle_upload(request)
