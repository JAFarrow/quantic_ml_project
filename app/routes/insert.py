from flask import Blueprint
from flask import request, jsonify
from marshmallow import ValidationError

from ..controllers import handle_insert

insert_bp = Blueprint('insert', __name__)

@insert_bp.route('/insert', methods=['POST'])
def insert_post():
    if not request.is_json:
        return {
            "success": False,
            "message": "Invalid JSON body"
        }, 400
    return handle_insert(request)