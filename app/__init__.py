import os

from flask import Flask
from flask_cors import CORS


def create_app():
    app = Flask(__name__)

    from .routes import register_routes

    register_routes(app)

    origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

    CORS(
        app,
        resources={r"/api/*": {"origins": origins}},
        supports_credentials=False,
        methods=["POST", "OPTIONS"],
        max_age=86400,
    )

    return app
