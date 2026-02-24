from .insert import insert_bp
from .upload import upload_bp

def register_routes(app):
    app.register_blueprint(insert_bp, url_prefix='/api')
    app.register_blueprint(upload_bp, url_prefix='/api')