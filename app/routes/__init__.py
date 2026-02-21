from .insert import insert_bp

def register_routes(app):
    app.register_blueprint(insert_bp, url_prefix='/api')