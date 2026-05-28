"""WSGI entry point for gunicorn."""

from sensqml.api.app import create_app

app = create_app()

if __name__ == "__main__":
    import os
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
