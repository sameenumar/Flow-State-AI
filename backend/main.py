"""Main entry point for FastAPI backend."""

import uvicorn
from backend.app.main import app
from backend.app.config.settings import HOST, PORT, DEBUG


if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info"
    )
