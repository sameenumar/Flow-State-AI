import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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