"""FastAPI application initialization."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config.settings import (
    APP_NAME,
    APP_VERSION,
    CORS_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_HEADERS
)
from backend.app.routes import analyze

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="AI-powered flow state detection system"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

# Include routes
app.include_router(analyze.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "running"
    }
│   │   │   └── analyze.py       # Analysis endpoints│   │   ├── services/│   │   │   └── processor.py     # Business logic│   │   ├── models/│   │   │   └── response_model.py # Pydantic schemas│   │   ├── config/│   │   │   └── settings.py      # Configuration│   │   └── __init__.py│   ├── main.py                  # Entry point│   ├── requirements.txt│   ├── .gitignore│   └── __init__.py│├── ai_modules/                  # Core AI Logic│   ├── face_module/│   │   ├── __init__.py│   │   └── face_detection.py│   ├── gesture_module/│   │   ├── __init__.py│   │   └── gesture.py│   ├── rppg_module/│   │   ├── __init__.py│   │   └── rppg.py│   ├── fusion/│   │   ├── __init__.py│   │   └── decision.py│   └── __init__.py│├── tests/│   ├── test_face.py│   ├── test_gesture.py│   ├── test_rppg.py│   ├── conftest.py│   └── __init__.py│├── data/                        # Datasets (optional)├── docs/│   └── report.pdf│├── .gitignore├── README.md├── requirements.txt└── pytest.ini```## Installation### Prerequisites- Python 3.9+- Node.js 16+- npm or yarn### Backend Setup1. **Install Python dependencies**   ```bash   pip install -r requirements.txt   pip install -r backend/requirements.txt   ```2. **Install AI module dependencies**   ```bash   pip install opencv-python tensorflow torch mediapipe numpy scipy scikit-learn   ```### Frontend Setup1. **Navigate to frontend directory**   ```bash   cd frontend   ```2. **Install dependencies**   ```bash   npm install   ```3. **Create .env file**   ```bash   echo "REACT_APP_API_URL=http://localhost:8000" > .env   ```## Running the Application### Option 1: Run Separately (Development)**Terminal 1 - Backend:**
```bash
python backend/main.py
```
Backend runs on: `http://localhost:8000`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```
Frontend runs on: `http://localhost:3000`

### Option 2: Run with Docker Compose

```bash
docker-compose up
```

## API Endpoints

### Health Check
```
GET /health
```

### Analyze Flow State
```
POST /api/analyze
Content-Type: application/json

{
  "image_data": "base64_encoded_image",
  "analyze_face": true,
  "analyze_gesture": true,
  "analyze_rppg": true
}
```

Response:
```json
{
  "success": true,
  "flow_state": "in_flow",
  "confidence": 0.85,
  "face_data": {...},
  "gesture_data": {...},
  "rppg_data": {...}
}
```

## Features

- ✅ Real-time webcam feed
- ✅ Live face detection
- ✅ Gesture recognition for hand movements
- ✅ Non-contact vital signs monitoring (rPPG)
- ✅ Multi-modal fusion for accurate flow state detection
- ✅ Interactive React UI with real-time results
- ✅ REST API with FastAPI
- ✅ Comprehensive testing suite

## Technology Stack

### Frontend
- **React 18** - UI framework
- **Axios** - HTTP client
- **CSS3** - Styling

### Backend
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### AI/ML
- **OpenCV** - Computer vision
- **TensorFlow/PyTorch** - Deep learning
- **MediaPipe** - Gesture recognition
- **NumPy/SciPy** - Numerical computing
- **scikit-learn** - ML utilities

## Configuration

### Backend Configuration
Edit `backend/app/config/settings.py`:
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8000)
- `DEBUG` - Debug mode (default: True)
- `CORS_ORIGINS` - Allowed origins for CORS

### Frontend Configuration
Edit `frontend/.env`:
- `REACT_APP_API_URL` - Backend API URL (default: http://localhost:8000)

## Testing

Run tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_face.py -v
```

With coverage:
```bash
pytest --cov=ai_modules
```

## Development Workflow

1. Create a new branch for your feature
2. Make changes to relevant modules
3. Run tests to ensure functionality
4. Commit with descriptive messages
5. Submit a pull request

## Project Structure Explanation

- **frontend/** - React application for user interaction
- **backend/** - FastAPI server handling requests
- **ai_modules/** - Core ML/AI processing logic
- **tests/** - Unit tests for AI modules
- **data/** - Optional datasets storage
- **docs/** - Project documentation

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test thoroughly
4. Submit a pull request with detailed description

## License

MIT License - see LICENSE file for details

## Authors

- **Sameen Umar** - Project Lead

## Acknowledgments

- Computer vision: OpenCV
- Deep learning: TensorFlow, PyTorch
- Gesture recognition: MediaPipe
- Web framework: FastAPI

## Contact

For questions or suggestions, open an issue or contact the maintainers.

---

**Last Updated**: April 2026
