"""
backend/app/main.py

FastAPI application with:
  - REST health endpoint
  - WebSocket endpoint for receiving data from ai_modules/main.py
  - WebSocket endpoint for streaming data to React frontend
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import time

from backend.app.config.settings import (
    APP_NAME, APP_VERSION,
    CORS_ORIGINS, CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_METHODS, CORS_ALLOW_HEADERS,
)

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Flow State AI — real-time analysis backend",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)


# ── Connection managers ────────────────────────────────────────────────────────

class AIConnectionManager:
    """
    Manages the single WebSocket connection from ai_modules/main.py.
    The AI module is always one process — only one connection expected.
    Stores the latest payload so frontend clients that connect later
    immediately receive current state rather than waiting up to 1 second.
    """
    def __init__(self):
        self.connection: WebSocket = None
        self.latest_payload: dict  = None

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connection = ws
        print("[Backend] AI module connected")

    def disconnect(self):
        self.connection = None
        print("[Backend] AI module disconnected")


class FrontendConnectionManager:
    """
    Manages all WebSocket connections from React frontend clients.
    Broadcasts every payload received from the AI module to all
    connected frontend clients simultaneously.
    """
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        print(f"[Backend] Frontend client connected — total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        print(f"[Backend] Frontend client disconnected — remaining: {len(self.active)}")

    async def broadcast(self, payload: dict):
        """Send payload to all connected frontend clients."""
        if not self.active:
            return
        message = json.dumps(payload)
        disconnected = []
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)


ai_manager       = AIConnectionManager()
frontend_manager = FrontendConnectionManager()


# ── WebSocket: AI module → backend ────────────────────────────────────────────

@app.websocket("/ws/ai")
async def ai_websocket(ws: WebSocket):
    """
    Receives structured JSON payloads from ai_modules/main.py.
    Validates the payload minimally, stores it, and broadcasts to frontend.

    Expected payload shape (sent every 1 second from ai_modules/main.py):
    {
      "timestamp": float,
      "fusion":   { cognitive_state, emotional_state, stress_level, ... },
      "decision": { message, suggestion, priority, alert },
      "vitals":   { bpm, hrv_sdnn, stress_index, signal_quality, pulse_sample },
      "face":     { joy, frustration, fatigue, blink_rate, ... },
      "gesture":  { fidget_level, typing_cadence, prob_typing, ... }
    }
    """
    await ai_manager.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                print("[Backend] Received malformed JSON from AI module — skipping")
                continue

            # Store latest for clients that connect mid-session
            ai_manager.latest_payload = payload

            # Broadcast to all frontend clients
            await frontend_manager.broadcast(payload)

    except WebSocketDisconnect:
        ai_manager.disconnect()


# ── WebSocket: backend → frontend ────────────────────────────────────────────

@app.websocket("/ws/frontend")
async def frontend_websocket(ws: WebSocket):
    """
    Streams analysis results to React frontend clients.
    On connect, immediately sends the latest payload so the UI
    doesn't show empty state while waiting for the next push.
    """
    await frontend_manager.connect(ws)

    # Send most recent data immediately on connect
    if ai_manager.latest_payload:
        try:
            await ws.send_text(json.dumps(ai_manager.latest_payload))
        except Exception:
            pass

    try:
        # Keep connection alive — frontend can send pings if needed
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        frontend_manager.disconnect(ws)


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"name": APP_NAME, "version": APP_VERSION, "status": "running"}


@app.get("/health")
async def health():
    """
    Health check. Also reports whether the AI module is currently connected
    and when the last payload was received.
    """
    return {
        "status":           "healthy",
        "ai_connected":     ai_manager.connection is not None,
        "frontend_clients": len(frontend_manager.active),
        "last_update":      (ai_manager.latest_payload or {}).get("timestamp"),
    }

@app.get("/latest")
async def latest():
    if ai_manager.latest_payload is None:
        return {"status": "no_data", "message": "AI module has not sent data yet"}
    
    # Warn frontend if data is stale (AI module likely disconnected)
    age = time.time() - ai_manager.latest_payload.get("timestamp", 0)
    if age > 5:
        return {
            "status":  "stale",
            "age_seconds": round(age, 1),
            "message": "AI module may be disconnected",
            "data":    ai_manager.latest_payload,
        }
    
    return ai_manager.latest_payload