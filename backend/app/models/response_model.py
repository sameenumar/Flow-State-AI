"""
backend/app/models/response_model.py

Pydantic models that match the actual payload schema produced by
ai_modules/main.py. These are used for the /latest REST endpoint
and for internal type validation.

The WebSocket endpoints don't validate with Pydantic — they relay
raw JSON directly for minimum latency. These models exist for
documentation, the REST fallback endpoint, and future test coverage.
"""

from pydantic import BaseModel
from typing import Optional


class FusionBlock(BaseModel):
    cognitive_state:   str   = "unknown"
    emotional_state:   str   = "unknown"
    stress_level:      str   = "unknown"
    engagement:        str   = "unknown"
    focus_score:       float = 0.0
    fusion_confidence: float = 0.0


class DecisionBlock(BaseModel):
    message:    str  = "Initializing..."
    suggestion: str  = ""
    priority:   str  = "passive"
    alert:      bool = False


class VitalsBlock(BaseModel):
    bpm:            Optional[float] = None
    hrv_sdnn:       Optional[float] = None
    stress_index:   Optional[float] = None
    signal_quality: str             = "warming_up"
    confidence:     float           = 0.0
    pulse_sample:   float           = 0.0   # single waveform point for live chart


class FaceBlock(BaseModel):
    joy:               float = 0.0
    frustration:       float = 0.0
    fatigue:           float = 0.0
    blink_rate:        int   = 0
    eye_closure_index: float = 0.0
    valence:           float = 0.0
    arousal:           float = 0.5
    face_detected:     bool  = False


class GestureBlock(BaseModel):
    fidget_level:       float           = 0.0
    face_touching:      bool            = False
    typing_cadence:     Optional[str]   = None
    posture_slump:      float           = 0.0
    prob_fidgeting:     float           = 0.0
    prob_typing:        float           = 0.0
    prob_face_touching: float           = 0.0
    hands_detected:     bool            = False


class AnalysisPayload(BaseModel):
    """
    Complete payload structure pushed from ai_modules/main.py
    to the backend every second.
    """
    timestamp: float
    fusion:    FusionBlock
    decision:  DecisionBlock
    vitals:    VitalsBlock
    face:      FaceBlock
    gesture:   GestureBlock