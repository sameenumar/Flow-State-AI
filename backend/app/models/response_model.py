"""Response models for API endpoints."""

from pydantic import BaseModel
from typing import Optional, Dict, Any


class FaceDetectionResponse(BaseModel):
    """Response model for face detection."""
    success: bool
    faces_detected: int
    face_regions: Optional[list] = None
    confidence: Optional[float] = None


class GestureRecognitionResponse(BaseModel):
    """Response model for gesture recognition."""
    success: bool
    gesture: Optional[str] = None
    confidence: Optional[float] = None


class RPPGAnalysisResponse(BaseModel):
    """Response model for rPPG analysis."""
    success: bool
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    confidence: Optional[float] = None


class FlowStateResponse(BaseModel):
    """Response model for flow state analysis."""
    success: bool
    flow_state: Optional[str] = None
    confidence: Optional[float] = None
    face_data: Optional[Dict[str, Any]] = None
    gesture_data: Optional[Dict[str, Any]] = None
    rppg_data: Optional[Dict[str, Any]] = None


class AnalysisRequest(BaseModel):
    """Request model for analysis."""
    image_data: str  # Base64 encoded image
    analyze_face: bool = True
    analyze_gesture: bool = True
    analyze_rppg: bool = True


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    detail: Optional[str] = None
