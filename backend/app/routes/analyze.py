"""API routes for analysis endpoints."""

from fastapi import APIRouter, HTTPException
from backend.app.models.response_model import (
    AnalysisRequest,
    FlowStateResponse,
    ErrorResponse
)
from backend.app.services.processor import AnalysisProcessor

router = APIRouter(prefix="/api", tags=["analysis"])
processor = AnalysisProcessor()


@router.post("/analyze", response_model=FlowStateResponse)
async def analyze_flow_state(request: AnalysisRequest):
    """
    Analyze flow state from image data.
    
    Args:
        request: AnalysisRequest with base64 image data
        
    Returns:
        FlowStateResponse with analysis results
    """
    try:
        result = processor.process_image(request.image_data)
        
        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Analysis failed")
            )
        
        return FlowStateResponse(
            success=True,
            flow_state=result.get("flow_state", {}).get("flow_state"),
            confidence=result.get("flow_state", {}).get("confidence"),
            face_data=result.get("face_data"),
            gesture_data=result.get("gesture_data"),
            rppg_data=result.get("rppg_data")
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Flow-State-AI"}
