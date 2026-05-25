from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class VerificationResponse(BaseModel):
    verified: bool
    verdict: str
    similarity: float
    threshold: float
    manual_review_threshold: float
    id_det_score: float
    selfie_det_score: float
    quality_metrics: Dict[str, Any]
    warnings: List[str]
    model: str

class ErrorResponse(BaseModel):
    error: str
    code: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None