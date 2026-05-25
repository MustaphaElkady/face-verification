from quality.quality_enums import ValidationResult
from core.config import settings

class OcclusionValidator:
    def __init__(self, min_det_score: float = None):
        self.min_det_score = min_det_score or settings.min_det_score

    def validate(self, image, face=None, image_type: str = "selfie") -> ValidationResult:
        if face is None:
            return ValidationResult(passed=False, code="NO_FACE_OBJECT", message="No face object provided.")
        det_score = float(getattr(face, "det_score", 0.0))
        h, w = image.shape[:2]
        x1, y1, x2, y2 = face.bbox.astype(int)
        touches_border = (x1 <= 2 or y1 <= 2 or x2 >= w-2 or y2 >= h-2)
        if det_score < self.min_det_score:
            return ValidationResult(
                passed=False, code="LOW_FACE_CONFIDENCE",
                message=f"Face confidence too low ({det_score:.2f}).",
                metrics={"det_score": round(det_score, 4), "min_det_score": self.min_det_score},
            )
        if touches_border:
            return ValidationResult(
                passed=False, code="FACE_OUT_OF_FRAME",
                message="Face is partially outside the frame.",
                metrics={"det_score": round(det_score, 4), "touches_border": True},
            )
        return ValidationResult(passed=True, metrics={"det_score": round(det_score, 4), "touches_border": False})