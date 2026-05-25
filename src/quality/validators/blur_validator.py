import cv2
from quality.quality_enums import ValidationResult
from core.config import settings

class BlurValidator:
    def __init__(self, document_threshold: float = None, selfie_threshold: float = None):
        self.document_threshold = document_threshold or settings.blur_threshold_document
        self.selfie_threshold   = selfie_threshold   or settings.blur_threshold_selfie

    def validate(self, image, face=None, image_type: str = "selfie") -> ValidationResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        threshold = self.document_threshold if image_type == "document" else self.selfie_threshold
        if score < threshold:
            return ValidationResult(
                passed=False, code="BLURRY_IMAGE",
                message="Image is too blurry. Please retake it.",
                metrics={"blur_score": round(score, 2), "threshold": threshold},
            )
        return ValidationResult(passed=True, metrics={"blur_score": round(score, 2), "threshold": threshold})