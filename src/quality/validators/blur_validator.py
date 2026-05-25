from ..quality_enums import ValidationResult
import cv2


class BlurValidator:
    def __init__(self, document_threshold=12, selfie_threshold=60):
        self.document_threshold = document_threshold
        self.selfie_threshold = selfie_threshold

    def validate(self, image, face=None, image_type="selfie") -> ValidationResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        threshold = self.document_threshold if image_type == "document" else self.selfie_threshold

        if score < threshold:
            return ValidationResult(
                passed=False,
                code="BLURRY_IMAGE",
                message="Image is too blurry. Please retake it.",
                metrics={"blur_score": score, "threshold": threshold},
            )

        return ValidationResult(
            passed=True,
            metrics={"blur_score": score, "threshold": threshold},
        )