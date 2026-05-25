import cv2
import numpy as np
from ..quality_enums import ValidationResult


class GlareValidator:
    def __init__(self, glare_threshold: float = 0.08):
        self.glare_threshold = glare_threshold

    def validate(self, image, face=None) -> ValidationResult:
        if face is not None:
            x1, y1, x2, y2 = face.bbox.astype(int)
            h, w = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                crop = image
        else:
            crop = image

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        glare_ratio = float(np.mean(gray > 245))

        if glare_ratio > self.glare_threshold:
            return ValidationResult(
                passed=False,
                code="GLARE_DETECTED",
                message="Too much glare detected. Please retake the image.",
                metrics={"glare_ratio": glare_ratio, "glare_threshold": self.glare_threshold},
            )

        return ValidationResult(
            passed=True,
            metrics={"glare_ratio": glare_ratio},
        )