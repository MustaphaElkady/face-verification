import cv2
import numpy as np
from ..quality_enums import ValidationResult
from core.config import settings

class BrightnessValidator:
    def __init__(self, low_threshold: float = 55.0, high_threshold: float = 235.0):
        self.low_threshold = settings.brightness_low
        self.high_threshold = settings.brightness_high

    def validate(self, image, face=None) -> ValidationResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))

        if brightness < self.low_threshold:
            return ValidationResult(
                passed=False,
                code="LOW_LIGHT_IMAGE",
                message="Image is too dark. Please retake it in better light.",
                metrics={"brightness": brightness, "low_threshold": self.low_threshold},
            )

        if brightness > self.high_threshold:
            return ValidationResult(
                passed=False,
                code="OVEREXPOSED_IMAGE",
                message="Image is overexposed. Please retake it.",
                metrics={"brightness": brightness, "high_threshold": self.high_threshold},
            )

        return ValidationResult(
            passed=True,
            metrics={"brightness": brightness},
        )