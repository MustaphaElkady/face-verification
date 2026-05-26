import cv2
import easyocr
import numpy as np
from quality.quality_enums import ValidationResult
from core.config import settings
from core.device import easyocr_gpu_enabled

class DocumentValidator:
    """
    Validates that an image contains enough text to be considered an ID document.
    Uses EasyOCR to detect text boxes.
    """
    def __init__(self, min_text_confidence: float = 0.5, min_text_boxes: int = 2):
        # Initialize the reader once for better performance
        # Supports Arabic ('ar') and English ('en') – change as needed
        self.reader = easyocr.Reader(
        ['ar', 'en'],
        gpu=easyocr_gpu_enabled(),
        )
        self.min_confidence = min_text_confidence
        self.min_text_boxes = min_text_boxes

    def validate(self, image, face=None, image_type: str = "selfie") -> ValidationResult:
        # Only apply this validation to the ID (document) image
        if image_type != "document":
            return ValidationResult(passed=True)

        # Convert BGR (OpenCV) to RGB (EasyOCR expects RGB)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run OCR
        results = self.reader.readtext(rgb, paragraph=False)

        # Filter results by confidence
        detected_texts = [
            text for (bbox, text, conf) in results if conf >= self.min_confidence
        ]

        if len(detected_texts) < self.min_text_boxes:
            return ValidationResult(
                passed=False,
                code="NOT_A_DOCUMENT",
                message=(
                    "The uploaded image does not contain enough text (not an ID document). "
                    "Please upload a clear photo of your ID document."
                ),
                metrics={
                    "text_boxes_found": len(detected_texts),
                    "required": self.min_text_boxes
                }
            )

        return ValidationResult(
            passed=True,
            metrics={"text_boxes_found": len(detected_texts)}
        )