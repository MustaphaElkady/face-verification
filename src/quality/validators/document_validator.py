import cv2
import easyocr

from quality.quality_enums import ValidationResult


class DocumentValidator:
    """
    Validates that an image contains enough text to be considered an ID document.
    Tries OCR on 0, 90, 180, 270 degree rotations.
    """

    def __init__(self, min_text_confidence: float = 0.5, min_text_boxes: int = 2):
        from core.device import easyocr_gpu_enabled

        self.reader = easyocr.Reader(
            ["ar", "en"],
            gpu=easyocr_gpu_enabled(),
        )

        self.min_confidence = min_text_confidence
        self.min_text_boxes = min_text_boxes

    def _rotate_image(self, image, angle: int):
        if angle == 0:
            return image

        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        if angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)

        if angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return image

    def _read_text_best_rotation(self, image):
        best_results = []
        best_angle = 0
        rotation_metrics = {}

        for angle in [0, 90, 180, 270]:
            rotated = self._rotate_image(image, angle)

            # OpenCV image is BGR, EasyOCR works better with RGB
            rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

            try:
                results = self.reader.readtext(rgb, paragraph=False)
            except Exception:
                results = []

            detected_texts = [
                text
                for (bbox, text, conf) in results
                if conf >= self.min_confidence
            ]

            rotation_metrics[str(angle)] = {
                "raw_boxes": len(results),
                "confident_boxes": len(detected_texts),
            }

            if len(detected_texts) > len(best_results):
                best_results = detected_texts
                best_angle = angle

        return best_results, best_angle, rotation_metrics

    def validate(self, image, face=None, image_type: str = "selfie") -> ValidationResult:
        # Only apply this validation to the ID/document image
        if image_type != "document":
            return ValidationResult(passed=True)

        detected_texts, best_angle, rotation_metrics = self._read_text_best_rotation(image)

        if len(detected_texts) < self.min_text_boxes:
            return ValidationResult(
                passed=False,
                code="NOT_A_DOCUMENT",
                message=(
                    "The uploaded image does not contain enough text "
                    "(not an ID document). Please upload a clear photo "
                    "of your ID document."
                ),
                metrics={
                    "text_boxes_found": len(detected_texts),
                    "required": self.min_text_boxes,
                    "best_rotation": best_angle,
                    "rotation_metrics": rotation_metrics,
                },
            )

        return ValidationResult(
            passed=True,
            metrics={
                "text_boxes_found": len(detected_texts),
                "best_rotation": best_angle,
                "rotation_metrics": rotation_metrics,
                "detected_text_preview": detected_texts[:5],
            },
        )