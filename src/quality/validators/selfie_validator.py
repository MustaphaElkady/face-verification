import cv2
import easyocr

from quality.quality_enums import ValidationResult
from core.config import settings


class SelfieValidator:
    """
    Validates that the selfie input is actually a selfie, not an ID card/document.

    Checks:
      - if OCR detects too much text in the selfie image -> probably document
      - if the face is too small -> probably ID card photo / printed photo
    """

    def __init__(
        self,
        max_text_boxes: int = 1,
        min_text_confidence: float = 0.45,
        min_selfie_face_ratio: float = 0.10,
    ):
        from core.device import easyocr_gpu_enabled

        self.reader = easyocr.Reader(
            ["ar", "en"],
            gpu=easyocr_gpu_enabled(),
        )

        self.max_text_boxes = max_text_boxes
        self.min_text_confidence = min_text_confidence
        self.min_selfie_face_ratio = float(
            getattr(settings, "min_selfie_face_ratio", min_selfie_face_ratio)
        )

    def _count_text_boxes(self, image) -> tuple[int, list[str]]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            results = self.reader.readtext(rgb, paragraph=False)
        except Exception:
            results = []

        detected_texts = [
            text
            for bbox, text, conf in results
            if conf >= self.min_text_confidence
        ]

        return len(detected_texts), detected_texts[:5]

    def validate(self, image, face=None, image_type: str = "selfie") -> ValidationResult:
        if image_type != "selfie":
            return ValidationResult(passed=True)

        text_boxes, text_preview = self._count_text_boxes(image)

        if text_boxes > self.max_text_boxes:
            return ValidationResult(
                passed=False,
                code="SELFIE_LOOKS_LIKE_DOCUMENT",
                message=(
                    "The selfie image appears to contain document text. "
                    "Please upload a real selfie photo, not an ID card or document."
                ),
                metrics={
                    "selfie_text_boxes": text_boxes,
                    "max_allowed_text_boxes": self.max_text_boxes,
                    "text_preview": text_preview,
                },
            )

        if face is not None:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = face.bbox.astype(int)

            face_area = max(0, x2 - x1) * max(0, y2 - y1)
            image_area = h * w
            face_ratio = float(face_area / image_area) if image_area > 0 else 0.0

            if face_ratio < self.min_selfie_face_ratio:
                return ValidationResult(
                    passed=False,
                    code="SELFIE_FACE_TOO_SMALL",
                    message=(
                        "The face in the selfie is too small. "
                        "Please upload a close, clear selfie facing the camera."
                    ),
                    metrics={
                        "selfie_face_area_ratio": face_ratio,
                        "min_selfie_face_ratio": self.min_selfie_face_ratio,
                    },
                )

            return ValidationResult(
                passed=True,
                metrics={
                    "selfie_text_boxes": text_boxes,
                    "selfie_face_area_ratio": face_ratio,
                },
            )

        return ValidationResult(
            passed=True,
            metrics={
                "selfie_text_boxes": text_boxes,
            },
        )