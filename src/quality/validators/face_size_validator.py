from ..quality_enums import ValidationResult


class FaceSizeValidator:
    def __init__(self, min_face_ratio: float = 0.04):
        self.min_face_ratio = min_face_ratio

    def validate(self, image, face=None) -> ValidationResult:
        if face is None:
            return ValidationResult(
                passed=False,
                code="NO_FACE_OBJECT",
                message="No face object provided for face size validation.",
            )

        h, w = image.shape[:2]
        x1, y1, x2, y2 = face.bbox.astype(int)

        face_area = max(0, x2 - x1) * max(0, y2 - y1)
        image_area = h * w
        ratio = float(face_area / image_area) if image_area > 0 else 0.0

        if ratio < self.min_face_ratio:
            return ValidationResult(
                passed=False,
                code="FACE_TOO_SMALL",
                message="Face is too small for reliable verification.",
                metrics={"face_area_ratio": ratio, "min_face_ratio": self.min_face_ratio},
            )

        return ValidationResult(
            passed=True,
            metrics={"face_area_ratio": ratio},
        )