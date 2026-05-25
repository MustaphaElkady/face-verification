from ..quality_enums import ValidationResult


class OcclusionValidator:
    def __init__(self, min_det_score: float = 0.50):
        self.min_det_score = min_det_score

    def validate(self, image, face=None) -> ValidationResult:
        if face is None:
            return ValidationResult(
                passed=False,
                code="NO_FACE_OBJECT",
                message="No face object provided for occlusion validation.",
            )

        det_score = float(getattr(face, "det_score", 0.0))
        h, w = image.shape[:2]
        x1, y1, x2, y2 = face.bbox.astype(int)

        touches_border = (
            x1 <= 2 or y1 <= 2 or x2 >= (w - 2) or y2 >= (h - 2)
        )

        if det_score < self.min_det_score:
            return ValidationResult(
                passed=False,
                code="LOW_FACE_CONFIDENCE",
                message="Face confidence is too low. The face may be partially obscured.",
                metrics={"det_score": det_score, "min_det_score": self.min_det_score},
            )

        if touches_border:
            return ValidationResult(
                passed=False,
                code="POSSIBLE_OCCLUSION",
                message="Face appears cut off or partially outside the frame.",
                metrics={"det_score": det_score, "touches_border": True},
            )

        return ValidationResult(
            passed=True,
            metrics={"det_score": det_score, "touches_border": False},
        )