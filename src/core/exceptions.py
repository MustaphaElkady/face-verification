class FaceVerificationError(Exception):
    """Base for all face verification errors."""


class ImageValidationError(FaceVerificationError):
    """Bad upload: too large, wrong format, corrupted, or oversized."""


class NoFaceDetectedError(FaceVerificationError):
    """No face could be located in the image."""


class LowQualityImageError(FaceVerificationError):
    """det_score is below min_det_score — embedding would be unreliable."""
    def __init__(self, message: str, det_score: float = 0.0):
        super().__init__(message)
        self.det_score = det_score


class QualityCheckError(FaceVerificationError):
    """A quality validator (blur / brightness / glare / size / occlusion) failed."""
    def __init__(self, message: str, code: str = "", metrics: dict = None):
        super().__init__(message)
        self.code = code
        self.metrics = metrics or {}


class DuplicateImageError(FaceVerificationError):
    """Same image submitted as both ID and selfie (cosine similarity ≈ 1.0)."""


class EmbeddingExtractionError(FaceVerificationError):
    """Model ran but returned a null embedding."""


class ModelLoadError(FaceVerificationError):
    """Model backend failed to initialise."""