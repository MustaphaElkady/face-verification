from models.base import FaceModelBackend, DetectedFace
from core.config import settings
from core.exceptions import ModelLoadError

def create_backend() -> FaceModelBackend:
    name = settings.model_backend.lower()
    if name == "insightface":
        from models.insightface_backend import InsightFaceBackend
        return InsightFaceBackend(model_name=settings.model_name)
    if name == "adaface":
        from models.adaface_backend import AdaFaceBackend
        return AdaFaceBackend()
    raise ModelLoadError(f"Unknown backend '{name}'. Use 'insightface' or 'adaface'.")

__all__ = ["FaceModelBackend", "DetectedFace", "create_backend"]