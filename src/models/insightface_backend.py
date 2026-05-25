import logging
from typing import List
import numpy as np
from models.base import DetectedFace, FaceModelBackend
from core.config import settings
from core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)

class InsightFaceBackend(FaceModelBackend):
    def __init__(self, model_name: str = "buffalo_l"):
        self._model_name = model_name
        self._app = None

    def load(self) -> None:
        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(name=self._model_name, allowed_modules=["detection", "recognition"])
            ctx_id = 0 if settings.use_gpu else -1
            self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info("InsightFace loaded | model=%s | device=%s", self._model_name, "GPU" if settings.use_gpu else "CPU")
        except Exception as exc:
            raise ModelLoadError(f"Failed to load InsightFace '{self._model_name}': {exc}") from exc

    def get_faces(self, image: np.ndarray) -> List[DetectedFace]:
        if self._app is None:
            raise RuntimeError("Call load() before get_faces().")
        return [
            DetectedFace(
                embedding=np.array(f.embedding, dtype=np.float32),
                det_score=float(f.det_score),
                bbox=np.array(f.bbox, dtype=np.float32),
                kps=np.array(f.kps, dtype=np.float32),
            )
            for f in self._app.get(image)
            if f.embedding is not None
        ]

    @property
    def backend_name(self) -> str:
        return f"insightface/{self._model_name}"