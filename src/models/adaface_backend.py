"""
AdaFace backend — quality-adaptive face recognition (CVPR 2022).

Why upgrade from InsightFace/ArcFace?
  ArcFace applies the same gradient weight to every sample regardless of
  image quality.  AdaFace adapts its loss margin based on feature norm
  (a proxy for quality), so low-quality ID scans lose ~12% accuracy vs
  ~26% for ArcFace.

Install:
    pip install torch torchvision adaface-pytorch

Enable:
    FACE_MODEL_BACKEND=adaface   (in .env)
"""

import logging
from typing import List

import numpy as np

from models.base import DetectedFace, FaceModelBackend
from core.config import settings
from core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class AdaFaceBackend(FaceModelBackend):
    """
    AdaFace IResNet embedding + InsightFace RetinaFace detection.

    The two models are independent: you get InsightFace's best-in-class
    detector and AdaFace's more robust low-quality embeddings together.

    Architectures:
        ir_50          ResNet-50,  MS1MV2   (~92 MB)
        ir_101         ResNet-101, MS1MV2   (~167 MB)  ← recommended
        ir_101_webface ResNet-101, WebFace4M (better demographic diversity)
    """

    def __init__(self, arch: str = "ir_101"):
        import os
        self._arch = os.getenv("ADAFACE_ARCH", arch)
        self._adaface = None
        self._detector = None

    def load(self) -> None:
        self._load_detector()
        self._load_adaface()

    def _load_detector(self) -> None:
        try:
            from insightface.app import FaceAnalysis
            self._detector = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection"],
            )
            ctx_id = 0 if settings.use_gpu else -1
            self._detector.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info("AdaFace: RetinaFace detector loaded")
        except Exception as exc:
            raise ModelLoadError(f"AdaFace: detector failed: {exc}") from exc

    def _load_adaface(self) -> None:
        try:
            from adaface import load_pretrained_model
            self._adaface = load_pretrained_model(self._arch)
            self._adaface.eval()
            logger.info("AdaFace embedding model loaded | arch=%s", self._arch)
        except ImportError as exc:
            raise ModelLoadError(
                "adaface-pytorch not installed.\n"
                "Run: pip install torch torchvision adaface-pytorch\n"
                f"Original: {exc}"
            ) from exc
        except Exception as exc:
            raise ModelLoadError(f"AdaFace: model load failed: {exc}") from exc

    def _align_112(self, image: np.ndarray, kps: np.ndarray) -> np.ndarray:
        from insightface.utils import face_align
        return face_align.norm_crop(image, landmark=kps, image_size=112)

    def _embed(self, face_112: np.ndarray) -> np.ndarray:
        import cv2
        import torch
        import torch.nn.functional as F

        rgb    = cv2.cvtColor(face_112, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb).permute(2, 0, 1)
            .float().div(255.0).sub(0.5).div(0.5).unsqueeze(0)
        )
        with torch.no_grad():
            emb, _ = self._adaface(tensor)
        return F.normalize(emb, p=2, dim=1).squeeze(0).cpu().numpy()

    def get_faces(self, image: np.ndarray) -> List[DetectedFace]:
        if self._detector is None or self._adaface is None:
            raise RuntimeError("Call load() before get_faces().")
        faces = []
        for f in self._detector.get(image):
            aligned  = self._align_112(image, f.kps)
            embedding = self._embed(aligned)
            faces.append(DetectedFace(
                embedding=embedding,
                det_score=float(f.det_score),
                bbox=np.array(f.bbox, dtype=np.float32),
                kps=np.array(f.kps, dtype=np.float32),
            ))
        return faces

    @property
    def backend_name(self) -> str:
        return f"adaface/{self._arch}"