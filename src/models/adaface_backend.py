"""
AdaFace backend — quality-adaptive face recognition.

Uses:
    - InsightFace RetinaFace for detection
    - AdaFace for embeddings

Enable in .env:
    FACE_MODEL_BACKEND=adaface
"""

import logging
from typing import List

import numpy as np

from models.base import DetectedFace, FaceModelBackend
from core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class AdaFaceBackend(FaceModelBackend):
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
            from core.device import onnxruntime_execution

            providers, ctx_id, device_label = onnxruntime_execution()

            self._detector = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection"],
                providers=providers,
            )

            self._detector.prepare(
                ctx_id=ctx_id,
                det_size=(640, 640),
            )

            logger.info(
                "AdaFace detector loaded | device=%s | providers=%s",
                device_label,
                providers,
            )

        except Exception as exc:
            raise ModelLoadError(f"AdaFace: detector failed: {exc}") from exc

    def _load_adaface(self) -> None:
        try:
            from adaface import load_pretrained_model
            from core.device import torch_device

            device = torch_device()

            self._adaface = load_pretrained_model(self._arch)
            self._adaface.eval()
            self._adaface.to(device)

            logger.info(
                "AdaFace embedding model loaded | arch=%s | device=%s",
                self._arch,
                device,
            )

        except ImportError as exc:
            raise ModelLoadError(
                "AdaFace package is not installed or import path is different. "
                "Install/check your AdaFace package.\n"
                f"Original error: {exc}"
            ) from exc

        except Exception as exc:
            raise ModelLoadError(f"AdaFace: model load failed: {exc}") from exc

    def _align_112(self, image: np.ndarray, kps: np.ndarray) -> np.ndarray:
        from insightface.utils import face_align

        return face_align.norm_crop(
            image,
            landmark=kps,
            image_size=112,
        )

    def _embed(self, face_112: np.ndarray) -> np.ndarray:
        import cv2
        import torch
        import torch.nn.functional as F
        from core.device import torch_device

        device = torch_device()

        rgb = cv2.cvtColor(face_112, cv2.COLOR_BGR2RGB)

        tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .sub(0.5)
            .div(0.5)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            emb, _ = self._adaface(tensor)

        return F.normalize(
            emb,
            p=2,
            dim=1,
        ).squeeze(0).cpu().numpy()

    def get_faces(self, image: np.ndarray) -> List[DetectedFace]:
        if self._detector is None or self._adaface is None:
            raise RuntimeError("Call load() before get_faces().")

        faces = []

        for face in self._detector.get(image):
            aligned = self._align_112(image, face.kps)
            embedding = self._embed(aligned)

            faces.append(
                DetectedFace(
                    embedding=embedding,
                    det_score=float(face.det_score),
                    bbox=np.array(face.bbox, dtype=np.float32),
                    kps=np.array(face.kps, dtype=np.float32),
                )
            )

        return faces

    @property
    def backend_name(self) -> str:
        return f"adaface/{self._arch}"