from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class DetectedFace:
    embedding: np.ndarray
    det_score: float
    bbox: np.ndarray
    kps: np.ndarray

class FaceModelBackend(ABC):
    @abstractmethod
    def load(self) -> None: ...
    @abstractmethod
    def get_faces(self, image: np.ndarray) -> List[DetectedFace]: ...
    @property
    @abstractmethod
    def backend_name(self) -> str: ...