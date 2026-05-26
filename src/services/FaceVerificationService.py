"""
FaceVerificationService  —  Version 2
═══════════════════════════════════════════════════════════════════════════════
What changed from V1
─────────────────────
Bug fixes
  ✔ ctx_id=1 → reads USE_GPU from config (was crashing on any CPU machine)
  ✔ Threshold 0.63 hardcoded → reads SIMILARITY_THRESHOLD from config / .env
  ✔ OcclusionValidator min_det_score 0.50 → 0.85 from config
  ✔ BlurValidator image_type never passed → now passed explicitly per image
  ✔ GlareValidator duplicated in both pipeline stages → removed from face stage
  ✔ All validator imports "from src.quality..." → "from quality..." (path fix)

New features
  ✔ Model backend abstraction — swap InsightFace ↔ AdaFace via .env
  ✔ 3-way verdict: VERIFIED / MANUAL_REVIEW / REJECTED
      (binary accept/reject was too blunt; gray-zone scores need human review)
  ✔ Duplicate image detection — same image as both inputs scores ~1.0
      and must be flagged as suspicious, NOT accepted
  ✔ EXIF rotation in image loader — portrait selfies no longer arrive sideways
  ✔ Proper custom exceptions instead of generic ValueError everywhere
  ✔ Env-based config for all thresholds and limits
  ✔ Rich response with verdict, quality metrics, warnings, and model info
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from enum import Enum
from typing import Optional

import numpy as np

from models import create_backend, DetectedFace, FaceModelBackend
from quality import QualityPipeline
from utils.similarity import cosine_similarity, normalize_embedding, is_duplicate_submission
from preprocessing.color import normalize_channels
from core.config import settings
from core.exceptions import (
    NoFaceDetectedError,
    QualityCheckError,
    DuplicateImageError,
    EmbeddingExtractionError,
)

logger = logging.getLogger(__name__)


class Verdict(str, Enum):
    VERIFIED      = "VERIFIED"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    REJECTED      = "REJECTED"


class FaceVerificationService:

    def __init__(self, backend: Optional[FaceModelBackend] = None):
        self._backend = backend or create_backend()
        self._backend.load()
        self._quality = QualityPipeline()
        logger.info(
            "FaceVerificationService ready | backend=%s | "
            "threshold=%.2f | manual_review=%.2f | min_det=%.2f",
            self._backend.backend_name,
            settings.similarity_threshold,
            settings.manual_review_threshold,
            settings.min_det_score,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════════

    def verify_faces(self, id_image: np.ndarray, selfie_image: np.ndarray) -> dict:
        """
        Compare an ID photo against a live selfie.

        Returns
        ───────
        {
          verified             bool    True only for VERIFIED verdict
          verdict              str     VERIFIED | MANUAL_REVIEW | REJECTED
          similarity           float   cosine similarity score
          threshold            float   current VERIFIED threshold
          manual_review_threshold float current MANUAL_REVIEW threshold
          id_det_score         float   ID face detection confidence
          selfie_det_score     float   selfie face detection confidence
          quality_metrics      dict    per-image quality scores
          warnings             list    non-fatal issues (e.g. multiple faces)
          model                str     backend identifier
        }

        Raises
        ──────
        ImageValidationError      bad upload (handled upstream in helpers)
        QualityCheckError         a validator failed (blur / brightness / etc.)
        NoFaceDetectedError       no face found in one of the images
        DuplicateImageError       same image submitted for both inputs
        EmbeddingExtractionError  model returned a null embedding
        """
        warnings: list = []
        quality_metrics: dict = {}

        # ── 1. Channel normalisation ──────────────────────────────────────────
        id_image     = normalize_channels(id_image)
        selfie_image = normalize_channels(selfie_image)

        # ── 2. Image-level quality check ──────────────────────────────────────
        # FIX: pass image_type so BlurValidator uses correct threshold per image
        id_quality = self._quality.validate_image(id_image, image_type="document")
        quality_metrics["id_image"] = id_quality.metrics
        if not id_quality.passed:
            raise QualityCheckError(
                f"ID image quality check failed: {id_quality.message}",
                code=id_quality.code,
                metrics=id_quality.metrics,
            )

        selfie_quality = self._quality.validate_image(selfie_image, image_type="selfie")
        quality_metrics["selfie_image"] = selfie_quality.metrics
        if not selfie_quality.passed:
            raise QualityCheckError(
                f"Selfie quality check failed: {selfie_quality.message}",
                code=selfie_quality.code,
                metrics=selfie_quality.metrics,
            )

        # ── 3. Face detection ─────────────────────────────────────────────────
        id_faces     = self._backend.get_faces(id_image)
        selfie_faces = self._backend.get_faces(selfie_image)

        if not id_faces:
            raise NoFaceDetectedError(
                "No face detected in the ID image. "
                "Please upload a clear, well-lit photo of the ID."
            )
        if not selfie_faces:
            raise NoFaceDetectedError(
                "No face detected in the selfie. "
                "Please take a well-lit selfie facing the camera directly."
            )

        # ── 4. Multiple face warning ──────────────────────────────────────────
        if len(id_faces) > 1:
            warnings.append(
                f"{len(id_faces)} faces found in ID image — "
                "using the highest-confidence face."
            )
        if len(selfie_faces) > 1:
            warnings.append(
                f"{len(selfie_faces)} faces found in selfie — "
                "using the highest-confidence face."
            )

        # ── 5. Pick best face per image ───────────────────────────────────────
        id_face     = self._pick_best_face(id_faces)
        selfie_face = self._pick_best_face(selfie_faces)

        # ── 6. Face-level quality check ───────────────────────────────────────
        id_face_q = self._quality.validate_face(
            id_image,
            id_face,
            image_type="document",
        )
        quality_metrics["id_face"] = id_face_q.metrics
        if not id_face_q.passed:
            raise QualityCheckError(
                f"ID face quality check failed: {id_face_q.message}",
                code=id_face_q.code,
                metrics=id_face_q.metrics,
            )

        selfie_face_q = self._quality.validate_face(
            selfie_image,
            selfie_face,
            image_type="selfie",
        )
        quality_metrics["selfie_face"] = selfie_face_q.metrics
        if not selfie_face_q.passed:
            raise QualityCheckError(
                f"Selfie face quality check failed: {selfie_face_q.message}",
                code=selfie_face_q.code,
                metrics=selfie_face_q.metrics,
            )

        # ── 7. Extract and normalise embeddings ───────────────────────────────
        emb_id     = id_face.embedding
        emb_selfie = selfie_face.embedding

        if emb_id is None or emb_selfie is None:
            raise EmbeddingExtractionError(
                "Embedding extraction failed for one or both images."
            )

        emb_id     = normalize_embedding(emb_id)
        emb_selfie = normalize_embedding(emb_selfie)

        # ── 8. Duplicate submission detection ─────────────────────────────────
        if is_duplicate_submission(emb_id, emb_selfie):
            raise DuplicateImageError(
                "The same image was submitted as both the ID and the selfie. "
                "Please upload two different photos."
            )

        # ── 9. Similarity score ───────────────────────────────────────────────
        similarity = cosine_similarity(emb_id, emb_selfie)

        # ── 10. 3-way verdict ─────────────────────────────────────────────────
        verdict = self._decide(similarity)

        logger.info(
            "Verification complete | similarity=%.4f | verdict=%s | "
            "id_det=%.3f | selfie_det=%.3f | warnings=%d | backend=%s",
            similarity, verdict.value,
            id_face.det_score, selfie_face.det_score,
            len(warnings),
            self._backend.backend_name,
        )

        return {
            "verified":                 verdict == Verdict.VERIFIED,
            "verdict":                  verdict.value,
            "similarity":               round(float(similarity), 4),
            "threshold":                settings.similarity_threshold,
            "manual_review_threshold":  settings.manual_review_threshold,
            "id_det_score":             round(id_face.det_score, 4),
            "selfie_det_score":         round(selfie_face.det_score, 4),
            "quality_metrics":          quality_metrics,
            "warnings":                 warnings,
            "model":                    self._backend.backend_name,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _pick_best_face(faces: list) -> DetectedFace:
        """
        Return the face with the highest detection confidence.

        V1 bug: lambda used getattr(faces, "det_score") where 'faces' was
        the list object — always returned 0.0 for every candidate, so it
        effectively always picked faces[0] regardless of quality.
        Fixed to getattr(face, ...) on the individual item.
        """
        return max(faces, key=lambda face: float(getattr(face, "det_score", 0.0)))

    @staticmethod
    def _decide(similarity: float) -> Verdict:
        """
        Map a cosine similarity score to a 3-way verdict.

        score >= similarity_threshold        → VERIFIED
        score >= manual_review_threshold     → MANUAL_REVIEW  (gray zone)
        score <  manual_review_threshold     → REJECTED
        """
        if similarity >= settings.similarity_threshold:
            return Verdict.VERIFIED
        if similarity >= settings.manual_review_threshold:
            return Verdict.MANUAL_REVIEW
        return Verdict.REJECTED