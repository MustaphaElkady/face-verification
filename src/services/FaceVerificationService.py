from insightface.app import FaceAnalysis
import logging

from utils.similarity import cosine_similarity, normalize_embedding
from preprocessing.crop import crop_face
from preprocessing.color import normalize_channels
from vision.align import align_face


class FaceVerificationService:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0)  # CPU
        self.logger = logging.getLogger(__name__)

    def _pick_best_face(self, faces):
        if not faces:
            return None

        return max(
            faces,
            key=lambda face: float(getattr(face, "det_score", 0.0))
        )

    def verify_faces(self, id_image, selfie_image):
        # 1) Normalize image channels
        id_image = normalize_channels(id_image)
        selfie_image = normalize_channels(selfie_image)

        # 2) First face detection pass
        id_faces = self.app.get(id_image)
        selfie_faces = self.app.get(selfie_image)

        if len(id_faces) == 0:
            raise ValueError(
                "No face detected in document image. Please upload a clearer photo."
            )

        if len(selfie_faces) == 0:
            raise ValueError(
                "No face detected in selfie image. Please upload a clearer selfie."
            )

        # 3) Pick the best face from each image
        id_face = self._pick_best_face(id_faces)
        selfie_face = self._pick_best_face(selfie_faces)

        if id_face is None or selfie_face is None:
            raise ValueError("Could not select a valid face.")

        # 4) Align the full images using facial landmarks
        id_aligned = align_face(id_image, id_face.kps)
        selfie_aligned = align_face(selfie_image, selfie_face.kps)

        # 5) Detect again after alignment
        id_aligned_faces = self.app.get(id_aligned)
        selfie_aligned_faces = self.app.get(selfie_aligned)

        if len(id_aligned_faces) == 0:
            raise ValueError("No face detected after alignment in document image.")

        if len(selfie_aligned_faces) == 0:
            raise ValueError("No face detected after alignment in selfie image.")

        # 6) Pick final aligned face
        id_face_final = self._pick_best_face(id_aligned_faces)
        selfie_face_final = self._pick_best_face(selfie_aligned_faces)

        if id_face_final is None or selfie_face_final is None:
            raise ValueError("Could not select a valid aligned face.")

        # 7) Optional crop for debugging/inspection
        # You do NOT need to use this crop for the embedding if the aligned face is already good.
        id_face_crop = crop_face(id_aligned, id_face_final.bbox)
        selfie_face_crop = crop_face(selfie_aligned, selfie_face_final.bbox)

        # 8) Use embeddings from aligned detections
        emb1 = id_face_final.embedding
        emb2 = selfie_face_final.embedding

        if emb1 is None or emb2 is None:
            raise ValueError("Embedding extraction failed.")

        # 9) Normalize embeddings
        emb1 = normalize_embedding(emb1)
        emb2 = normalize_embedding(emb2)

        # 10) Similarity
        similarity = cosine_similarity(emb1, emb2)

        # 11) Threshold
        threshold = 0.63
        verified = bool(similarity >= threshold)

        self.logger.info(
            "Verification done | similarity=%.4f | verified=%s",
            similarity,
            verified
        )

        return {
            "verified": verified,
            "similarity": round(float(similarity), 4)
        }