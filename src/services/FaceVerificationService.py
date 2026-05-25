from insightface.app import FaceAnalysis
from quality import QualityPipeline
from utils.similarity import cosine_similarity, normalize_embedding
from preprocessing.crop import crop_face
from preprocessing.color import normalize_channels
from pathlib import Path
from vision.align import align_face
import logging


class FaceVerificationService:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=1)  # GPU
        self.quality = QualityPipeline()
        self.logger = logging.getLogger(__name__)

    # Face Selection
    def _pick_best_face(self, faces):
        if not faces:
            return None

        return max(
            faces,
            key=lambda face: float(getattr(face, "det_score", 0.0))
        )

    # face detection
    def _detect_face(self, image, image_name= "image"):
        faces = self.app.get(image)
        if len(faces) == 0:
            raise ValueError(
                f"No face detected in {image_name}. Please upload a clearer photo."
            )
        best_face = self._pick_best_face(faces)
        if best_face is None:
            raise ValueError(
                f"Could not select a valid face from {image_name}."
            )
        return best_face


    # IMAGE QUALITY VALIDATION
    def _validate_image_quality(self, image, image_name = "image", image_type = "selfie"):
        quality_results = self.quality.validate_image(image, image_type=image_type)
        if not quality_results.passed:
            raise ValueError(
                f"Quality check failed for {image_name}: {quality_results.message}"
            )
    # FACE QUALITY VALIDATION
    def _validate_face_quality(self, image, face, image_name = "image"):
        quality_results = self.quality.validate_face(image, face)
        if not quality_results.passed:
            raise ValueError(
                f"Quality check failed for {image_name}: {quality_results.message}"
            )

    # FACE ALIGNMENT
    def _align_face_image(
        self,
        image,
        face
    ):
        """
        Align image using face landmarks.
        """

        if face.kps is None:

            raise ValueError(
                "Face landmarks not found."
            )

        aligned_image = align_face(
            image=image,
            keypoints=face.kps
        )

        return aligned_image

    # EMBEDDING EXTRACTION
    def _extract_embedding(
        self,
        image,
        image_name = "image"
    ):
        face = self._detect_face(
            image=image,
            image_name=image_name
        )

        embedding = face.embedding

        if embedding is None:

            raise ValueError(
                f"Embedding extraction failed for {image_name}."
            )

        embedding = normalize_embedding(
            embedding
        )

        return embedding, face

    # =====================================================
    # MAIN VERIFICATION PIPELINE
    # =====================================================

    def verify_faces(
        self,
        id_image,
        selfie_image,
        image_type_id = "document",
        image_type_selfie = "selfie"
    ):

        # -------------------------------------------------
        # 1. Normalize Channels
        # -------------------------------------------------

        id_image = normalize_channels(
            id_image
        )

        selfie_image = normalize_channels(
            selfie_image
        )

        # 2. Validate Image Quality
        self._validate_image_quality(
            id_image,
            "document image",
            image_type=image_type_id
        )
        self._validate_image_quality(
            selfie_image,
            "selfie image",
            image_type=image_type_selfie
        )

        # 3. Detect Faces
        id_face = self._detect_face(
            id_image,
            "document image"
        )
        selfie_face = self._detect_face(
            selfie_image,
            "selfie image"
        )

        # 4. Face Quality Validation
        self._validate_face_quality(
            id_image,
            id_face,
            "document image"
        )
        self._validate_face_quality(
            selfie_image,
            selfie_face,
            "selfie image"
        )

        # 5. Align Faces
        id_aligned = self._align_face_image(
            id_image,
            id_face
        )
        selfie_aligned = self._align_face_image(
            selfie_image,
            selfie_face
        )

        # 6. Extract Embeddings
        emb1, id_face_final = self._extract_embedding(
            id_aligned,
            "aligned document image"
        )
        emb2, selfie_face_final = self._extract_embedding(
            selfie_aligned,
            "aligned selfie image"
        )

        # 7. Optional Face Crops
        id_crop = crop_face(
            id_aligned,
            id_face_final.bbox
        )
        selfie_crop = crop_face(
            selfie_aligned,
            selfie_face_final.bbox
        )

        # 8. Similarity Calculation
        similarity = cosine_similarity(
            emb1,
            emb2
        )

        # 9. Verification Decision
        threshold = 0.63

        verified = bool(
            similarity >= threshold
        )

        # 10. Logging

        self.logger.info(
            (
                "Verification completed | "
                "similarity=%.4f | "
                "verified=%s"
            ),
            similarity,
            verified
        )

        # 11. Response

        return {
            "verified": verified,

            "similarity": round(
                float(similarity),
                4
            ),

            "threshold": threshold,

            "debug": {

                "document_det_score":
                    float(id_face_final.det_score),

                "selfie_det_score":
                    float(selfie_face_final.det_score),
            }
        }