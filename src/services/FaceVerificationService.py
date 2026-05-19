from utils.similarity import cosine_similarity
from routes.schemes.verification import VerificationResponse
from insightface.app import FaceAnalysis
import logging


class FaceVerificationService:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=1) #cpu
        
        self.logger = logging.getLogger("uvicorn")

    def verify_faces(self, id_image, selfie_image):
        faces1 = self.app.get(id_image)
        faces2 = self.app.get(selfie_image)

# validate images
        if len(faces1) == 0:
            raise ValueError(
                "No face detected in document image. Please reupload a clearer photo."
            )

        if len(faces2) == 0:
            raise ValueError(
                "No face detected in selfie image. Please reupload a clearer selfie."
            )
        emb1 = faces1[0].embedding
        emb2 = faces2[0].embedding

# calculate similarity score 
        if emb1 is None or emb2 is None:
            return self.logger.error("Embedding extraction failed")
        similarity = cosine_similarity(a=emb1, b=emb2)

        threshold = 0.76

        verified = bool(similarity >= threshold)
        return VerificationResponse(
            verified = verified,
            similarity= round(float(similarity), 4)
        )
