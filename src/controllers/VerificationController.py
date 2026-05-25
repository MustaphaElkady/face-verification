import logging
from helpers.image import bytes_to_cv2_image
from services.FaceVerificationService import FaceVerificationService

logger = logging.getLogger(__name__)

class VerificationController:
    def __init__(self):
        self._service = FaceVerificationService()

    def verify_from_bytes(self, id_bytes: bytes, selfie_bytes: bytes) -> dict:
        id_img = bytes_to_cv2_image(id_bytes, label="ID image")
        selfie_img = bytes_to_cv2_image(selfie_bytes, label="selfie")
        return self._service.verify_faces(id_img, selfie_img)