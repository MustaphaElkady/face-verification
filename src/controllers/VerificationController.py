from helpers.image import bytes_to_cv2_image
from services.FaceVerificationService import FaceVerificationService

import logging


class VerificationController:

    def __init__(self):

        self.service = FaceVerificationService()

    def verify_from_bytes(
        self,
        id_bytes: bytes,
        selfie_bytes: bytes
    ):

        id_img = bytes_to_cv2_image(id_bytes)
        selfie_img = bytes_to_cv2_image(selfie_bytes)

        if id_img is None:

            logging.error(
                "Could not decode document image"
            )

            raise ValueError(
                "Please reupload a clearer document photo"
            )

        if selfie_img is None:

            logging.error(
                "Could not decode selfie image"
            )

            raise ValueError(
                "Please reupload a clearer selfie photo"
            )

        return self.service.verify_faces(
            id_img,
            selfie_img
        )