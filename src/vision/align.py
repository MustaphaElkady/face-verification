import cv2
import numpy as np


def align_face(image, keypoints):

    left_eye = keypoints[0]
    right_eye = keypoints[1]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle = np.degrees(
        np.arctan2(dy, dx)
    )

    eyes_center = (
        int((left_eye[0] + right_eye[0]) / 2),
        int((left_eye[1] + right_eye[1]) / 2)
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        eyes_center,
        angle,
        1.0
    )

    aligned = cv2.warpAffine(
        image,
        rotation_matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC
    )

    return aligned
