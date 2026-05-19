import cv2


def normalize_channels(image):

    # grayscale -> BGR
    if len(image.shape) == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    # RGBA -> BGR
    elif len(image.shape) == 3 and image.shape[2] == 4:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR
        )

    return image
