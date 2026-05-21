import cv2


def improve_brightness(
    image,
    alpha: float = 1.1,
    beta: int = 10
):

    enhanced = cv2.convertScaleAbs(
        image,
        alpha=alpha,
        beta=beta
    )

    return enhanced
