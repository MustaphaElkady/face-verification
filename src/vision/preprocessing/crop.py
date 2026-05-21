import cv2


def crop_face(
    image,
    bbox,
    padding: float = 0.25
):

    x1, y1, x2, y2 = bbox.astype(int)

    h, w = image.shape[:2]

    face_width = x2 - x1
    face_height = y2 - y1

    pad_x = int(face_width * padding)
    pad_y = int(face_height * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)

    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    cropped = image[y1:y2, x1:x2]

    return cropped
