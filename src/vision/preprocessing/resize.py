import cv2 
def resize_for_detection(image, max_side: int = 1280):
    h, w = image.shape[:2]
    longest = max(h,w)

    if longest <= max_side:
        return image, 1.0

    scale = max_side / float(longest)

    new_w = int(w*scale)
    new_h = int(h*scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale
