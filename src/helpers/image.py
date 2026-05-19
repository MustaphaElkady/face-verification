import cv2
import numpy as np 

def bytes_to_cv2_image(image_bytes: bytes ):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image

    except  Exception  as e:
        raise ValueError(f"ERROR: try to reupload your image {e}")