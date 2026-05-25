import io
import cv2
import numpy as np
from PIL import Image, ExifTags, UnidentifiedImageError
from core.config import settings
from core.exceptions import ImageValidationError

ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "BMP"}
_EXIF_ROTATIONS = {3: 180, 6: 270, 8: 90}

def bytes_to_cv2_image(data: bytes, label: str = "image") -> np.ndarray:
    # Size guard
    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.max_image_size_mb:
        raise ImageValidationError(f"{label} too large ({size_mb:.1f} MB). Max {settings.max_image_size_mb} MB.")

    try:
        pil_img = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise ImageValidationError(f"{label} cannot be decoded. Use JPG, PNG, or WEBP.")
    except Exception as exc:
        raise ImageValidationError(f"{label} corrupted: {exc}")

    fmt = (pil_img.format or "").upper()
    if fmt not in ALLOWED_FORMATS:
        raise ImageValidationError(f"{label} format {fmt} not allowed. Allowed: {sorted(ALLOWED_FORMATS)}")

    # Apply EXIF rotation
    try:
        exif = pil_img._getexif()
        if exif:
            orientation_key = next((k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None)
            if orientation_key and orientation_key in exif:
                degrees = _EXIF_ROTATIONS.get(exif[orientation_key])
                if degrees:
                    pil_img = pil_img.rotate(degrees, expand=True)
    except Exception:
        pass

    w, h = pil_img.size
    if max(w, h) > settings.max_image_dimension:
        raise ImageValidationError(f"{label} too large ({w}x{h}). Max {settings.max_image_dimension} px.")

    rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)