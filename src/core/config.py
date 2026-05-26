from dataclasses import dataclass
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class Settings:
    # Model
    model_backend: str = os.getenv("FACE_MODEL_BACKEND", "insightface")
    model_name: str = os.getenv("FACE_MODEL_NAME", "buffalo_l")
    use_gpu: bool = os.getenv("USE_GPU", "false").strip().lower() in ("1", "true", "yes", "y", "on")

    # Thresholds
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.57"))
    manual_review_threshold: float = float(os.getenv("MANUAL_REVIEW_THRESHOLD", "0.45"))
    min_det_score: float = float(os.getenv("MIN_DET_SCORE", "0.58"))

    # Image limits
    max_image_size_mb: float = float(os.getenv("MAX_IMAGE_SIZE_MB", "10.0"))
    max_image_dimension: int = int(os.getenv("MAX_IMAGE_DIMENSION", "4096"))

    # Quality thresholds
    blur_threshold_selfie: float = float(os.getenv("BLUR_THRESHOLD_SELFIE", "30"))
    blur_threshold_document: float = float(os.getenv("BLUR_THRESHOLD_DOCUMENT", "20"))
    brightness_low: float = float(os.getenv("BRIGHTNESS_LOW", "55"))
    brightness_high: float = float(os.getenv("BRIGHTNESS_HIGH", "235"))
    glare_threshold: float = float(os.getenv("GLARE_THRESHOLD", "10.0"))
    min_face_ratio: float = float(os.getenv("MIN_FACE_RATIO", "0.04"))

    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

settings = Settings()