from dataclasses import dataclass


@dataclass
class Settings:

    similarity_threshold: float = 0.63

    max_image_size: int = 1600

    model_name: str = "buffalo_l"

    use_cpu: bool = True


settings = Settings()
