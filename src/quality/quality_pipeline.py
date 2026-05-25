from dataclasses import dataclass, field

from quality.validators.blur_validator import BlurValidator
from quality.validators.brightness_validator import BrightnessValidator
from quality.validators.face_size_validator import FaceSizeValidator
from quality.validators.glare_validator import GlareValidator
from quality.validators.occlusion_validator import OcclusionValidator


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    code: str | None = None
    message: str | None = None
    metrics: dict = field(default_factory=dict)


class QualityPipeline:
    def __init__(self):
        self.image_validators = [
            BlurValidator(),  
            BrightnessValidator(),
            GlareValidator(),
        ]
        self.face_validators = [
            FaceSizeValidator(),
            OcclusionValidator(),
            GlareValidator(),
        ]

    def validate_image(self, image, image_type="selfie") -> QualityResult:
        combined_metrics = {}

        for validator in self.image_validators:
            if isinstance(validator, BlurValidator):
                result = validator.validate(image, image_type=image_type)
            else:
                result = validator.validate(image)
            if result.metrics:
                combined_metrics.update(result.metrics)

            if not result.passed:
                return QualityResult(
                    passed=False,
                    code=result.code,
                    message=result.message,
                    metrics=combined_metrics,
                )

        return QualityResult(
            passed=True,
            metrics=combined_metrics,
        )

    def validate_face(self, image, face) -> QualityResult:
        combined_metrics = {}

        for validator in self.face_validators:
            result = validator.validate(image, face)
            if result.metrics:
                combined_metrics.update(result.metrics)

            if not result.passed:
                return QualityResult(
                    passed=False,
                    code=result.code,
                    message=result.message,
                    metrics=combined_metrics,
                )

        return QualityResult(
            passed=True,
            metrics=combined_metrics,
        )