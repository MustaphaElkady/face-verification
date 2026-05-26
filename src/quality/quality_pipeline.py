from dataclasses import dataclass, field
import inspect

from quality.validators.blur_validator import BlurValidator
from quality.validators.brightness_validator import BrightnessValidator
from quality.validators.face_size_validator import FaceSizeValidator
from quality.validators.glare_validator import GlareValidator
from quality.validators.occlusion_validator import OcclusionValidator
from quality.validators.document_validator import DocumentValidator
from quality.validators.selfie_validator import SelfieValidator


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    code: str | None = None
    message: str | None = None
    metrics: dict = field(default_factory=dict)


class QualityPipeline:
    def __init__(self):
        self._image_validators = [
            BlurValidator(),
            BrightnessValidator(),
            GlareValidator(),
            DocumentValidator(),
            SelfieValidator(),
        ]

        self._face_validators = [
            FaceSizeValidator(),
            OcclusionValidator(),
            SelfieValidator(),
        ]

    def _run_validator(self, validator, image, face=None, image_type: str = "selfie"):
        """
        Some validators accept only:
            validate(image)

        Some accept:
            validate(image, face)

        New validators accept:
            validate(image, face=None, image_type="selfie")

        This helper passes only the arguments supported by each validator.
        """
        signature = inspect.signature(validator.validate)
        params = signature.parameters

        kwargs = {}

        if "face" in params and face is not None:
            kwargs["face"] = face

        if "image_type" in params:
            kwargs["image_type"] = image_type

        return validator.validate(image, **kwargs)

    def validate_image(self, image, image_type: str = "selfie") -> QualityResult:
        combined_metrics = {}

        for validator in self._image_validators:
            result = self._run_validator(
                validator,
                image,
                image_type=image_type,
            )

            if getattr(result, "metrics", None):
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

    def validate_face(self, image, face, image_type: str = "selfie") -> QualityResult:
        combined_metrics = {}

        for validator in self._face_validators:
            result = self._run_validator(
                validator,
                image,
                face=face,
                image_type=image_type,
            )

            if getattr(result, "metrics", None):
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