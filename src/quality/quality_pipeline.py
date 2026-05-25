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
        self._image_validators = [BlurValidator(), BrightnessValidator(), GlareValidator()]
        self._face_validators = [FaceSizeValidator(), OcclusionValidator()]

    def validate_image(self, image, image_type: str = "selfie") -> QualityResult:
        combined = {}
        for v in self._image_validators:
            # BlurValidator needs image_type, others ignore it
            if hasattr(v, "validate") and "image_type" in v.validate.__code__.co_varnames:
                res = v.validate(image, image_type=image_type)
            else:
                res = v.validate(image)
            if res.metrics:
                combined.update(res.metrics)
            if not res.passed:
                return QualityResult(passed=False, code=res.code, message=res.message, metrics=combined)
        return QualityResult(passed=True, metrics=combined)

    def validate_face(self, image, face) -> QualityResult:
        combined = {}
        for v in self._face_validators:
            res = v.validate(image, face=face)
            if res.metrics:
                combined.update(res.metrics)
            if not res.passed:
                return QualityResult(passed=False, code=res.code, message=res.message, metrics=combined)
        return QualityResult(passed=True, metrics=combined)