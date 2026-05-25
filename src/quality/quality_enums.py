from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    code: str | None = None
    message: str | None = None
    metrics: dict | None = None
