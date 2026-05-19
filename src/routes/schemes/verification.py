from pydantic import BaseModel


class VerificationResponse(BaseModel):
    verified: bool
    similarity: float