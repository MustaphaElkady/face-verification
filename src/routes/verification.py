from fastapi import APIRouter, File, UploadFile, HTTPException, status
from controllers.VerificationController import VerificationController
from api.schemas.verification import VerificationResponse, ErrorResponse
from core.exceptions import (
    ImageValidationError, QualityCheckError, NoFaceDetectedError,
    DuplicateImageError, EmbeddingExtractionError, FaceVerificationError
)

router = APIRouter(prefix="/api/v2", tags=["verification"])
controller = VerificationController()

@router.post("/verify", response_model=VerificationResponse)
async def verify(id_image: UploadFile = File(...), selfie_image: UploadFile = File(...)):
    try:
        result = controller.verify_from_bytes(await id_image.read(), await selfie_image.read())
        return result
    except (ImageValidationError, NoFaceDetectedError, DuplicateImageError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail={"error": str(exc)})
    except QualityCheckError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail={"error": exc.message, "metrics": exc.metrics})
    except EmbeddingExtractionError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Embedding extraction failed.")
    except Exception as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error.")