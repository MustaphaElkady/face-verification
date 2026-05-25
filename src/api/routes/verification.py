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
        return controller.verify_from_bytes(await id_image.read(), await selfie_image.read())
    except (ImageValidationError, NoFaceDetectedError, DuplicateImageError) as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail={"error": str(e)})
    except QualityCheckError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail={"error": e.message, "code": e.code, "metrics": e.metrics})
    except EmbeddingExtractionError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Embedding extraction failed.")
    except FaceVerificationError as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error.")