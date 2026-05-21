from fastapi import APIRouter, UploadFile, File, HTTPException

from controllers import VerificationController


router = APIRouter(
    prefix="/api/v1",
    tags=["verification"]
)

controller = VerificationController()


@router.post("/verify")
async def verify(
    id_image: UploadFile = File(...),
    selfie_image: UploadFile = File(...)
):

    try:

        print("========== REQUEST RECEIVED ==========")

        id_bytes = await id_image.read()
        selfie_bytes = await selfie_image.read()

        print("Files read successfully")

        result = controller.verify_from_bytes(
            id_bytes=id_bytes,
            selfie_bytes=selfie_bytes
        )

        print("FINAL RESULT:", result)

        return result

    except Exception as e:

        print("ERROR:", str(e))

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )