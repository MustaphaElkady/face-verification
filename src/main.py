from fastapi import FastAPI
from routes.verification import router as verification_router

app = FastAPI(title="Face Verification API")
app.include_router(verification_router)