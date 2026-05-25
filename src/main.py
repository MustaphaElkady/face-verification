from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.verification import router

app = FastAPI(title="Face Verification API v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST"], allow_headers=["*"])
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}