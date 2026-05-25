import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import uvicorn
from core.config import settings

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=True, app_dir="src")