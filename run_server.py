"""
AI CCTV Platform - Server Runner
Launches the FastAPI backend and Web UI on localhost:8000.
"""

import uvicorn
from src.api.app import app

if __name__ == "__main__":
    print("=" * 65)
    print("  AI CCTV Platform - SOC Video Analytics Dashboard Server")
    print("  URL: http://127.0.0.1:8000")
    print("=" * 65)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
