#!/usr/bin/env python3
"""
Audio API Server
Runs on port 8000
"""
import uvicorn
import sys
import os
from pathlib import Path

# Change to the src directory
os.chdir(str(Path(__file__).parent / "deepbuster" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "deepbuster" / "src"))

if __name__ == "__main__":
    print("🔊 Starting Audio Analysis API on http://localhost:8000")
    print("📁 Loading audio model from deepbuster/src/audio_detector_v2.h5")
    
    uvicorn.run(
        "audio_api:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )
