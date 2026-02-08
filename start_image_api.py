#!/usr/bin/env python3
"""
Image API Server
Runs on port 8002
"""
import uvicorn
import sys
import os
from pathlib import Path

# Change to the src directory
os.chdir(str(Path(__file__).parent / "deepbuster" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "deepbuster" / "src"))

if __name__ == "__main__":
    print("🖼️  Starting Image Analysis API on http://localhost:8002")
    
    # Check if real model exists
    model_path = Path(__file__).parent / "deepbuster" / "src" / "image_model.h5"
    if model_path.exists():
        print("📁 Using image model: image_model.h5")
        module = "image_api:app"
    else:
        print("⚠️  Image model not found, using placeholder API")
        print("💡 To enable real analysis, add image_model.h5 to deepbuster/src/")
        module = "image_api_placeholder:app"
    
    uvicorn.run(
        module,
        host="127.0.0.1",
        port=8002,
        reload=True,
        log_level="info"
    )
