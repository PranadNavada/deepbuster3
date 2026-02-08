"""
Placeholder Image Analysis API
This file provides placeholder responses until the image model and inference script are available.
Runs on port 8002
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_placeholder_image(text="Placeholder", width=400, height=300):
    """Create a placeholder image with text"""
    img = Image.new('RGB', (width, height), color=(50, 50, 70))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((width - text_width) // 2, (height - text_height) // 2)
    
    draw.text(position, text, fill=(200, 200, 200), font=font)
    
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Placeholder image analysis endpoint.
    Returns mock data until the actual model is available.
    """
    try:
        print(f"Received image: {file.filename}")
     
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    
        buffer = io.BytesIO()
        image.thumbnail((400, 300))
        image.save(buffer, format='PNG')
        buffer.seek(0)
        original_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        image_url = f"data:image/png;base64,{original_base64}"
        
        heatmap_img = Image.new('RGB', (400, 300), color=(20, 20, 40))
        draw = ImageDraw.Draw(heatmap_img)
        
        for _ in range(3):
            x = random.randint(50, 350)
            y = random.randint(50, 250)
            size = random.randint(30, 60)
            draw.ellipse([x-size, y-size, x+size, y+size], fill=(255, 100, 50, 180))
        
        buffer = io.BytesIO()
        heatmap_img.save(buffer, format='PNG')
        buffer.seek(0)
        heatmap_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        heatmap_url = f"data:image/png;base64,{heatmap_base64}"
        
        fake_probability = random.uniform(0.3, 0.8)
        
        regions = [
            "⚠️ Image model not available - showing placeholder data",
            "To enable real image analysis:",
            "1. Place your image_model.h5 file in deepbuster/src/",
            "2. Create inference_gradcam.py script",
            "3. Restart the image API server"
        ]
        
        return JSONResponse({
            "image_url": None,  # Return None to trigger base64 fallback
            "image_base64": image_url,
            "heatmap_url": None,
            "heatmap_base64": heatmap_url,
            "regions": regions,
            "fake_probability": fake_probability,
            "prediction": "PLACEHOLDER" if fake_probability > 0.5 else "REAL (PLACEHOLDER)"
        })
        
    except Exception as e:
        import traceback
        print(f"Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "Image API running (placeholder mode)",
        "note": "This is a placeholder. Add image_model.h5 and inference_gradcam.py for real analysis."
    }

@app.get("/")
async def root():
    return {
        "service": "DeepBuster Image Analysis API (Placeholder)",
        "status": "running",
        "endpoints": {
            "/api/analyze": "POST - Analyze image for AI generation",
            "/health": "GET - Health check"
        }
    }
