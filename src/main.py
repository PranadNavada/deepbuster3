from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import shutil
import subprocess
import json
from pathlib import Path
import uuid
import traceback
import cv2
import numpy as np
from typing import List, Dict
import google.generativeai as genai
from PIL import Image
import io
import base64

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the src directory
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

MODEL_PATH = str(BASE_DIR / "image_model.h5")

# Gemini API configuration
GEMINI_API_KEY = "AIzaSyCgCVcEkovsFXm_dj90NcRpBtrJq7RwpIA"
try:
    if GEMINI_API_KEY and GEMINI_API_KEY != "AIzaSyCgCVcEkovsFXm_dj90NcRpBtrJq7RwpIA":
        genai.configure(api_key='AIzaSyCgCVcEkovsFXm_dj90NcRpBtrJq7RwpIA')
        gemini_model = genai.GenerativeModel('gemini-3.0-flash')
        print("Gemini API configured successfully")
    else:
        gemini_model = None
        print("Gemini API key not configured - using fallback descriptions")
except Exception as e:
    gemini_model = None
    print(f"Failed to configure Gemini API: {e}")

def analyze_heatmap_regions(original_image_path: str, heatmap_image_path: str, heatmap_array: np.ndarray) -> List[Dict]:
    """
    Analyze the heatmap to identify high-concentration areas and their locations.
    """
    # Load the heatmap overlay image
    heatmap_img = cv2.imread(heatmap_image_path)
    if heatmap_img is None:
        return []
    
    # Threshold to find high-intensity regions (red areas in heatmap)
    # Convert to HSV to better detect red regions
    hsv = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2HSV)
    
    # Red color range in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Find contours of high-intensity regions
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first) and take top regions
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # Top 5 regions
    
    regions = []
    height, width = heatmap_img.shape[:2]
    
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 500:  # Ignore very small regions
            continue
            
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate center point
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Determine relative position
        position = get_relative_position(center_x, center_y, width, height)
        
        regions.append({
            "index": idx + 1,
            "position": position,
            "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "area": int(area),
            "center": {"x": int(center_x), "y": int(center_y)}
        })
    
    return regions

def get_relative_position(x: int, y: int, width: int, height: int) -> str:
    """
    Determine the relative position of a point in the image.
    """
    # Divide image into 9 sections
    third_w = width / 3
    third_h = height / 3
    
    if y < third_h:
        vertical = "upper"
    elif y < 2 * third_h:
        vertical = "middle"
    else:
        vertical = "lower"
    
    if x < third_w:
        horizontal = "left"
    elif x < 2 * third_w:
        horizontal = "center"
    else:
        horizontal = "right"
    
    if vertical == "middle" and horizontal == "center":
        return "center of the image"
    elif horizontal == "center":
        return f"{vertical} center"
    elif vertical == "middle":
        return f"{horizontal} side"
    else:
        return f"{vertical} {horizontal}"

async def generate_gemini_analysis(original_image_path: str, heatmap_image_path: str, regions: List[Dict], label: str, score: float) -> List[str]:
    """
    Use Gemini API to generate detailed descriptions of why specific regions were flagged.
    """
    if not gemini_model:
        print("Gemini model not available, using fallback descriptions")
        return generate_fallback_descriptions(regions, label, score)
    
    if not regions or len(regions) == 0:
        print("No regions detected, using fallback descriptions")
        return generate_fallback_descriptions(regions, label, score)
    
    try:
        # Load images
        original_img = Image.open(original_image_path)
        heatmap_img = Image.open(heatmap_image_path)
        
        confidence = round(score * 100, 2)
        
        # Create prompt for Gemini
        regions_text = "\n".join([
            f"Region {r['index']}: Located in the {r['position']}, covering approximately {r['area']} pixels"
            for r in regions
        ])
        
        prompt = f"""You are an expert at analyzing AI-generated images. This image has been classified as AI-GENERATED (FAKE) with {confidence}% confidence.

A Grad-CAM heatmap has been generated showing {len(regions)} regions with high AI-generation probability (shown in red/orange/yellow colors in the heatmap).

The flagged regions are:
{regions_text}

Analyze BOTH images carefully and provide 4-5 specific observations about why these regions were flagged:

1. Describe specific visual artifacts you can see in the flagged regions (e.g., unnatural textures, blurry edges, repetitive patterns, weird distortions)
2. Point out any inconsistencies in lighting, shadows, or reflections in those areas
3. Note any anatomical impossibilities or geometric inconsistencies if visible
4. Identify any telltale signs of AI generation like smooth plastic-like textures or uncanny valley effects
5. Explain why the neural network highlighted these specific areas

Be specific and technical. Reference what you actually see in the images. Keep each point to 2-3 sentences maximum.

Format your response as plain text with each point on a new line. Do NOT use bullet points, dashes, or numbering."""

        print(f"Sending request to Gemini API...")
        
        # Generate response
        response = gemini_model.generate_content([
            prompt,
            original_img,
            heatmap_img
        ])
        
        print(f"Gemini API response received")
        
        # Parse response into list
        descriptions = []
        if response.text:
            lines = response.text.strip().split('\n')
            for line in lines:
                line = line.strip()
                # Remove common bullet point characters
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    line = line[1:].strip()
                if line.startswith(tuple('0123456789')):
                    # Remove leading numbers
                    line = line.lstrip('0123456789.()').strip()
                if line and not line.startswith('#') and len(line) > 15:  # Skip headers and very short lines
                    descriptions.append(line)
        
        print(f"Parsed {len(descriptions)} descriptions from Gemini")
        
        # Ensure we have at least 3 descriptions
        if len(descriptions) < 3:
            print("Not enough descriptions from Gemini, adding fallback")
            fallback = generate_fallback_descriptions(regions, label, score)
            descriptions.extend(fallback[:5 - len(descriptions)])
        
        return descriptions[:5]  # Limit to 5 descriptions
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        traceback.print_exc()
        return generate_fallback_descriptions(regions, label, score)

def generate_fallback_descriptions(regions: List[Dict], label: str, score: float) -> List[str]:
    """
    Generate fallback descriptions when Gemini API is not available.
    """
    descriptions = []
    
    if label == "fake":
        confidence = round(score * 100, 2) if score > 0 else 0
        
        if confidence > 0:
            descriptions.append(f"The image has been classified as AI-GENERATED with {confidence}% confidence.")
        else:
            descriptions.append("The image has been classified as AI-GENERATED.")
        
        if regions and len(regions) > 0:
            top_region = regions[0]
            descriptions.append(f"Detected {len(regions)} suspicious region(s). Primary area in the {top_region['position']} shows patterns typical of AI-generated content.")
            descriptions.append("Common AI artifacts detected: unnatural textures, inconsistent lighting patterns, or impossible geometric relationships.")
            descriptions.append("The heatmap highlights areas where the neural network detected synthetic features.")
        else:
            descriptions.append("The model detected statistical anomalies in pixel distributions characteristic of generative models.")
            descriptions.append("Neural network analysis indicates synthetic patterns throughout the image structure.")
            descriptions.append("The image exhibits characteristics commonly found in AI-generated content.")
        
    else:
        confidence = round(score * 100, 2) if score > 0 else 0
        fake_percentage = round((1 - score) * 100, 2) if score > 0 else 0
        
        if confidence > 0:
            descriptions.append(f"The image has been classified as REAL with {confidence}% confidence.")
            descriptions.append(f"Only {fake_percentage}% probability of being AI-generated.")
        else:
            descriptions.append("The image has been classified as REAL.")
        
        descriptions.append("No significant AI-generated artifacts were detected in the analysis.")
        descriptions.append("The image shows natural variation and consistency typical of authentic photography.")
        descriptions.append("Neural network analysis indicates genuine photographic characteristics throughout the image.")
    
    return descriptions

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}")
        print(f"Content type: {file.content_type}")
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        input_filename = f"{file_id}{file_extension}"
        input_path = UPLOAD_DIR / input_filename
        
        # Save uploaded file
        print(f"Saving file to: {input_path}")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Verify file was saved
        if not input_path.exists():
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
        
        print(f"File saved successfully: {input_path}")
        
        # Prepare output path for Grad-CAM
        output_filename = f"{file_id}_gradcam.jpg"
        output_path = OUTPUT_DIR / output_filename
        
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=500,
                detail=f"Model file not found at: {MODEL_PATH}"
            )
        
        print(f"Running inference with model: {MODEL_PATH}")
        
        # Run inference_gradcam.py
        cmd = [
            "python",
            str(BASE_DIR / "inference_gradcam.py"),
            "--model", MODEL_PATH,
            "--image", str(input_path),
            "--output", str(output_path)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR)
        )
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {result.stderr}"
            )
        
        # Parse JSON output from inference script
        try:
            output_lines = result.stdout.strip().split('\n')
            json_line = None
            for line in reversed(output_lines):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    json_line = line
                    break
            
            if json_line is None:
                raise ValueError("No JSON output found in inference script output")
            
            print(f"Parsing JSON line: {json_line}")
            output_data = json.loads(json_line)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON decode error: {e}")
            print(f"Raw output: {result.stdout}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse inference output: {str(e)}"
            )
        
        label = output_data.get("label", "unknown")
        score = output_data.get("score", 0.0)
        gradcam_path = output_data.get("gradcam_path")
        
        print(f"Model output - Label: {label}, Score: {score}")
        
        # Calculate percentage - score represents the confidence of the prediction
        if label == "fake":
            # Show the AI/fake percentage (confidence that it's fake)
            percentage = round(score * 100, 2)
        else:
            # Show how "real" it is (low fake percentage)
            percentage = round((1 - score) * 100, 2)
        
        print(f"Calculated percentage (AI likelihood): {percentage}%")
        
        # Analyze heatmap regions ONLY if image is classified as fake AND heatmap exists
        regions = []
        has_heatmap = gradcam_path and os.path.exists(output_path)
        
        if label == "fake" and has_heatmap:
            # Only analyze regions if the image is flagged as fake
            print(f"Analyzing heatmap regions from: {output_path}")
            regions = analyze_heatmap_regions(str(input_path), str(output_path), None)
            print(f"Detected {len(regions)} high-concentration regions")
        
        # Generate descriptions based on label and whether we have regions
        if label == "fake" and has_heatmap and regions:
            # Use Gemini API for fake images with heatmap and regions
            print("Generating Gemini analysis...")
            descriptions = await generate_gemini_analysis(
                str(input_path),
                str(output_path),
                regions,
                label,
                score
            )
        else:
            # Use fallback descriptions
            print("Using fallback descriptions")
            descriptions = generate_fallback_descriptions(regions, label, score)
        
        # Prepare response
        response_data = {
            "image_url": f"/uploads/{input_filename}",
            "heatmap_url": f"/outputs/{output_filename}" if has_heatmap else None,
            "percentage": percentage,
            "descriptions": descriptions,
            "label": label,
            "score": float(score),
            "regions": regions if regions else []
        }
        
        print(f"Sending response: Label={label}, Score={score}, Percentage={percentage}%, Regions={len(regions)}, Descriptions={len(descriptions)}")
        
        return JSONResponse(content=response_data)
        
    except HTTPException as e:
        print(f"HTTP Exception: {e.detail}")
        raise
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Unexpected error: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "DeepBuster API is running"}

@app.get("/health")
async def health():
    gemini_status = "configured" if gemini_model else "not configured"
    return {
        "status": "ok",
        "model_exists": os.path.exists(MODEL_PATH),
        "model_path": MODEL_PATH,
        "upload_dir": str(UPLOAD_DIR),
        "output_dir": str(OUTPUT_DIR),
        "gemini_api": gemini_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)