from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
import tempfile
import librosa
import os

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "audio_detector_v2.h5")
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

def audio_to_spectrogram(audio_file_path):
    """Convert audio file to mel-spectrogram"""
    
    y, sr = librosa.load(audio_file_path, sr=16000, duration=3.0)
    
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=96, 
        n_fft=512, 
        hop_length=256
    )
    
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def process_spectrogram(spec):
    """Preprocess spectrogram to match training shape"""
    s = np.array(spec)
    if s.ndim == 2:
        if s.shape[0] < s.shape[1] and s.shape[0] < 96:
            s = s.T
        s = s[:96, :64]
        if s.shape[0] < 96:
            s = np.pad(s, ((0, 96-s.shape[0]), (0, 0)), mode="constant")
        if s.shape[1] < 64:
            s = np.pad(s, ((0, 0), (0, 64-s.shape[1])), mode="constant")
        s = (s - np.mean(s)) / (np.std(s) + 1e-8)
    return s

def generate_gradcam(model, input_tensor, spec):
    """Generate GradCAM heatmap"""
    last_conv_layer = model.get_layer("last_conv")
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
    
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(input_tensor)
        if isinstance(preds, list):
            preds = preds[0]
        loss = preds[:, 0]
    
    grads = tape.gradient(loss, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_output = last_conv_output[0].numpy()
    heatmap = last_conv_output @ pooled_grads.numpy()[..., np.newaxis]
    heatmap = np.squeeze(np.maximum(heatmap, 0))
    heatmap /= (np.max(heatmap) + 1e-8)
    
    return heatmap, float(preds[0])

def create_image_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

def analyze_heatmap_colors(heatmap):
    """
    Analyze the heatmap to determine blue vs red distribution for audio
    Returns: (red_ratio, blue_ratio, is_fake, percentage)
    
    For audio:
    - If blue < red → Real, percentage 0-50%
    - If red < blue → Fake, percentage 50-100%
    """
    blue_threshold = 0.35
    red_threshold = 0.65
    
    total_pixels = heatmap.size
    blue_pixels = np.sum(heatmap < blue_threshold)
    red_pixels = np.sum(heatmap > red_threshold)
    
    blue_ratio = blue_pixels / total_pixels
    red_ratio = red_pixels / total_pixels
    
    is_fake = red_ratio < blue_ratio
    

    if is_fake:
       
        if blue_ratio + red_ratio > 0:
            ratio = blue_ratio / (blue_ratio + red_ratio)
            percentage = int(50 + (ratio * 50)) 
        else:
            percentage = 50
    else:
        if blue_ratio + red_ratio > 0:
            ratio = blue_ratio / (blue_ratio + red_ratio)
            percentage = int(ratio * 50) 
        else:
            percentage = 50
    
    print(f"Audio heatmap analysis - Red: {red_ratio:.2%}, Blue: {blue_ratio:.2%}, Classification: {'Fake' if is_fake else 'Real'}, Percentage: {percentage}%")
    
    return red_ratio, blue_ratio, is_fake, percentage

@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}")
        
        model = load_model()
        print(f"Model loaded successfully")
        
        contents = await file.read()
        print(f"File size: {len(contents)} bytes")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        print(f"Temporary file created: {tmp_path}")
        
        spec = audio_to_spectrogram(tmp_path)
        print(f"Spectrogram shape: {spec.shape}")
        Path(tmp_path).unlink()
        
        processed_spec = process_spectrogram(spec)
        print(f"Processed spectrogram shape: {processed_spec.shape}")
        input_tensor = processed_spec[np.newaxis, ..., np.newaxis]
        
        heatmap, score = generate_gradcam(model, input_tensor, processed_spec)
        print(f"GradCAM generated, raw score: {score}")
        
        red_ratio, blue_ratio, is_fake, percentage = analyze_heatmap_colors(heatmap)
        
        label = "Fake" if is_fake else "Real"
        fake_probability = percentage / 100.0
        
        print(f"Final classification - Label: {label}, Percentage: {percentage}%")
        
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.imshow(processed_spec.T, aspect="auto", origin="lower", cmap="magma")
        ax1.set_xlabel("Time Frames")
        ax1.set_ylabel("Mel Bins")
        ax1.set_title("Audio Spectrogram")
        spectrogram_img = create_image_base64(fig1)
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.imshow(processed_spec.T, aspect="auto", origin="lower", cmap="magma")
        heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (96, 64)).numpy()
        heatmap_resized = np.squeeze(heatmap_resized, axis=-1)
        ax2.imshow(heatmap_resized.T, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
        ax2.set_xlabel("Time Frames")
        ax2.set_ylabel("Mel Bins")
        ax2.set_title("GradCAM Heat Map")
        heatmap_img = create_image_base64(fig2)
        
        if is_fake:
            description = [
                f"Audio classified as: {label}",
                f"Fake probability: {percentage}%",
                f"Blue regions ({blue_ratio:.1%}) dominate red regions ({red_ratio:.1%})",
                "High activation areas indicate suspicious patterns",
                "Analysis based on mel-spectrogram GradCAM"
            ]
        else:
            description = [
                f"Audio classified as: {label}",
                f"Fake probability: {percentage}%",
                f"Red regions ({red_ratio:.1%}) dominate blue regions ({blue_ratio:.1%})",
                "Low suspicious pattern activation detected",
                "Analysis based on mel-spectrogram GradCAM"
            ]
        
        print(f"Analysis complete. Label: {label}, Percentage: {percentage}%")
        
        return JSONResponse({
            "spectrogram": spectrogram_img,
            "heatmap": heatmap_img,
            "description": description,
            "percentage": percentage,
            "score": fake_probability,
            "label": label
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR: {error_details}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}