from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
import tempfile
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
MODEL_PATH = os.path.join(SCRIPT_DIR, "image_model.h5")
model = None

def load_model():
    global model
    if model is None:
        print(f"Loading model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully")
    return model

def preprocess_image(img_path, target_size=(224, 224), max_side=1024):
    """Preprocess image for model inference"""
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    h, w = img.shape[:2]
    if max_side and max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    img = cv2.resize(img, (int(target_size[1]), int(target_size[0])), interpolation=cv2.INTER_AREA)
    
    batch = np.expand_dims(img.astype(np.float32), axis=0)
    
    return batch / 127.5 - 1.0, img

def find_last_conv_layer(model):
    """Find the last Conv2D layer in the model"""
    from tensorflow.keras import layers
    
    def _iter_layers(module):
        for layer in getattr(module, "layers", []):
            yield layer
            if hasattr(layer, "layers") and layer.layers:
                yield from _iter_layers(layer)
    
    for layer in reversed(list(_iter_layers(model))):
        if isinstance(layer, layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found in model.")

def _build_grad_model(model, conv_layer_obj):
    """Build a functional graph for GradCAM with nested models"""
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if not shape or len(shape) < 4:
        raise ValueError("Unsupported model input shape.")

    inp = tf.keras.Input(shape=shape[1:], name="gradcam_input")

    parent = None
    for layer in model.layers:
        if hasattr(layer, "layers") and layer.layers:
            try:
                nested = []
                for l in layer.layers:
                    nested.append(l)
                    if hasattr(l, 'layers') and l.layers:
                        nested.extend(l.layers)
            except Exception:
                nested = []
            if conv_layer_obj in nested or any(l.name == conv_layer_obj.name for l in nested):
                parent = layer
                break

    if parent is None:
        try:
            conv_out = tf.keras.Model(inputs=model.inputs, outputs=conv_layer_obj.output)(inp)
            x = inp
            for layer in model.layers:
                x = layer(x)
            return tf.keras.Model(inputs=inp, outputs=[conv_out, x])
        except:
            pass

   
    base = parent
    base_cam = tf.keras.Model(
        inputs=base.input,
        outputs=[base.get_layer(conv_layer_obj.name).output, base.output]
    )
    conv_out, base_out = base_cam(inp)

    try:
        base_idx = model.layers.index(base)
    except ValueError:
        base_idx = 0
    x = base_out
    for layer in model.layers[base_idx + 1:]:
        x = layer(x)

    return tf.keras.Model(inputs=inp, outputs=[conv_out, x])


def compute_gradcam(model, img_array, conv_layer_obj):
    """Generate GradCAM heatmap using proper nested model support"""
    grad_model = _build_grad_model(model, conv_layer_obj)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            target_class_index = int(tf.argmax(predictions[0]))
            loss = predictions[:, target_class_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients are None; cannot compute Grad-CAM.")
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_outputs = conv_outputs.numpy()[0]
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()
    return heatmap

def create_overlay_image(original_img, heatmap, alpha=0.45):
    """Create overlay of heatmap on original image"""
    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    
    return overlay

def image_to_base64(img_array):
    """Convert image array to base64 string"""
    _, buffer = cv2.imencode('.png', img_array)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def create_matplotlib_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

def analyze_heatmap_colors(heatmap):
    """
    Analyze the heatmap to determine blue vs other colors (red, orange, yellow) distribution
    Returns: (other_ratio, blue_ratio, is_fake)
    """
   
    
    blue_threshold = 0.35
    
    total_pixels = heatmap.size
    blue_pixels = np.sum(heatmap < blue_threshold)
    other_pixels = np.sum(heatmap >= blue_threshold)
    
    blue_ratio = blue_pixels / total_pixels
    other_ratio = other_pixels / total_pixels
    
   
    is_fake = other_ratio > blue_ratio
    
    print(f"Heatmap analysis - Blue: {blue_ratio:.2%}, Other colors (warm): {other_ratio:.2%}, Classification: {'Fake' if is_fake else 'Real'}")
    
    return other_ratio, blue_ratio, is_fake

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    tmp_path = None
    try:
        print(f"Received file: {file.filename}")
        
     
        model = load_model()
        
        
        contents = await file.read()
        print(f"File size: {len(contents)} bytes")
        

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        print(f"Temporary file created: {tmp_path}")

        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        target_size = (int(shape[1]), int(shape[2])) if shape and len(shape) >= 3 else (224, 224)
        print(f"Target size: {target_size}")
        

        img_array, original_img = preprocess_image(tmp_path, target_size=target_size)
        print(f"Image preprocessed, array shape: {img_array.shape}")
        

        predictions = model.predict(img_array, verbose=0)
        print(f"Predictions: {predictions}")
        

        if predictions.ndim == 1:
            fake_probability = float(predictions[0])
        elif predictions.shape[-1] == 1:
            fake_probability = float(predictions[0][0])
        elif predictions.shape[-1] == 2:

            fake_probability = float(predictions[0][0])
            real_probability = float(predictions[0][1])
        else:
   
            fake_probability = float(predictions[0][0])

        label = "Fake" if fake_probability >= 0.5 else "Real"
        percentage = int(fake_probability * 100)
        score = fake_probability  
        
        print(f"Initial classification - Label: {label}, Fake probability: {fake_probability:.6f}")
        
        heatmap_img = None
        regions = []
        heatmap_adjustment_applied = False
        
    
        try:
            conv_layer = find_last_conv_layer(model)
            print(f"Using conv layer: {conv_layer.name}")
            
            heatmap = compute_gradcam(model, img_array, conv_layer)
            print(f"GradCAM computed, heatmap shape: {heatmap.shape}")
            other_ratio, blue_ratio, is_fake = analyze_heatmap_colors(heatmap)
        
            calculated_percentage = int(other_ratio * 100)
            
            if is_fake:
                label = "Fake"
                percentage = calculated_percentage
                if percentage < 51:
                    percentage = 51
                fake_probability = percentage / 100.0
                heatmap_adjustment_applied = True
                print(f"Heatmap adjustment: More warm colors ({other_ratio:.2%}) detected, adjusting to Fake ({percentage}%)")
            else:

                label = "Real"
                percentage = calculated_percentage

                if percentage > 49:
                    percentage = 49
                fake_probability = percentage / 100.0
                heatmap_adjustment_applied = True
                print(f"Heatmap adjustment: More blue ({blue_ratio:.2%}) detected, adjusting to Real ({percentage}%)")
            

            overlay = create_overlay_image(original_img, heatmap)
            heatmap_img = image_to_base64(overlay)
            
            if label == "Fake":
                regions = [
                    f"Image classified as: {label}",
                    f"Fake probability: {fake_probability:.2%}",
                    "Red/yellow regions indicate suspicious areas",
                    "Blue regions show less suspicious areas",
                    "Analysis based on GradCAM heat distribution" if heatmap_adjustment_applied else "Analysis based on deep learning features"
                ]
            else:
                regions = [
                    f"Image classified as: {label}",
                    f"Fake probability: {fake_probability:.2%}",
                    "Minimal suspicious patterns detected",
                    "Blue-dominant heatmap indicates authenticity",
                    "Image appears to be authentic"
                ]
                
        except Exception as e:
            print(f"GradCAM generation failed: {e}")
            import traceback
            traceback.print_exc()

            heatmap_img = image_to_base64(original_img)
            regions = [
                f"Image classified as: {label}",
                f"Fake probability: {fake_probability:.2%}",
                "GradCAM analysis unavailable for this image"
            ]
        
     
        image_base64 = image_to_base64(original_img)
        
        score = fake_probability  
        
        print(f"Final analysis - Label: {label}, Fake probability: {fake_probability:.6f}, Percentage: {percentage}%")
        
        return JSONResponse({
            "image_base64": image_base64,
            "heatmap_base64": heatmap_img,
            "regions": regions,
            "fake_probability": float(fake_probability),
            "prediction": label,
            "score": float(score),
            "percentage": percentage
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR: {error_details}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

@app.get("/health")
async def health_check():
    try:
        load_model()
        return {
            "status": "ok",
            "model_loaded": True,
            "model_path": MODEL_PATH
        }
    except Exception as e:
        return {
            "status": "error",
            "model_loaded": False,
            "error": str(e)
        }

@app.get("/")
async def root():
    return {
        "service": "DeepBuster Image Analysis API",
        "status": "running",
        "endpoints": {
            "/api/analyze": "POST - Analyze image for AI generation",
            "/health": "GET - Health check"
        }
    }
