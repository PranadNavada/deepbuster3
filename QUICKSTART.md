# DeepBuster - Quick Start Guide

## Starting the Application

### Option 1: Start All Services at Once (Recommended)
```bash
cd /Users/pranad/Documents/deepbuster-1
./start_all.sh
```

### Option 2: Start Services Individually

1. **Start Audio Analysis API (Port 8000)**
```bash
cd /Users/pranad/Documents/deepbuster-1
source .venv/bin/activate
cd deepbuster/src
python -m uvicorn audio_api:app --host 127.0.0.1 --port 8000 --reload
```

2. **Start Image Analysis API (Port 8002)**
```bash
cd /Users/pranad/Documents/deepbuster-1
python3 start_image_api.py
```
Or manually:
```bash
cd /Users/pranad/Documents/deepbuster-1
source .venv/bin/activate
cd deepbuster/src
python -m uvicorn image_api:app --host 127.0.0.1 --port 8002 --reload
```

3. **Start Text Analysis API (Port 8001)**
```bash
cd /Users/pranad/Documents/deepbuster-1
source .venv/bin/activate
cd deepbuster/src/TextAI_test/GPTZero
python -m uvicorn api:app --host 127.0.0.1 --port 8001 --reload
```

4. **Start React Frontend (Port 3000)**
```bash
cd /Users/pranad/Documents/deepbuster-1/deepbuster
npm start
```

## Stopping Services

```bash
cd /Users/pranad/Documents/deepbuster-1
./stop_services.sh
```

Or manually:
```bash
lsof -ti:8000 | xargs kill -9  # Audio API
lsof -ti:8001 | xargs kill -9  # Text API
lsof -ti:8002 | xargs kill -9  # Image API
lsof -ti:3000 | xargs kill -9  # React Frontend
```

## API Endpoints

- **Audio Analysis**: http://localhost:8000
  - POST `/analyze-audio` - Upload audio file for analysis
  - GET `/health` - Health check

- **Image Analysis**: http://localhost:8002
  - POST `/api/analyze` - Upload image file for analysis
  - GET `/health` - Health check
  - GET `/uploads/{filename}` - Access uploaded images
  - GET `/outputs/{filename}` - Access generated heatmaps

- **Text Analysis**: http://localhost:8001
  - POST `/analyze` - Analyze text for AI generation
  - GET `/health` - Health check

- **React Frontend**: http://localhost:3000

## Required Files

### Models (in `deepbuster/src/`)
- ✅ `audio_detector_v2.h5` - Audio deepfake detection model
- ✅ `image_model.h5` - Image deepfake detection model
- ✅ `inference_gradcam.py` - Image inference script with Grad-CAM

### API Files
- ✅ `audio_api.py` - Audio analysis API
- ✅ `image_api.py` - Image analysis API (new, improved!)
- ✅ `TextAI_test/GPTZero/api.py` - Text analysis API

## Troubleshooting

### Port Already in Use
```bash
# Check what's using a port
lsof -ti:8000

# Kill process on port
lsof -ti:8000 | xargs kill -9
```

### Virtual Environment Issues
```bash
# Recreate virtual environment
cd /Users/pranad/Documents/deepbuster-1
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Model Not Found
- Ensure `audio_detector_v2.h5` and `image_model.h5` are in `deepbuster/src/`
- Check file permissions: `chmod 644 deepbuster/src/*.h5`

### CORS Errors
- Ensure all APIs have CORS middleware configured
- Check that frontend is connecting to correct ports (8000, 8001, 8002)

## Testing APIs

```bash
# Test Audio API
curl http://localhost:8000/health

# Test Image API
curl http://localhost:8002/health

# Test Text API
curl http://localhost:8001/

# Test with file upload
curl -X POST -F "file=@test.wav" http://localhost:8000/analyze-audio
curl -X POST -F "file=@test.jpg" http://localhost:8002/api/analyze
```
