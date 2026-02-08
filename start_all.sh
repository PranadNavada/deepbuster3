#!/bin/bash
# Master launcher for DeepBuster
# Starts all backend APIs and the React frontend

echo "🚀 Starting DeepBuster Services..."
echo "================================="

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  Warning: Virtual environment not found at .venv/"
fi

# Kill any existing processes on our ports
echo "🧹 Cleaning up existing services..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8001 | xargs kill -9 2>/dev/null || true
lsof -ti:8002 | xargs kill -9 2>/dev/null || true
sleep 1

# Start Audio API (port 8000)
echo "🔊 Starting Audio Analysis API on port 8000..."
python3 start_audio_api.py > logs/audio_api.log 2>&1 &
AUDIO_PID=$!
echo "   PID: $AUDIO_PID"

# Start Image API (port 8002)
echo "🖼️  Starting Image Analysis API on port 8002..."
python3 start_image_api.py > logs/image_api.log 2>&1 &
IMAGE_PID=$!
echo "   PID: $IMAGE_PID"

# Start Text API (port 8001)
echo "📝 Starting Text Analysis API on port 8001..."
cd deepbuster/src/TextAI_test/GPTZero
python3 -m uvicorn api:app --reload --port 8001 > ../../../../logs/text_api.log 2>&1 &
TEXT_PID=$!
echo "   PID: $TEXT_PID"
cd "$SCRIPT_DIR"

# Wait a moment for APIs to start
sleep 3

echo ""
echo "================================="
echo "✅ Backend APIs Started"
echo "================================="
echo "Audio API:  http://localhost:8000"
echo "Text API:   http://localhost:8001"
echo "Image API:  http://localhost:8002"
echo ""
echo "Process IDs:"
echo "  Audio: $AUDIO_PID"
echo "  Text:  $TEXT_PID"
echo "  Image: $IMAGE_PID"
echo ""
echo "Logs are being written to logs/ directory"
echo ""
echo "To stop all services, run: ./stop_services.sh"
echo "Or manually kill processes: kill $AUDIO_PID $TEXT_PID $IMAGE_PID"
echo ""
echo "================================="
echo "🌐 Starting React Frontend..."
echo "================================="
cd deepbuster
npm start
