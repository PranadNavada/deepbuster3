#!/bin/bash
# Stop all DeepBuster services

echo "🛑 Stopping DeepBuster Services..."

# Kill processes on ports
echo "Stopping Audio API (port 8000)..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || echo "  No process on port 8000"

echo "Stopping Text API (port 8001)..."
lsof -ti:8001 | xargs kill -9 2>/dev/null || echo "  No process on port 8001"

echo "Stopping Image API (port 8002)..."
lsof -ti:8002 | xargs kill -9 2>/dev/null || echo "  No process on port 8002"

echo "Stopping React Frontend (port 3000)..."
lsof -ti:3000 | xargs kill -9 2>/dev/null || echo "  No process on port 3000"

echo ""
echo "✅ All services stopped"
