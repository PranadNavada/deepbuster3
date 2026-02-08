#!/usr/bin/env python3
"""
Test audio API with heatmap analysis
"""
import requests
from pathlib import Path

# Use test audio file
test_audio = Path("/Users/pranad/Documents/deepbuster-1/.venv/lib/python3.11/site-packages/gradio/test_data/test_audio.wav")

if not test_audio.exists():
    print(f"Test audio not found: {test_audio}")
    exit(1)

print(f"Testing Audio API with: {test_audio.name}")
print("=" * 60)

url = "http://localhost:8000/analyze-audio"

with open(test_audio, "rb") as f:
    files = {"file": (test_audio.name, f, "audio/wav")}
    response = requests.post(url, files=files)

if response.status_code == 200:
    data = response.json()
    print("✅ API Response successful!")
    print(f"Prediction: {data.get('label', 'N/A')}")
    print(f"Score: {data.get('score', 'N/A'):.4f}")
    print(f"Percentage: {data.get('percentage', 'N/A')}%")
    print(f"\nDescription:")
    for desc in data.get('description', []):
        print(f"  - {desc}")
    print(f"\nHas spectrogram: {'spectrogram' in data}")
    print(f"Has heatmap: {'heatmap' in data}")
else:
    print(f"❌ API Error: {response.status_code}")
    print(response.text)
