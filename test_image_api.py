#!/usr/bin/env python3
"""
Quick test script for the image API
"""
import requests
from pathlib import Path

# Test image path
test_image = Path("/Users/pranad/Documents/deepbuster-1/data/test/FAKE/0 (10).jpg")

if not test_image.exists():
    print(f"Test image not found: {test_image}")
    exit(1)

# Test the API
print(f"Testing Image API with: {test_image.name}")
print("=" * 60)

url = "http://localhost:8002/api/analyze"

with open(test_image, "rb") as f:
    files = {"file": (test_image.name, f, "image/jpeg")}
    response = requests.post(url, files=files)

if response.status_code == 200:
    data = response.json()
    print("✅ API Response successful!")
    print(f"Prediction: {data.get('prediction', 'N/A')}")
    print(f"Score: {data.get('score', 'N/A'):.4f}")
    print(f"Fake Probability: {data.get('fake_probability', 'N/A'):.4f}")
    print(f"Percentage: {data.get('percentage', 'N/A')}%")
    print(f"\nRegions/Description:")
    for region in data.get('regions', []):
        print(f"  - {region}")
    print(f"\nHas image_base64: {'image_base64' in data}")
    print(f"Has heatmap_base64: {'heatmap_base64' in data}")
    if 'image_base64' in data and data['image_base64']:
        print(f"Image base64 length: {len(data['image_base64'])} chars")
    if 'heatmap_base64' in data and data['heatmap_base64']:
        print(f"Heatmap base64 length: {len(data['heatmap_base64'])} chars")
else:
    print(f"❌ API Error: {response.status_code}")
    print(response.text)
