#!/usr/bin/env python3
"""
Test script for both fake and real images
"""
import requests
from pathlib import Path

# Test images
test_images = [
    Path("/Users/pranad/Documents/deepbuster-1/data/test/FAKE/0 (10).jpg"),
    Path("/Users/pranad/Documents/deepbuster-1/data/test/REAL/0000 (10).jpg"),
]

url = "http://localhost:8002/api/analyze"

for test_image in test_images:
    if not test_image.exists():
        print(f"Test image not found: {test_image}")
        continue
    
    print(f"\n{'='*60}")
    print(f"Testing: {test_image.parent.name}/{test_image.name}")
    print('='*60)
    
    with open(test_image, "rb") as f:
        files = {"file": (test_image.name, f, "image/jpeg")}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Prediction: {data.get('prediction', 'N/A')}")
        print(f"   Score: {data.get('score', 'N/A'):.6f}")
        print(f"   Fake Probability: {data.get('fake_probability', 'N/A'):.6f}")
        print(f"   Percentage: {data.get('percentage', 'N/A')}%")
        print(f"   Has heatmap: {bool(data.get('heatmap_base64'))}")
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)

print(f"\n{'='*60}")
print("✅ All tests completed!")
