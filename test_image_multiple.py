#!/usr/bin/env python3
"""
Test multiple images to see heatmap distribution
"""
import requests
from pathlib import Path

# Test multiple images
fake_dir = Path("/Users/pranad/Documents/deepbuster-1/data/test/FAKE")
real_dir = Path("/Users/pranad/Documents/deepbuster-1/data/test/REAL")

fake_images = list(fake_dir.glob("*.jpg"))[:5]
real_images = list(real_dir.glob("*.jpg"))[:5]

url = "http://localhost:8002/api/analyze"

print("\n" + "="*70)
print("TESTING FAKE IMAGES")
print("="*70)

for img_path in fake_images:
    with open(img_path, "rb") as f:
        files = {"file": (img_path.name, f, "image/jpeg")}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n{img_path.name:30} → {data['prediction']:5} | {data['percentage']:3}% fake")
    else:
        print(f"\n{img_path.name:30} → ERROR {response.status_code}")

print("\n" + "="*70)
print("TESTING REAL IMAGES")
print("="*70)

for img_path in real_images:
    with open(img_path, "rb") as f:
        files = {"file": (img_path.name, f, "image/jpeg")}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n{img_path.name:30} → {data['prediction']:5} | {data['percentage']:3}% fake")
    else:
        print(f"\n{img_path.name:30} → ERROR {response.status_code}")

print("\n" + "="*70)
