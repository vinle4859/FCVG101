"""Thử inference một ảnh: python predict.py --image path/to/photo.jpg"""
import argparse
from PIL import Image
from src.inference import load_model, predict_pil

p = argparse.ArgumentParser()
p.add_argument('--image', required=True)
args = p.parse_args()
model, idx_to_class, transform = load_model()
with Image.open(args.image) as img:
    label, score = predict_pil(model, idx_to_class, transform, img)
print(f'Predicted: {label}; model score: {score:.3f}')
