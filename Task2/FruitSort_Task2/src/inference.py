"""Load checkpoint và dự đoán một ảnh PIL; có thể gọi từ Task 3."""
from pathlib import Path
import torch
from PIL import Image, ImageOps
from src.model import FruitCNN
from src.data_utils import make_transform


def read_checkpoint(path='artifacts/model.pth'):
    # Chỉ load file checkpoint từ nguồn đáng tin cậy.
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:  # PyTorch cũ chưa hỗ trợ tham số weights_only
        return torch.load(path, map_location='cpu')


def load_model(path='artifacts/model.pth'):
    if not Path(path).exists():
        raise FileNotFoundError(f'Chưa có checkpoint: {path}')
    saved = read_checkpoint(path)
    mapping = saved['class_to_idx']
    idx_to_class = {idx: name for name, idx in mapping.items()}
    model = FruitCNN(n_classes=len(mapping))
    model.load_state_dict(saved['model_state'])
    model.eval()
    transform = make_transform(False, saved['image_size'], saved['mean'], saved['std'])
    return model, idx_to_class, transform


def predict_pil(model, idx_to_class, transform, image: Image.Image):
    rgb = ImageOps.exif_transpose(image).convert('RGB')
    x = transform(rgb).unsqueeze(0)
    with torch.no_grad():
        scores = torch.softmax(model(x), dim=1)[0]
    index = int(scores.argmax())
    return idx_to_class[index], float(scores[index])  # score không phải xác suất đã hiệu chuẩn
