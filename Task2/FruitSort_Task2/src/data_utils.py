"""Định nghĩa tiền xử lý thống nhất giữa train, validation, test và inference."""
from torchvision import transforms

IMAGE_SIZE = 128
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
SEED = 42


def make_transform(train: bool = False, size: int = IMAGE_SIZE,
                   mean=MEAN, std=STD):
    operations = [transforms.Resize((size, size))]
    if train:
        operations.extend([
            transforms.RandomRotation(12),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ])
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transforms.Compose(operations)
