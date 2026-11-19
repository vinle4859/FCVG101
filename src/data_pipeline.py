"""FCVG101 Coursework - Task 2.1: Data Pipeline, EDA & Leak-Free Dataset Preparation.
Performs data ingestion from zip, rigorous EDA (class distribution, duplicate detection via SHA-256),
canonical 10-class normalization, and leak-free stratified splitting (Train/Val/Test 70/15/15) before augmentation.
"""
from pathlib import Path
from collections import defaultdict, Counter
import csv
import json
import hashlib
import shutil
import zipfile
from typing import Dict, List, Tuple
from PIL import Image, ImageOps
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGE_SIZE = 128
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEED = 42

# Canonical 10 Fruit Classes mapping
CLASS_NORM_MAP = {
    'apple': 'apple',
    'banana': 'banana',
    'avocado': 'avocado',
    'cherry': 'cherry',
    'kiwi': 'kiwi',
    'mango': 'mango',
    'orange': 'orange',
    'pinenapple': 'pineapple',
    'pineapple': 'pineapple',
    'stawberries': 'strawberry',
    'strawberries': 'strawberry',
    'strawberry': 'strawberry',
    'watermelon': 'watermelon'
}


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 byte hash for byte-level duplicate detection."""
    hasher = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_and_eda(zip_path: Path, extract_dir: Path) -> Dict:
    """Extracts raw dataset from zip and performs in-depth Exploratory Data Analysis (EDA).

    Normalizes directory names across case sensitivity and typos to strictly 10 fruit categories.
    """
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {zip_path.name} to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

    base_folder = extract_dir
    if (extract_dir / 'MY_data').exists():
        base_folder = extract_dir / 'MY_data'

    raw_train_dir = base_folder / 'train'
    raw_test_dir = base_folder / 'test'
    raw_predict_dir = base_folder / 'predict'

    eda_stats = {
        'raw_train_counts': {},
        'raw_test_counts': {},
        'raw_predict_count': 0,
        'flaw_identified': (
            "Raw Kaggle archive has casing mismatches ('Apple'/'Banana' in train vs 'apple'/'banana' in test) "
            "and a spelling typo ('stawberries' in test vs 'strawberries' in train). When unnormalized, naive scripts "
            "falsely create 11 or 12 classes or report 0 test samples for 3 categories. Normalizing casing and mapping "
            "'stawberries' -> 'strawberry' correctly restores the genuine 10 balanced fruit categories."
        ),
        'all_valid_images': [],
        'duplicates_detected': [],
        'corrupted_images': []
    }

    if raw_train_dir.exists():
        for cdir in sorted(raw_train_dir.iterdir()):
            if cdir.is_dir():
                raw_c = cdir.name.lower().strip()
                norm_c = CLASS_NORM_MAP.get(raw_c, raw_c)
                eda_stats['raw_train_counts'][norm_c] = len([p for p in cdir.glob('*.*') if p.is_file()])

    if raw_test_dir.exists():
        for cdir in sorted(raw_test_dir.iterdir()):
            if cdir.is_dir():
                raw_c = cdir.name.lower().strip()
                norm_c = CLASS_NORM_MAP.get(raw_c, raw_c)
                eda_stats['raw_test_counts'][norm_c] = len([p for p in cdir.glob('*.*') if p.is_file()])

    if raw_predict_dir.exists():
        eda_stats['raw_predict_count'] = len([p for p in raw_predict_dir.glob('*.*') if p.is_file()])

    seen_hashes = {}
    valid_records = []

    # Gather images directly from the actual folders to be robust across case-sensitive & insensitive filesystems
    for split_dir in [raw_train_dir, raw_test_dir]:
        if not split_dir.exists():
            continue
        for cdir in sorted(split_dir.iterdir()):
            if not cdir.is_dir():
                continue
            raw_c = cdir.name.lower().strip()
            norm_c = CLASS_NORM_MAP.get(raw_c, raw_c)
            for p in cdir.glob('*.*'):
                if not p.is_file() or p.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                    continue
                try:
                    with Image.open(p) as img:
                        img.verify()
                    with Image.open(p) as img:
                        w, h = img.size
                        if w < 32 or h < 32:
                            raise ValueError(f"Image too small: {w}x{h}")
                        ImageOps.exif_transpose(img).convert('RGB')

                    digest = compute_file_sha256(p)
                    if digest in seen_hashes:
                        prev_p, prev_cls = seen_hashes[digest]
                        eda_stats['duplicates_detected'].append({
                            'duplicate_file': str(p),
                            'original_file': str(prev_p),
                            'class': norm_c,
                            'sha256': digest
                        })
                        continue

                    seen_hashes[digest] = (p, norm_c)
                    valid_records.append({
                        'path': p,
                        'class': norm_c,
                        'sha256': digest,
                        'width': w,
                        'height': h
                    })
                except Exception as e:
                    eda_stats['corrupted_images'].append({'file': str(p), 'class': norm_c, 'error': str(e)})

    eda_stats['all_valid_images'] = valid_records
    eda_stats['total_valid_unique'] = len(valid_records)
    eda_stats['total_duplicates'] = len(eda_stats['duplicates_detected'])
    eda_stats['total_corrupted'] = len(eda_stats['corrupted_images'])

    clean_counts = Counter(r['class'] for r in valid_records)
    eda_stats['deduped_class_counts'] = dict(sorted(clean_counts.items()))

    return eda_stats


def create_stratified_split(
    valid_records: List[Dict],
    splits_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Dict[str, int], Path]:
    """Performs stratified split into Train, Val, and Test sets and saves images into standard folders."""
    if splits_dir.exists():
        shutil.rmtree(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    labels = [r['class'] for r in valid_records]
    indices = np.arange(len(valid_records))

    # Split Train vs (Val + Test)
    test_val_ratio = val_ratio + test_ratio
    train_idx, val_test_idx = train_test_split(
        indices,
        test_size=test_val_ratio,
        stratify=labels,
        random_state=seed
    )

    # Split Val vs Test
    val_test_labels = [labels[i] for i in val_test_idx]
    relative_test_ratio = test_ratio / test_val_ratio
    val_sub_idx, test_sub_idx = train_test_split(
        np.arange(len(val_test_idx)),
        test_size=relative_test_ratio,
        stratify=val_test_labels,
        random_state=seed
    )
    val_idx = val_test_idx[val_sub_idx]
    test_idx = val_test_idx[test_sub_idx]

    split_map = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }

    manifest = []
    split_counts = {'train': len(train_idx), 'val': len(val_idx), 'test': len(test_idx)}

    for split_name, idx_list in split_map.items():
        for i in idx_list:
            rec = valid_records[i]
            target_folder = splits_dir / split_name / rec['class']
            target_folder.mkdir(parents=True, exist_ok=True)
            target_filename = f"{rec['sha256'][:8]}_{rec['path'].name}"
            target_path = target_folder / target_filename

            # Save clean normalized RGB image
            with Image.open(rec['path']) as img:
                clean_rgb = ImageOps.exif_transpose(img).convert('RGB')
                clean_rgb.save(target_path, quality=95)

            manifest.append({
                'source': str(rec['path']),
                'split': split_name,
                'class': rec['class'],
                'sha256': rec['sha256'],
                'saved_path': str(target_path)
            })

    # Save manifest CSV
    manifest_path = splits_dir.parent / 'data_manifest.csv'
    with manifest_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'split', 'class', 'sha256', 'saved_path'])
        writer.writeheader()
        writer.writerows(manifest)

    return split_counts, manifest_path


def get_data_transforms(image_size: int = IMAGE_SIZE):
    """Returns leak-free transforms. Augmentations are STRICTLY restricted to training."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    return train_transform, eval_transform


def build_dataloaders(splits_dir: Path, batch_size: int = 32, num_workers: int = 0):
    """Creates PyTorch DataLoaders for train, val, and test splits."""
    train_tf, eval_tf = get_data_transforms()

    train_dataset = ImageFolder(splits_dir / 'train', transform=train_tf)
    val_dataset = ImageFolder(splits_dir / 'val', transform=eval_tf)
    test_dataset = ImageFolder(splits_dir / 'test', transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }
