"""Kiểm tra dữ liệu và tạo train/val/test theo group, KHÔNG augmentation offline.

Quy ước đặt tên: <group_id>__<view_id>.jpg nếu có nhiều ảnh cùng vật/phiên chụp.
Các ảnh cùng group_id trong cùng class luôn thuộc cùng một split.
Chỉ dùng với nhãn ở thư mục con data/raw/<class>/.
"""
from pathlib import Path
from collections import defaultdict, Counter
import csv
import hashlib
import random
import shutil
from PIL import Image, ImageOps

RAW = Path('data/raw')
OUT = Path('data/splits')
ART = Path('artifacts')
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SEED = 42


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def group_name(path: Path) -> str:
    # Với file item001__cam2.jpg, cả item001__cam1 và item001__cam2 cùng group.
    return path.stem.split('__')[0]


def main():
    if not RAW.exists() or not any(p.is_dir() for p in RAW.iterdir()):
        raise SystemExit('Cần tạo data/raw/<class>/*.jpg trước (xem README).')
    ART.mkdir(exist_ok=True, parents=True)
    image_groups = defaultdict(lambda: defaultdict(list))
    skipped = []
    seen_hash = {}
    classes = sorted(p.name for p in RAW.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise SystemExit('Cần ít nhất 2 thư mục lớp trong data/raw.')
    for label in classes:
        for p in sorted((RAW / label).rglob('*')):
            if not p.is_file() or p.suffix.lower() not in EXT:
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    if im.width < 32 or im.height < 32:
                        raise ValueError('Ảnh nhỏ hơn 32 px')
                    ImageOps.exif_transpose(im).convert('RGB').load()
                digest = file_sha256(p)
                if digest in seen_hash:
                    prev_path, prev_label = seen_hash[digest]
                    if prev_label != label:
                        raise ValueError(f'TRÙNG ảnh giữa 2 lớp: {prev_path}')
                    skipped.append((str(p), label, 'exact duplicate'))
                    continue
                seen_hash[digest] = (str(p), label)
                image_groups[label][group_name(p)].append((p, digest))
            except Exception as exc:
                skipped.append((str(p), label, str(exc)))
    # Bản split được làm lại từ đầu nhằm tránh sót ảnh từ lần chạy trước.
    if OUT.exists():
        shutil.rmtree(OUT)
    rng = random.Random(SEED)
    manifest = []
    for label in classes:
        groups = sorted(image_groups[label])
        if len(groups) < 3:
            raise SystemExit(f'Lớp {label}: cần ít nhất 3 group độc lập; hiện có {len(groups)}.')
        rng.shuffle(groups)
        count = len(groups)
        n_test = max(1, round(count * 0.15))
        n_val = max(1, round(count * 0.15))
        if n_test + n_val >= count:
            n_test, n_val = 1, 1
        partitions = {
            'test': groups[:n_test],
            'val': groups[n_test:n_test + n_val],
            'train': groups[n_test + n_val:]
        }
        for split, group_ids in partitions.items():
            folder = OUT / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for group_id in group_ids:
                for p, digest in image_groups[label][group_id]:
                    target = folder / f'{digest[:10]}_{p.stem}.jpg'
                    with Image.open(p) as im:
                        ImageOps.exif_transpose(im).convert('RGB').save(
                            target, quality=95, optimize=True)
                    manifest.append({
                        'source': str(p), 'split': split, 'label': label,
                        'group_id': group_id, 'sha256': digest,
                        'saved_path': str(target)
                    })
    with (ART / 'data_manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['source', 'split', 'label', 'group_id', 'sha256', 'saved_path'])
        w.writeheader()
        w.writerows(manifest)
    with (ART / 'rejected_images.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['source', 'label', 'reason'])
        w.writerows(skipped)
    print('Tổng số ảnh hợp lệ:', len(manifest))
    counts = Counter((r['split'], r['label']) for r in manifest)
    for split in ['train', 'val', 'test']:
        print(split, {c: counts[(split, c)] for c in classes})
    print('Đã tạo data/splits và artifacts/data_manifest.csv')
    print('LƯU Ý: near-duplicate cần kiểm tra bằng mắt/perceptual hash; group_id chỉ hữu ích khi đặt tên đúng.')


if __name__ == '__main__':
    main()
