"""Huấn luyện mô hình và lưu checkpoint tốt nhất theo validation loss."""
from pathlib import Path
import argparse
import csv
import json
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.model import FruitCNN
from src.data_utils import IMAGE_SIZE, MEAN, STD, SEED, make_transform


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total_loss, total_right, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()
            n = y.size(0)
            total_loss += loss.item() * n
            total_right += (logits.argmax(dim=1) == y).sum().item()
            total += n
    if total == 0:
        raise RuntimeError('DataLoader rỗng. Kiểm tra split và ảnh hợp lệ.')
    return total_loss / total, total_right / total


def main():
    p = argparse.ArgumentParser(description='FCVG101 Task 2: Train CNN')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()
    set_seed(SEED)
    root = Path('data/splits')
    tr = ImageFolder(root / 'train', transform=make_transform(train=True))
    va = ImageFolder(root / 'val', transform=make_transform(train=False))
    te = ImageFolder(root / 'test', transform=make_transform(train=False))
    if tr.class_to_idx != va.class_to_idx or tr.class_to_idx != te.class_to_idx:
        raise RuntimeError('Class mapping không nhất quán giữa train / val / test.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=0)
    net = FruitCNN(n_classes=len(tr.classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    out = Path('artifacts'); out.mkdir(parents=True, exist_ok=True)
    result = Path('results'); result.mkdir(parents=True, exist_ok=True)
    (out / 'classes.json').write_text(json.dumps(tr.class_to_idx, ensure_ascii=False, indent=2), encoding='utf-8')
    best_loss = float('inf')
    history = []
    print('device:', device, '| classes:', tr.classes, '| train/val/test:', len(tr), len(va), len(te))
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(net, loader, loss_fn, optim, device)
        va_loss, va_acc = run_epoch(net, val_loader, loss_fn, None, device)
        history.append({'epoch': ep, 'train_loss': tr_loss, 'train_accuracy': tr_acc,
                        'val_loss': va_loss, 'val_accuracy': va_acc})
        if va_loss < best_loss:
            best_loss = va_loss
            torch.save({
                'model_state': net.state_dict(), 'class_to_idx': tr.class_to_idx,
                'image_size': IMAGE_SIZE, 'mean': MEAN, 'std': STD,
                'architecture': 'FruitCNN_3Conv_GAP', 'best_epoch': ep,
                'best_val_loss': va_loss, 'best_val_accuracy': va_acc,
                'seed': SEED
            }, out / 'model.pth')
        print(f'epoch {ep:02d} train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} '
              f'val_loss={va_loss:.4f} val_acc={va_acc:.3f}')
    with (result / 'train_history.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for k, ax in enumerate(axes):
        suffix = 'loss' if k == 0 else 'accuracy'
        ax.plot([h['epoch'] for h in history], [h['train_' + suffix] for h in history], label='train')
        ax.plot([h['epoch'] for h in history], [h['val_' + suffix] for h in history], label='validation')
        ax.set_xlabel('epoch'); ax.set_ylabel(suffix); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(result / 'learning_curves.png', dpi=150); plt.close(fig)
    print('Lưu: artifacts/model.pth, artifacts/classes.json, results/train_history.csv/learning_curves.png')
    print('Chạy python evaluate.py để đánh giá trên test (chỉ sau khi chọn xong mô hình).')


if __name__ == '__main__':
    main()
