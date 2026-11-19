"""Đánh giá checkpoint đã khóa trên tập test (không chọn lại model bằng test)."""
from pathlib import Path
import csv
import json
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from src.inference import load_model


def main():
    model, idx_to_class, transform = load_model()
    test = ImageFolder('data/splits/test', transform=transform)
    if test.class_to_idx != {name: i for i, name in idx_to_class.items()}:
        raise RuntimeError('Class-to-index test khác checkpoint!')
    loader = DataLoader(test, batch_size=32, shuffle=False, num_workers=0)
    targets, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x).argmax(1).cpu().tolist())
            targets.extend(y.tolist())
    labels = list(range(len(idx_to_class)))
    names = [idx_to_class[i] for i in labels]
    accuracy = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, labels=labels, average='macro', zero_division=0)
    report = classification_report(targets, preds, labels=labels, target_names=names,
                                   output_dict=True, zero_division=0)
    print(f'TEST n={len(test)} | accuracy={accuracy:.4f} | macro_F1={macro_f1:.4f}')
    from sklearn.metrics import classification_report as cr
    print(cr(targets, preds, labels=labels, target_names=names, zero_division=0))
    out = Path('results'); out.mkdir(exist_ok=True)
    (out / 'test_metrics.json').write_text(json.dumps({
        'test_samples': len(test), 'accuracy': accuracy, 'macro_f1': macro_f1,
        'per_class': report
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out / 'test_predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'actual', 'predicted', 'correct'])
        for (path, _), real, predicted in zip(test.samples, targets, preds):
            writer.writerow([path, names[real], names[predicted], int(real == predicted)])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cm = confusion_matrix(targets, preds, labels=labels)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, colorbar=False, cmap='Blues')
    fig.tight_layout(); fig.savefig(out / 'confusion_matrix.png', dpi=150); plt.close(fig)
    print('Đã lưu: results/test_metrics.json, results/test_predictions.csv, results/confusion_matrix.png')


if __name__ == '__main__':
    main()
