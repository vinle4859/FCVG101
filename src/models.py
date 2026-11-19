"""FCVG101 Coursework - Task 2.2: Deep Learning Architectures, Training Loop & Evaluation.
Implements Custom CNN (FruitSortCNN) and Transfer Learning (MobileNetV2),
along with complete training, metric evaluation, and visualization utilities.
"""
from pathlib import Path
import csv
import json
import time
from typing import Dict, Tuple, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class FruitSortCNN(nn.Module):
    """Custom Convolutional Neural Network with BatchNorm, MaxPool, AdaptiveAvgPool, and Dropout."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 128 -> 64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 16 -> 8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


class FruitTransferNet(nn.Module):
    """Transfer Learning model using MobileNetV2 pretrained on ImageNet."""
    def __init__(self, num_classes: int = 10, freeze_backbone: bool = False):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT
        base = mobilenet_v2(weights=weights)
        self.features = base.features
        
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
                
        in_features = base.classifier[1].in_features  # 1280
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = nn.functional.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        logits = self.classifier(feat)
        return logits


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    save_path: Path = None
) -> Tuple[nn.Module, List[Dict]]:
    """Trains model with validation loss monitoring, checkpointing, and history tracking."""
    dev = torch.device(device)
    model.to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    best_val_loss = float('inf')
    best_state = None

    print(f"Starting training on {device} for {num_epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += x.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total += x.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        history.append({
            'epoch': epoch,
            'train_loss': epoch_train_loss,
            'train_acc': epoch_train_acc,
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc
        })

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")

    # Load best state dict
    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model checkpoint to {save_path}")

    return model, history


def evaluate_model_on_test(
    model: nn.Module,
    test_loader,
    class_names: List[str],
    device: str = 'cpu',
    output_dir: Path = None
) -> Dict:
    """Evaluates locked model on test dataset. Computes Accuracy, F1, Per-class stats and Confusion Matrix."""
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(dev)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(y.tolist())

    labels_idx = list(range(len(class_names)))
    overall_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels_idx, target_names=class_names, output_dict=True, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    results = {
        'accuracy': float(overall_acc),
        'macro_f1': float(macro_f1),
        'per_class_report': report,
        'confusion_matrix': cm.tolist(),
        'y_true': y_true,
        'y_pred': y_pred
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save metrics json
        with (output_dir / 'test_metrics.json').open('w', encoding='utf-8') as f:
            json.dump({
                'accuracy': overall_acc,
                'macro_f1': macro_f1,
                'classification_report': report
            }, f, indent=2, ensure_ascii=False)

        # Plot Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 7))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap='Blues', colorbar=False, xticks_rotation=45)
        ax.set_title(f'Test Confusion Matrix (Accuracy: {overall_acc*100:.2f}%)', fontsize=12, fontweight='bold')
        fig.tight_layout()
        fig.savefig(output_dir / 'confusion_matrix.png', dpi=150)
        plt.close(fig)

    return results


def plot_learning_curves(history: List[Dict], save_path: Path):
    """Plots Train vs Val Loss and Accuracy curves."""
    epochs = [h['epoch'] for h in history]
    tr_loss = [h['train_loss'] for h in history]
    va_loss = [h['val_loss'] for h in history]
    tr_acc = [h['train_acc'] * 100 for h in history]
    va_acc = [h['val_acc'] * 100 for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Loss curve
    ax1.plot(epochs, tr_loss, 'o-', color='#3b82f6', label='Train Loss')
    ax1.plot(epochs, va_loss, 's--', color='#ef4444', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Cross-Entropy Loss Curve', fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend()

    # Accuracy curve
    ax2.plot(epochs, tr_acc, 'o-', color='#10b981', label='Train Acc')
    ax2.plot(epochs, va_acc, 's--', color='#f59e0b', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Classification Accuracy Curve', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
