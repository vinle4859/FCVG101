"""FCVG101 Coursework - Task 2 Pipeline Execution Script.
Executes EDA, SHA-256 deduplication, stratified split, training of Custom CNN and MobileNetV2,
evaluates on locked test set, and generates Grad-CAM visualizations.
"""
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
import json
import torch
from src.data_pipeline import extract_and_eda, create_stratified_split, build_dataloaders
from src.models import (
    FruitSortCNN, FruitTransferNet, train_model,
    evaluate_model_on_test, plot_learning_curves
)
from src.gradcam import GradCAM, overlay_gradcam_on_image
import numpy as np
from PIL import Image

def main():
    base_dir = Path(__file__).resolve().parent
    zip_path = base_dir / 'Task2' / 'data' / 'fruit-10-class.zip'
    raw_dir = base_dir / 'data' / 'fruit10_raw'
    splits_dir = base_dir / 'data' / 'splits'
    artifacts_dir = base_dir / 'artifacts'
    results_dir = base_dir / 'results' / 'task2_deep_learning'

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Ingestion & Exploratory Data Analysis (EDA) ===")
    eda_stats = extract_and_eda(zip_path, raw_dir)
    print(f"Total valid unique images found: {eda_stats['total_valid_unique']}")
    print(f"Duplicates detected via SHA-256: {eda_stats['total_duplicates']}")
    print(f"Flaw identified in raw test set: {eda_stats['flaw_identified']}")

    with (results_dir / 'eda_summary.json').open('w', encoding='utf-8') as f:
        json.dump({
            'raw_train_counts': eda_stats['raw_train_counts'],
            'raw_test_counts': eda_stats['raw_test_counts'],
            'raw_predict_count': eda_stats['raw_predict_count'],
            'deduped_class_counts': eda_stats['deduped_class_counts'],
            'total_valid_unique': eda_stats['total_valid_unique'],
            'total_duplicates': eda_stats['total_duplicates'],
            'flaw_identified': eda_stats['flaw_identified']
        }, f, indent=2, ensure_ascii=False)

    print("\n=== Step 2: Stratified Splitting (70% Train / 15% Val / 15% Test) ===")
    split_counts, manifest_path = create_stratified_split(
        eda_stats['all_valid_images'], splits_dir,
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    print(f"Dataset split counts: {split_counts}")

    print("\n=== Step 3: PyTorch DataLoaders (Leak-free Transforms) ===")
    batch_size = 32
    loaders = build_dataloaders(splits_dir, batch_size=batch_size)
    class_names = loaders['classes']
    print(f"Classes ({len(class_names)}): {class_names}")

    with (artifacts_dir / 'classes.json').open('w', encoding='utf-8') as f:
        json.dump(loaders['class_to_idx'], f, indent=2, ensure_ascii=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training hardware device: {device}")

    print("\n=== Step 4: Model A - Custom CNN (FruitSortCNN) Training ===")
    cnn_model = FruitSortCNN(num_classes=len(class_names))
    cnn_save_path = artifacts_dir / 'custom_cnn_best.pth'
    cnn_model, cnn_history = train_model(
        cnn_model, loaders['train_loader'], loaders['val_loader'],
        num_epochs=8, learning_rate=0.001, device=device, save_path=cnn_save_path
    )
    plot_learning_curves(cnn_history, results_dir / 'custom_cnn_curves.png')

    print("\nEvaluating Custom CNN on locked Test set...")
    cnn_eval = evaluate_model_on_test(
        cnn_model, loaders['test_loader'], class_names, device=device,
        output_dir=results_dir / 'custom_cnn'
    )
    print(f"Custom CNN Test Accuracy: {cnn_eval['accuracy']*100:.2f}% | Macro F1: {cnn_eval['macro_f1']:.4f}")

    print("\n=== Step 5: Model B - Transfer Learning (MobileNetV2) Training ===")
    transfer_model = FruitTransferNet(num_classes=len(class_names), freeze_backbone=False)
    transfer_save_path = artifacts_dir / 'best_model.pth'  # Primary model for Task 3 app
    transfer_model, transfer_history = train_model(
        transfer_model, loaders['train_loader'], loaders['val_loader'],
        num_epochs=5, learning_rate=0.0003, device=device, save_path=transfer_save_path
    )
    plot_learning_curves(transfer_history, results_dir / 'mobilenet_curves.png')

    print("\nEvaluating Transfer Learning model on locked Test set...")
    transfer_eval = evaluate_model_on_test(
        transfer_model, loaders['test_loader'], class_names, device=device,
        output_dir=results_dir / 'mobilenet'
    )
    print(f"MobileNetV2 Test Accuracy: {transfer_eval['accuracy']*100:.2f}% | Macro F1: {transfer_eval['macro_f1']:.4f}")

    print("\n=== Step 6: Explainable AI - Grad-CAM Visualizations ===")
    # Target last conv layer in features
    target_layer = transfer_model.features[-1]
    
    # Pick 5 diverse test samples for Grad-CAM
    test_samples = list((splits_dir / 'test').rglob('*.jpg'))[:6]
    gradcam_dir = results_dir / 'gradcam_samples'
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    from src.data_pipeline import get_data_transforms
    _, eval_tf = get_data_transforms()

    with GradCAM(transfer_model, target_layer) as gradcam:
        for idx, sample_path in enumerate(test_samples):
            pil_img = Image.open(sample_path).convert('RGB')
            tensor_img = eval_tf(pil_img).unsqueeze(0).to(device)
            
            logits = transfer_model(tensor_img)
            pred_idx = int(logits.argmax(dim=1).item())
            pred_label = class_names[pred_idx]
            actual_label = sample_path.parent.name

            cam = gradcam.generate_cam(tensor_img, target_class=pred_idx)
            orig_np = np.array(pil_img)
            blended, colored_cam = overlay_gradcam_on_image(orig_np, cam, alpha=0.5)

            save_img = Image.fromarray(blended)
            save_img.save(gradcam_dir / f"gradcam_sample_{idx+1}_{actual_label}_pred_{pred_label}.jpg")
            print(f"Grad-CAM sample {idx+1}: Actual={actual_label}, Pred={pred_label} saved.")

    print("\nAll Task 2 pipeline operations completed successfully!")

if __name__ == '__main__':
    main()
