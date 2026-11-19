# SpectraBio Vision Assistant: Precision Agronomy & Optical Inspection

[![Course](https://img.shields.io/badge/Course-FCVG101_Intro_to_Computer_Vision-059669.svg)](https://www.gre.ac.uk/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch_2.x_%7C_OpenCV-3b82f6.svg)](https://pytorch.org/)
[![UI](https://img.shields.io/badge/Dashboard-Streamlit_Native-10b981.svg)](https://streamlit.io/)
[![Colab](https://img.shields.io/badge/Colab-Master_Notebook_Ready-f59e0b.svg)](https://colab.research.google.com/)

> **Coursework Project for FCVG101 — Introduction to Computer Vision (Weight: 60%)**  
> University of Greenwich  
> Deliverable: 01 Single PDF Report ($\le$ 5 MB) exported from `FCVG101_Coursework_Master.ipynb`

---

## 🌟 Executive Overview & Scenario

The **SpectraBio Vision Assistant** is an MVP Computer Vision system designed as an intelligent visual inspection assistant for precision agronomy, automated warehouse management, and high-throughput fruit quality sorting. The project satisfies all core Learning Outcomes:

* **LO1 (Image Formation & Camera Calibration):** Pinhole camera model calibration using real captured checkerboard images (9x6 inner corners, 18 mm square size), intrinsic matrix $K$, lens distortion coefficients ($k_1, k_2, p_1, p_2, k_3$), per-view Euclidean reprojection error analysis, before/after optical undistortion, and 3D camera extrinsic poses visualization.
* **LO2 (Filtering, Gradients & Interest Points):** Linear spatial filtering (Box, Gaussian, Median) evaluated quantitatively (MSE & PSNR) under Gaussian and impulse Salt-and-Pepper noise; Sobel & Canny edge detection; 2D FFT frequency analysis with Ideal vs Gaussian Low-Pass Filtering and rigorous analysis of the **Gibbs Ringing Artifact**; Harris Corner Detector implemented **from scratch** vs OpenCV; Multi-view SIFT/ORB feature matching with Lowe's ratio test and **RANSAC Homography** geometric verification.
* **LO3 (Deep Learning & Data Pipeline):** In-depth Exploratory Data Analysis (EDA) on the Kaggle 10-Class Fruit dataset, identifying and fixing critical dataset defects (casing mismatches, typo in "stawberries", and zero test samples for Apple/Banana/Strawberries in raw split); SHA-256 byte deduplication (781 duplicates removed); Leak-free stratified splitting (70% Train, 15% Val, 15% Test); Custom CNN architecture (`FruitSortCNN`) vs Transfer Learning (`MobileNetV2`); Evaluation on locked test set (Accuracy, Macro F1, Confusion Matrix, Classification Report); Explainable AI (XAI) via **Grad-CAM** visual attention heatmaps.
* **LO4 (Application Integration & Assistant Feature):** Interactive Streamlit web application (`app.py`) featuring a distinctive **"SpectraBio Agronomy Lab"** industrial bio-inspection aesthetic (dark slate palette, glassmorphic telemetry cards, real-time sliders, zero cookie-cutter styling).

---

## 📁 Repository Structure

```text
g:/My Drive/UOG/Intro to CV/Coursework/
├── FCVG101_Coursework_Master.ipynb  # Master Notebook runnable on Colab / Jupyter for PDF Export
├── app.py                           # Task 3 Streamlit Visual Inspection Assistant Dashboard
├── run_task2_pipeline.py            # Automated Task 2 EDA, deduplication, training & Grad-CAM script
├── README.md                        # Documentation & Execution Guide
├── requirements.txt                 # Project dependencies
│
├── src/                             # Modular Python Core Library
│   ├── __init__.py
│   ├── calibration.py               # Task 1.1: Camera calibration, K matrix, undistortion, 3D poses
│   ├── filtering.py                 # Task 1.2: Box/Gauss/Median, MSE/PSNR, Sobel/Canny, 2D FFT & Gibbs
│   ├── features.py                  # Task 1.3: Harris from scratch vs OpenCV, SIFT/ORB + RANSAC
│   ├── data_pipeline.py             # Task 2.1: Ingestion, SHA-256 deduplication, stratified split
│   ├── models.py                    # Task 2.2: FruitSortCNN, FruitTransferNet, training loop & metrics
│   └── gradcam.py                   # Task 2.3: Explainable AI Grad-CAM implementation
│
├── artifacts/                       # Trained Models & Metadata Checkpoints
│   ├── best_model.pth               # Best locked model checkpoint (MobileNetV2 Transfer Learning)
│   ├── custom_cnn_best.pth          # Best locked model checkpoint (Custom Light-CNN)
│   ├── classes.json                 # Label to index mapping (10 fruit categories)
│   └── calibration.npz              # Camera intrinsics K and distortion parameters
│
├── data/                            # Processed & Partitioned Data
│   ├── captured_images/             # Real user photos of 9x6 checkerboard pattern
│   └── splits/                      # Leak-free partitioned datasets
│       ├── train/                   # 70% stratified training split (augmented dynamically)
│       ├── val/                     # 15% stratified validation split (deterministic)
│       └── test/                    # 15% strictly locked test split (deterministic)
│
└── results/                         # Evaluation Figures, Metrics & Heatmaps
    ├── task1_calibration/           # Reprojection error plots & 3D extrinsics
    └── task2_deep_learning/         # Learning curves, confusion matrices, Grad-CAM samples
```

---

## ⚡ Quick Start Guide (1-Click Run)

### 1. Environment Installation
Ensure Python 3.10+ is installed. Open PowerShell / Terminal in this folder:

```bash
# Optional: create and activate virtual environment
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Camera Calibration (Task 1.1)
```bash
python -c "from src.calibration import calibrate_camera_pipeline; from pathlib import Path; calibrate_camera_pipeline(Path('Task1A/FCVG101_Task1A_Workshop/data/captured_images'), cols=9, rows=6, square_mm=18.0, output_dir=Path('results/task1_calibration'))"
```

### 3. Run Data Pipeline & Train Models (Task 2)
```bash
python run_task2_pipeline.py
```
*Automatically unzips the dataset, runs EDA and SHA-256 deduplication, generates leak-free stratified splits, trains both Custom CNN and MobileNetV2, computes locked test metrics, and exports Grad-CAM heatmaps.*

### 4. Launch the Visual Inspection Assistant Dashboard (Task 3)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to access the interactive dashboard.

---

## 🔬 Interactive Dashboard Operational Modes

1. **Mode 1: Camera Calibration & Optical Undistortion Workbench**
   * Real-time before/after optical correction slider.
   * Full $K$ matrix readout with focal lengths ($f_x, f_y$) and principal point ($c_x, c_y$).
   * Per-view reprojection error bar chart and 3D camera extrinsics poses.
2. **Mode 2: Spatial & Frequency Domain Inspection (FFT & Gibbs Ringing)**
   * Live noise injection (Gaussian vs Salt & Pepper) with real-time MSE and PSNR metrics.
   * Box, Gaussian, and Median filter comparison proving Median filter superiority on impulse noise.
   * Sobel gradients and Canny edge detection with hysteresis threshold controls.
   * 2D FFT magnitude spectrum and direct empirical proof of the **Gibbs Ringing Artifact** under Ideal LPF vs Gaussian LPF.
3. **Mode 3: Interest-Point Detection & Geometric Feature Matching**
   * Harris Corner Detector implemented **from scratch** vs OpenCV `cv2.cornerHarris` (demonstrating ~100% keypoint agreement).
   * SIFT and ORB multi-view matching with Lowe's ratio test and **RANSAC Homography** geometric verification.
4. **Mode 4: AI Fruit Quality Inspection & Explainable AI (Grad-CAM)**
   * Upload any fruit image or select a sample from the test split.
   * Real-time CNN inference (<20 ms latency) with top-3 category confidence meters.
   * Grad-CAM attention heatmap overlay with adjustable alpha blending slider to visualize where the neural network is looking.

---

## 📑 Generating the CMS Submission PDF ($\le$ 5 MB)

1. Open `FCVG101_Coursework_Master.ipynb` in **Google Colab** or **Jupyter Lab**.
2. Run all cells sequentially (`Runtime` $\rightarrow$ `Run all`).
3. Export to PDF:
   * **In Google Colab:** Select `File` $\rightarrow$ `Print` $\rightarrow$ `Save as PDF` (Ensure `Background graphics` is checked).
   * **In Jupyter:** Select `File` $\rightarrow$ `Save and Export Notebook As...` $\rightarrow$ `PDF / HTML`.
4. All images and plots are generated at 150 DPI, guaranteeing the output PDF is well below the **5 MB** CMS limit while maintaining crisp, text-selectable code.

---
*Developed for FCVG101 Introduction to Computer Vision — University of Greenwich.*
