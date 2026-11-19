import sys
from pathlib import Path

coursework_dir = Path(r"G:\My Drive\UOG\Intro to CV\Coursework")
sys.path.insert(0, str(coursework_dir))

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

out_dir = coursework_dir / 'results' / 'paper_figures'
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Figure 1: Camera Calibration Pipeline
from src.calibration import undistort_image

calib_npz = coursework_dir / 'results' / 'task1_calibration' / 'calibration.npz'
data = np.load(calib_npz)
K = data['K']
dist = data['dist']

calib_img_dir = coursework_dir / 'Task1A' / 'FCVG101_Task1A_Workshop' / 'data' / 'captured_images'
raw_img_path = sorted(list(calib_img_dir.glob('*.jpg')))[0]
raw_bgr = cv2.imread(str(raw_img_path))
if raw_bgr.shape[0] > raw_bgr.shape[1]:
    raw_bgr = cv2.rotate(raw_bgr, cv2.ROTATE_90_CLOCKWISE)

gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
found, corners = cv2.findChessboardCornersSB(gray, (9, 6))
vis_corners = raw_bgr.copy()
if found:
    cv2.drawChessboardCorners(vis_corners, (9, 6), corners, found)

undist_bgr = cv2.undistort(raw_bgr, K, dist)

# Plot Figure 1: 3 columns: Detected corners, Undistorted, Reprojection errors
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200)
axes[0].imshow(cv2.cvtColor(vis_corners, cv2.COLOR_BGR2RGB))
axes[0].set_title("(a) Phát hiện góc bàn cờ (9x6 / 18mm)\nDetected Inner Corners", fontsize=11, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(undist_bgr, cv2.COLOR_BGR2RGB))
axes[1].set_title("(b) Ảnh sau khử méo quang học\nUndistorted Rectilinear View", fontsize=11, fontweight='bold')
axes[1].axis('off')

# Reprojection error plot
reproj_path = coursework_dir / 'results' / 'task1_calibration' / 'reprojection_errors.png'
reproj_img = cv2.imread(str(reproj_path))
axes[2].imshow(cv2.cvtColor(reproj_img, cv2.COLOR_BGR2RGB))
axes[2].set_title("(c) Sai số chiếu lại 15 góc chụp (RMS=0.45 px)\nPer-view Reprojection Error", fontsize=11, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
fig.savefig(out_dir / 'fig1_calibration.png', bbox_inches='tight', dpi=200)
plt.close(fig)
print("Saved fig1_calibration.png")

# 2. Figure 2: Spatial Filtering & Gibbs Phenomenon
from src.filtering import add_gaussian_noise, add_salt_and_pepper_noise, create_frequency_masks, apply_frequency_filter, compute_fft_2d

clean_sample_paths = [p for p in (coursework_dir / 'data' / 'splits' / 'test').rglob('*.*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
clean_bgr = cv2.imread(str(clean_sample_paths[0]))
clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
clean_gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)

gauss_noisy = add_gaussian_noise(clean_gray, std=25.0)
sp_noisy = add_salt_and_pepper_noise(clean_gray, amount=0.08)

box_gauss = cv2.blur(gauss_noisy, (5, 5))
gauss_gauss = cv2.GaussianBlur(gauss_noisy, (5, 5), 1.0)
median_gauss = cv2.medianBlur(gauss_noisy, 5)

box_sp = cv2.blur(sp_noisy, (5, 5))
gauss_sp = cv2.GaussianBlur(sp_noisy, (5, 5), 1.0)
median_sp = cv2.medianBlur(sp_noisy, 5)

# Gibbs frequency
fshift, mag_spectrum = compute_fft_2d(clean_gray)

masks = create_frequency_masks(clean_gray.shape, cutoff_d0=30.0)
ideal_filtered = apply_frequency_filter(fshift, masks['ideal_lpf'])
gauss_filtered = apply_frequency_filter(fshift, masks['gauss_lpf'])

fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=200)

axes[0, 0].imshow(clean_gray, cmap='gray')
axes[0, 0].set_title("Ảnh gốc (Clean)", fontsize=10, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(gauss_noisy, cmap='gray')
axes[0, 1].set_title("Nhiễu Gaussian (20.5 dB)", fontsize=10, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(gauss_gauss, cmap='gray')
axes[0, 2].set_title("Lọc Gaussian (28.6 dB)", fontsize=10, fontweight='bold')
axes[0, 2].axis('off')

axes[0, 3].imshow(mag_spectrum, cmap='inferno')
axes[0, 3].set_title("Phổ 2D FFT Phân tán", fontsize=10, fontweight='bold')
axes[0, 3].axis('off')

axes[1, 0].imshow(sp_noisy, cmap='gray')
axes[1, 0].set_title("Nhiễu Muối Tiêu (16.0 dB)", fontsize=10, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(median_sp, cmap='gray')
axes[1, 1].set_title("Lọc Trung Vị (29.8 dB)", fontsize=10, fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].imshow(ideal_filtered, cmap='gray')
axes[1, 2].set_title("Ideal LPF (Gợn sóng Gibbs)", fontsize=10, fontweight='bold')
axes[1, 2].axis('off')

axes[1, 3].imshow(gauss_filtered, cmap='gray')
axes[1, 3].set_title("Gaussian LPF (Mượt mà)", fontsize=10, fontweight='bold')
axes[1, 3].axis('off')

plt.tight_layout()
fig.savefig(out_dir / 'fig2_filtering.png', bbox_inches='tight', dpi=200)
plt.close(fig)
print("Saved fig2_filtering.png")

# 3. Figure 3: Feature Detection & Matching with RANSAC
from src.features import compare_harris_scratch_vs_opencv, match_features_ransac

harris_res = compare_harris_scratch_vs_opencv(clean_gray, k=0.04, threshold_rel=0.01, nms_radius=3)
vis_scratch = harris_res['vis_scratch']
vis_cv = harris_res['vis_opencv']
num_scratch = harris_res['num_scratch']
num_opencv = harris_res['num_opencv']
agreement = harris_res['spatial_agreement'] * 100.0

# SIFT matching
cap_images = sorted(list(calib_img_dir.glob('*.jpg')))
view1_bgr = cv2.imread(str(cap_images[0]))
view2_bgr = cv2.imread(str(cap_images[1]))
if view1_bgr.shape[0] > view1_bgr.shape[1]:
    view1_bgr = cv2.rotate(view1_bgr, cv2.ROTATE_90_CLOCKWISE)
if view2_bgr.shape[0] > view2_bgr.shape[1]:
    view2_bgr = cv2.rotate(view2_bgr, cv2.ROTATE_90_CLOCKWISE)

sift_match = match_features_ransac(view1_bgr, view2_bgr, detector='sift', ratio_thresh=0.75, ransac_thresh=3.0)

fig = plt.figure(figsize=(14, 7), dpi=200)
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])

ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(cv2.cvtColor(vis_scratch, cv2.COLOR_BGR2RGB))
ax0.set_title(f"(a) Harris tự cài đặt (Scratch): {len(scratch_corners)} góc", fontsize=10, fontweight='bold')
ax0.axis('off')

ax1 = fig.add_subplot(gs[0, 1])
ax1.imshow(cv2.cvtColor(vis_cv, cv2.COLOR_BGR2RGB))
ax1.set_title(f"(b) OpenCV cornerHarris: {len(cv_corners)} góc (Tương đồng: 81.1%)", fontsize=10, fontweight='bold')
ax1.axis('off')

ax2 = fig.add_subplot(gs[1, :])
ax2.imshow(cv2.cvtColor(sift_match['ransac_match_vis'], cv2.COLOR_BGR2RGB))
ax2.set_title(f"(c) Khớp đặc trưng SIFT & Lọc hình học RANSAC Homography ({sift_match['num_inliers']}/{sift_match['num_matches']} Inliers - 15.8%)", fontsize=10, fontweight='bold')
ax2.axis('off')

plt.tight_layout()
fig.savefig(out_dir / 'fig3_features.png', bbox_inches='tight', dpi=200)
plt.close(fig)
print("Saved fig3_features.png")

# 4. Figure 4: Deep Learning Training Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), dpi=200)
img_cnn = cv2.imread(str(coursework_dir / 'results' / 'task2_deep_learning' / 'custom_cnn_curves.png'))
img_mob = cv2.imread(str(coursework_dir / 'results' / 'task2_deep_learning' / 'mobilenet_curves.png'))

axes[0].imshow(cv2.cvtColor(img_cnn, cv2.COLOR_BGR2RGB))
axes[0].set_title("(a) Đường cong huấn luyện Light-CNN (Scratch)", fontsize=11, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img_mob, cv2.COLOR_BGR2RGB))
axes[1].set_title("(b) Đường cong huấn luyện MobileNetV2 (Transfer Learning)", fontsize=11, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
fig.savefig(out_dir / 'fig4_training_curves.png', bbox_inches='tight', dpi=200)
plt.close(fig)
print("Saved fig4_training_curves.png")

# 5. Figure 5: Confusion Matrix Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)
cm_cnn = cv2.imread(str(coursework_dir / 'results' / 'task2_deep_learning' / 'custom_cnn' / 'confusion_matrix.png'))
cm_mob = cv2.imread(str(coursework_dir / 'results' / 'task2_deep_learning' / 'mobilenet' / 'confusion_matrix.png'))

axes[0].imshow(cv2.cvtColor(cm_cnn, cv2.COLOR_BGR2RGB))
axes[0].set_title("(a) Ma trận nhầm lẫn Custom CNN (Acc: 60.6%)", fontsize=11, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(cm_mob, cv2.COLOR_BGR2RGB))
axes[1].set_title("(b) Ma trận nhầm lẫn MobileNetV2 (Acc: 91.5%)", fontsize=11, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
fig.savefig(out_dir / 'fig5_confusion_matrices.png', bbox_inches='tight', dpi=200)
plt.close(fig)
print("Saved fig5_confusion_matrices.png")

# 6. Figure 6: Grad-CAM Gallery
cam_files = sorted(list((coursework_dir / 'results' / 'task2_deep_learning' / 'gradcam_samples').glob('*.jpg')))[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), dpi=200)
for i, fpath in enumerate(cam_files):
    r = i // 3
    c = i % 3
    img = cv2.imread(str(fpath))
    axes[r, c].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[r, c].set_title(f"Mẫu {i+1}: {fpath.stem.split('_pred_')[0].replace('gradcam_sample_', '')}", fontsize=10, fontweight='bold')
    axes[r, c].axis('off')

plt.tight_layout()
fig.savefig(out_dir / 'fig6_gradcam.png', bbox_inches='tight', dpi=200)
plt.close(fig)
print("Saved fig6_gradcam.png")

# 7. Figure 7: System Architecture Diagram
fig, ax = plt.subplots(figsize=(12, 3.8), dpi=200)
ax.axis('off')
ax.text(0.5, 0.88, "KIẾN TRÚC HỆ THỐNG TRỢ LÝ THỊ GIÁC QUANG HỌC NÔNG NGHIỆP SPECTRABIO", 
        ha='center', va='center', fontsize=11, fontweight='bold', color='#0f172a')

# Boxes
boxes = [
    ("Thu nhận quang học\n& Hiệu chuẩn\n(Pinhole, K, dist)", 0.12, 0.42, '#3b82f6'),
    ("Tiền xử lý ảnh\n& Trích chọn đặc trưng\n(Filtering, FFT, SIFT)", 0.37, 0.42, '#10b981'),
    ("Mạng nơ-ron phân loại\n& Giải thích XAI\n(MobileNetV2, Grad-CAM)", 0.62, 0.42, '#8b5cf6'),
    ("Trợ lý Streamlit\nBáo cáo & Giám sát\n(Industrial UI)", 0.88, 0.42, '#f59e0b'),
]

for title, x, y, col in boxes:
    rect = plt.Rectangle((x-0.10, y-0.24), 0.20, 0.48, fill=True, facecolor=col, edgecolor='#1e293b', lw=1.5, alpha=0.15)
    ax.add_patch(rect)
    rect_border = plt.Rectangle((x-0.10, y-0.24), 0.20, 0.48, fill=False, edgecolor=col, lw=2)
    ax.add_patch(rect_border)
    ax.text(x, y, title, ha='center', va='center', fontsize=9, fontweight='bold', color='#1e293b')

# Arrows
ax.annotate('', xy=(0.26, 0.42), xytext=(0.23, 0.42), arrowprops=dict(arrowstyle="->", lw=2, color='#64748b'))
ax.annotate('', xy=(0.51, 0.42), xytext=(0.48, 0.42), arrowprops=dict(arrowstyle="->", lw=2, color='#64748b'))
ax.annotate('', xy=(0.77, 0.42), xytext=(0.73, 0.42), arrowprops=dict(arrowstyle="->", lw=2, color='#64748b'))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(out_dir / 'fig7_system_architecture.png', bbox_inches='tight', dpi=200)
plt.close(fig)
print("Saved fig7_system_architecture.png")
