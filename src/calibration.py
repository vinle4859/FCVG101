"""FCVG101 Coursework - Task 1.1: Camera Calibration & Image Formation Engine.
Computes intrinsic camera matrix K, lens distortion coefficients, per-view reprojection errors,
undistorts real captured images, and visualizes 3D camera extrinsic poses.
"""
from pathlib import Path
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def load_image_paths(folder: Path):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {folder.resolve()}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)


def detect_checkerboard_corners(gray: np.ndarray, pattern_size=(9, 6)):
    """Detects inner corners using robust SB detector with fallback to classic subpixel."""
    if hasattr(cv2, 'findChessboardCornersSB'):
        flags_sb = cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_EXHAUSTIVE
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags_sb)
        if ok:
            return True, corners.astype(np.float32), 'ChessboardCornersSB'
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size)
        if ok:
            return True, corners.astype(np.float32), 'ChessboardCornersSB_Default'

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if ok:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners, 'Classic_SubPix'

    return False, None, 'None'


def audit_captured_images(image_folder: Path, cols: int = 9, rows: int = 6):
    """Audits captured checkerboard images for dimension, sharpness, lighting, and corner detectability."""
    image_folder = Path(image_folder)
    image_paths = load_image_paths(image_folder)
    pattern = (cols, rows)
    audit_records = []

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            audit_records.append({
                'filename': p.name, 'width': 0, 'height': 0, 'aspect': 'unknown',
                'sharpness': 0.0, 'mean_brightness': 0.0, 'corners_found': False, 'detector': 'N/A'
            })
            continue

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_bright = float(np.mean(gray))

        ok, corners, detector_name = detect_checkerboard_corners(gray, pattern)
        if not ok and h > w:
            gray_rot = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ok, corners, detector_name = detect_checkerboard_corners(gray_rot, pattern)
            if ok:
                detector_name += "_Rotated90"

        audit_records.append({
            'filename': p.name,
            'width': w,
            'height': h,
            'orientation': 'landscape' if w >= h else 'portrait',
            'sharpness': round(sharpness, 1),
            'mean_brightness': round(mean_bright, 1),
            'corners_found': ok,
            'detector': detector_name
        })

    return audit_records


def calibrate_camera_pipeline(
    image_folder: Path,
    cols: int = 9,
    rows: int = 6,
    square_mm: float = 18.0,
    output_dir: Path = None,
    normalize_orientation: bool = True,
    filter_orientation: str = 'none'
):
    """Executes full camera calibration pipeline on folder of images.
    
    Args:
        filter_orientation: 'none' (use all views), 'landscape_only' (only native w > h), 'portrait_only'.
    """
    image_folder = Path(image_folder)
    output_dir = Path(output_dir) if output_dir else None
    image_paths = load_image_paths(image_folder)
    if not image_paths:
        raise ValueError(f"No valid images found in {image_folder}")

    pattern = (cols, rows)
    square_m = square_mm / 1000.0

    # 3D world points for planar checkerboard (Z = 0)
    world_pts = np.zeros((cols * rows, 3), np.float32)
    world_pts[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * square_m

    object_points = []
    image_points = []
    accepted_paths = []
    records = []
    reference_size = None  # (width, height)

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            records.append({'filename': p.name, 'status': 'UNREADABLE', 'shape': 'N/A', 'detector': 'N/A'})
            continue

        h, w = img.shape[:2]
        is_portrait = (h > w)

        if filter_orientation == 'landscape_only' and is_portrait:
            records.append({'filename': p.name, 'status': 'FILTERED_PORTRAIT', 'shape': f"{w}x{h}", 'detector': 'N/A'})
            continue
        elif filter_orientation == 'portrait_only' and not is_portrait:
            records.append({'filename': p.name, 'status': 'FILTERED_LANDSCAPE', 'shape': f"{w}x{h}", 'detector': 'N/A'})
            continue

        # Normalize portrait (h > w) to landscape if requested
        if normalize_orientation and is_portrait:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            h, w = img.shape[:2]

        cur_size = (w, h)
        if reference_size is None:
            reference_size = cur_size
        elif cur_size != reference_size:
            records.append({'filename': p.name, 'status': 'SIZE_MISMATCH', 'shape': f"{w}x{h}", 'detector': 'N/A'})
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners, detector_name = detect_checkerboard_corners(gray, pattern)

        if ok:
            object_points.append(world_pts.copy())
            image_points.append(corners)
            accepted_paths.append(p)
            records.append({'filename': p.name, 'status': 'ACCEPTED', 'shape': f"{w}x{h}", 'detector': detector_name})
        else:
            records.append({'filename': p.name, 'status': 'CORNERS_NOT_FOUND', 'shape': f"{w}x{h}", 'detector': detector_name})

    if len(accepted_paths) < 3:
        raise RuntimeError(f"Only {len(accepted_paths)} accepted views. At least 3 views are required for calibration.")

    # Calibrate camera
    rms_api, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, reference_size, None, None
    )

    # Calculate per-view Euclidean reprojection error
    per_view_errors = []
    all_euclidean_errors = []

    for path, obj, img, rv, tv in zip(accepted_paths, object_points, image_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        errors = np.linalg.norm(img.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
        mean_err = float(np.mean(errors))
        rms_err = float(np.sqrt(np.mean(errors**2)))
        all_euclidean_errors.extend(errors.tolist())
        per_view_errors.append({
            'filename': path.name,
            'mean_error_px': mean_err,
            'rms_error_px': rms_err
        })

    total_mean_err = float(np.mean(all_euclidean_errors))
    total_rms_err = float(np.sqrt(np.mean(np.square(all_euclidean_errors))))

    results = {
        'K': K,
        'dist': dist,
        'image_size': reference_size,
        'pattern': pattern,
        'square_mm': square_mm,
        'num_total': len(image_paths),
        'num_accepted': len(accepted_paths),
        'opencv_rms': float(rms_api),
        'mean_reprojection_px': total_mean_err,
        'rms_reprojection_px': total_rms_err,
        'per_view_errors': per_view_errors,
        'records': records,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'accepted_paths': accepted_paths
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save calibration .npz
        np.savez(
            output_dir / 'calibration.npz',
            K=K, dist=dist, width=reference_size[0], height=reference_size[1],
            square_size_m=square_m, pattern=np.array(pattern),
            rms_api=rms_api, mean_error_px=total_mean_err, rms_error_px=total_rms_err
        )

        # Plot per-view reprojection error bar chart
        fig, ax = plt.subplots(figsize=(9, 4.5))
        view_names = [f"V{i+1}: {e['filename'][:15]}" for i, e in enumerate(per_view_errors)]
        mean_vals = [e['mean_error_px'] for e in per_view_errors]
        ax.bar(range(len(view_names)), mean_vals, color='#10b981', edgecolor='#065f46', alpha=0.85, width=0.6)
        ax.axhline(total_mean_err, color='#ef4444', linestyle='--', linewidth=1.5, label=f'Mean Error: {total_mean_err:.3f} px')
        ax.set_xticks(range(len(view_names)))
        ax.set_xticklabels(view_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Reprojection Error (pixels)')
        ax.set_title('Per-View Mean Reprojection Error Analysis', fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / 'reprojection_errors.png', dpi=150)
        plt.close(fig)

        # Plot 3D Extrinsics visualization
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')
        # Draw checkerboard grid
        grid_x, grid_y = np.meshgrid(np.arange(cols) * square_m, np.arange(rows) * square_m)
        grid_z = np.zeros_like(grid_x)
        ax.plot_wireframe(grid_x, grid_y, grid_z, color='gray', alpha=0.5, label='Checkerboard Plane (Z=0)')

        # Plot camera centers in world coordinates: C = -R^T * t
        cam_x, cam_y, cam_z = [], [], []
        for rv, tv in zip(rvecs, tvecs):
            R, _ = cv2.Rodrigues(rv)
            C = -np.dot(R.T, tv).flatten()
            cam_x.append(C[0])
            cam_y.append(C[1])
            cam_z.append(C[2])
            # Draw optical axis ray pointing towards origin
            ray_dir = np.dot(R.T, np.array([0, 0, 0.05]))
            ax.quiver(C[0], C[1], C[2], ray_dir[0], ray_dir[1], ray_dir[2], color='#3b82f6', length=0.03, normalize=True)

        ax.scatter(cam_x, cam_y, cam_z, c='#ef4444', s=40, label='Camera Centers (Views)')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('3D Camera Extrinsic Poses Relative to Checkerboard', fontsize=11, fontweight='bold')
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / 'extrinsics_3d.png', dpi=150)
        plt.close(fig)

    return results


def undistort_image(bgr: np.ndarray, K: np.ndarray, dist: np.ndarray, balance: float = 0.0):
    """Undistorts BGR image with option to compute optimal new camera matrix."""
    h, w = bgr.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), balance, (w, h))
    undistorted = cv2.undistort(bgr, K, dist, None, new_K)
    return undistorted, new_K
