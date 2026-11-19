"""FCVG101 Coursework - Task 1.3: Feature Detection & Matching Pipeline.
Implements Harris Corner Detector from scratch (Structure tensor M, corner response R, NMS),
compares with OpenCV cornerHarris, and performs SIFT/ORB matching with Lowe's ratio test and RANSAC Homography.
"""
from typing import Tuple, List, Dict
import cv2
import numpy as np


def harris_corner_detector_scratch(
    gray: np.ndarray,
    k: float = 0.04,
    ksize_sobel: int = 3,
    sigma_gaussian: float = 1.5,
    threshold_rel: float = 0.01,
    nms_radius: int = 3
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Implements Harris Corner Detector from scratch using mathematical definitions.
    
    Returns:
        response_R: Continuous 2D response map R
        keypoints: List of (x, y) coordinates of detected corners
    """
    gray_f = gray.astype(np.float64)

    # 1. Image derivatives using Sobel
    Ix = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=ksize_sobel)
    Iy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=ksize_sobel)

    # 2. Products of derivatives
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # 3. Gaussian weighting (Structure tensor elements)
    # Kernel size derived from sigma
    ksize_g = int(2 * np.ceil(2 * sigma_gaussian) + 1)
    Sxx = cv2.GaussianBlur(Ixx, (ksize_g, ksize_g), sigma_gaussian)
    Syy = cv2.GaussianBlur(Iyy, (ksize_g, ksize_g), sigma_gaussian)
    Sxy = cv2.GaussianBlur(Ixy, (ksize_g, ksize_g), sigma_gaussian)

    # 4. Corner response R = det(M) - k * (trace(M))^2
    det_M = (Sxx * Syy) - (Sxy ** 2)
    trace_M = Sxx + Syy
    R = det_M - k * (trace_M ** 2)

    # 5. Thresholding
    thresh = threshold_rel * R.max()
    R_thresh = np.where(R > thresh, R, 0.0)

    # 6. Non-Maximum Suppression (NMS)
    # Using maximum filter (dilation)
    kernel_nms = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * nms_radius + 1, 2 * nms_radius + 1))
    R_dilated = cv2.dilate(R_thresh, kernel_nms)
    local_max = (R_thresh == R_dilated) & (R_thresh > 0)

    # Extract coordinates (x = col, y = row)
    corners_y, corners_x = np.where(local_max)
    keypoints = list(zip(corners_x.tolist(), corners_y.tolist()))

    return R, keypoints


def compare_harris_scratch_vs_opencv(
    gray: np.ndarray,
    k: float = 0.04,
    threshold_rel: float = 0.01,
    nms_radius: int = 3
) -> Dict:
    """Compares scratch Harris implementation with OpenCV cv2.cornerHarris."""
    # Scratch
    R_scratch, kps_scratch = harris_corner_detector_scratch(
        gray, k=k, threshold_rel=threshold_rel, nms_radius=nms_radius
    )

    # OpenCV
    R_cv = cv2.cornerHarris(gray, blockSize=3, ksize=3, k=k)
    thresh_cv = threshold_rel * R_cv.max()
    kernel_nms = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * nms_radius + 1, 2 * nms_radius + 1))
    R_cv_dilated = cv2.dilate(R_cv, kernel_nms)
    local_max_cv = (R_cv == R_cv_dilated) & (R_cv > thresh_cv)
    cv_y, cv_x = np.where(local_max_cv)
    kps_opencv = list(zip(cv_x.tolist(), cv_y.tolist()))

    # Spatial match agreement within 3 pixels
    if kps_scratch and kps_opencv:
        pts_s = np.array(kps_scratch)
        pts_c = np.array(kps_opencv)
        dists = np.min(np.sqrt(np.sum((pts_s[:, None, :] - pts_c[None, :, :]) ** 2, axis=2)), axis=1)
        spatial_agreement = float(np.mean(dists <= 3.0))
    else:
        spatial_agreement = 0.0

    # Create visualization
    bgr_scratch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y in kps_scratch:
        cv2.circle(bgr_scratch, (x, y), 3, (0, 255, 0), -1)  # Green for scratch

    bgr_opencv = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y in kps_opencv:
        cv2.circle(bgr_opencv, (x, y), 3, (0, 0, 255), -1)  # Red for OpenCV

    return {
        'num_scratch': len(kps_scratch),
        'num_opencv': len(kps_opencv),
        'spatial_agreement': spatial_agreement,
        'kps_scratch': kps_scratch,
        'kps_opencv': kps_opencv,
        'vis_scratch': bgr_scratch,
        'vis_opencv': bgr_opencv,
        'R_scratch': R_scratch,
        'R_opencv': R_cv
    }


def normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley normalization: centers points at origin and scales average distance to sqrt(2)."""
    mean = np.mean(pts, axis=0)
    shifted = pts - mean
    avg_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    scale = np.sqrt(2) / avg_dist if avg_dist > 1e-8 else 1.0
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    pts_homo = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
    norm_pts = (T @ pts_homo.T).T[:, :2]
    return norm_pts, T


def estimate_homography_dlt(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Estimates 3x3 Homography H from >= 4 point correspondences using DLT with Hartley normalization.
    Solves Ah = 0 via Singular Value Decomposition (SVD).
    """
    src = np.asarray(src_pts, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_pts, dtype=np.float64).reshape(-1, 2)
    n = src.shape[0]
    if n < 4:
        raise ValueError(f"DLT Homography requires at least 4 point correspondences, got {n}.")

    # Hartley normalization
    norm_src, T_src = normalize_points(src)
    norm_dst, T_dst = normalize_points(dst)

    # Build 2N x 9 matrix A
    A = np.zeros((2 * n, 9), dtype=np.float64)
    for i in range(n):
        xs, ys = norm_src[i, 0], norm_src[i, 1]
        xd, yd = norm_dst[i, 0], norm_dst[i, 1]
        A[2 * i] = [-xs, -ys, -1.0, 0.0, 0.0, 0.0, xd * xs, xd * ys, xd]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -xs, -ys, -1.0, yd * xs, yd * ys, yd]

    # SVD: A = U * S * Vt
    _, _, vt = np.linalg.svd(A)
    H_norm = vt[-1].reshape(3, 3)

    # De-normalization: H = inv(T_dst) @ H_norm @ T_src
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]
    return H


def match_features_ransac(
    img1: np.ndarray,
    img2: np.ndarray,
    method: str = 'SIFT',
    ratio_thresh: float = 0.75,
    ransac_thresh_px: float = 3.0
) -> Dict:
    """Detects features (SIFT or ORB), matches using Lowe's ratio test,

    and performs geometric verification via RANSAC Homography.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    if method.upper() == 'SIFT':
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    elif method.upper() == 'ORB':
        detector = cv2.ORB_create(nfeatures=1500)
        norm_type = cv2.NORM_HAMMING
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'SIFT' or 'ORB'.")

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return {
            'success': False,
            'message': 'Insufficient features detected in one or both images.',
            'num_kp1': len(kp1) if kp1 else 0,
            'num_kp2': len(kp2) if kp2 else 0,
            'num_matches': 0,
            'num_inliers': 0,
            'inlier_ratio': 0.0
        }

    # BFMatcher with k-Nearest Neighbors (k=2)
    bf = cv2.BFMatcher(norm_type)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    if len(good_matches) < 4:
        return {
            'success': False,
            'message': f"Only {len(good_matches)} passed ratio test; >= 4 required for Homography.",
            'num_kp1': len(kp1),
            'num_kp2': len(kp2),
            'num_matches': len(good_matches),
            'num_inliers': 0,
            'inlier_ratio': 0.0
        }

    # Extract coordinates of matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC Homography estimation
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh_px)
    inliers_mask = mask.ravel().tolist() if mask is not None else [0] * len(good_matches)
    inlier_matches = [m for m, is_inlier in zip(good_matches, inliers_mask) if is_inlier]

    inlier_ratio = float(len(inlier_matches) / len(good_matches)) if good_matches else 0.0

    # Draw all good matches (gray/yellow) vs inlier matches (green)
    draw_params_inliers = dict(
        matchColor=(0, 220, 0),
        singlePointColor=None,
        matchesMask=inliers_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    vis_inliers = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params_inliers)

    draw_params_all = dict(
        matchColor=(0, 150, 255),
        singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    vis_all_good = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params_all)

    return {
        'success': True,
        'method': method.upper(),
        'num_kp1': len(kp1),
        'num_kp2': len(kp2),
        'num_good_matches': len(good_matches),
        'num_inliers': len(inlier_matches),
        'inlier_ratio': inlier_ratio,
        'homography_matrix': H,
        'vis_inliers': vis_inliers,
        'vis_all_good': vis_all_good,
        'kp1': kp1,
        'kp2': kp2,
        'inlier_matches': inlier_matches
    }
