"""FCVG101 Task 1A: checkerboard camera calibration and undistortion.

Usage:
  python calibration_task1a.py --input data/calibration --cols 9 --rows 6 --square-mm 18
  python calibration_task1a.py --input data/calibration --sample my_photo.jpg

All printed K, distortion and error values derive from actual input images; do not
report synthetic demonstration results as real-camera measurements.
"""
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def load_paths(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f'Input folder not found: {folder.resolve()}')
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)


def find_inner_corners(gray, pattern):
    # Prefer robust SB detector if available; the classic method has a subpixel refinement.
    if hasattr(cv2, 'findChessboardCornersSB'):
        ok, corners = cv2.findChessboardCornersSB(gray, pattern)
        if ok:
            return True, corners.astype(np.float32), 'SB'
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return False, None, '-'
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, .001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners, 'classic + subpixel'


def run(args):
    input_folder = Path(args.input)
    result_dir = Path(args.output)
    result_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = result_dir / 'corners'
    overlay_dir.mkdir(exist_ok=True)
    paths = load_paths(input_folder)
    if not paths:
        raise RuntimeError('No images found. Put checkerboard JPG/PNG files in data/calibration/')

    pattern = (args.cols, args.rows)  # (columns of INNER corners, rows of INNER corners)
    square_m = args.square_mm / 1000.0
    world = np.zeros((args.cols * args.rows, 3), np.float32)
    world[:, :2] = (np.mgrid[:args.cols, :args.rows].T.reshape(-1, 2) * square_m)
    object_points, image_points, accepted = [], [], []
    records = []
    image_size = None

    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            records.append((path.name, 'UNREADABLE', '-', '')); continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cur_size = (gray.shape[1], gray.shape[0])  # OpenCV: (width, height)
        if image_size is None:
            image_size = cur_size
        if cur_size != image_size:
            raise ValueError(f'Mixed image sizes: {path.name} is {cur_size}, expected {image_size}; '
                             'do not silently resize calibration photos.')
        ok, corners, detector = find_inner_corners(gray, pattern)
        if not ok:
            records.append((path.name, 'REJECTED', '-', 'no complete inner-corner grid'))
            print('[SKIP] corners not found:', path.name)
            continue
        object_points.append(world.copy())
        image_points.append(corners)
        accepted.append(path)
        vis = bgr.copy()
        cv2.drawChessboardCorners(vis, pattern, corners, True)
        cv2.imwrite(str(overlay_dir / (path.stem + '_corners.jpg')), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        records.append((path.name, 'ACCEPTED', detector, ''))
        print('[OK] corners detected:', path.name, f'({detector})')

    if len(accepted) < 3:
        raise RuntimeError(f'Only {len(accepted)} usable views. 10–20 diverse views '
                           'recommended; recheck checkerboard inner corners and lighting.')

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None)
    if not np.isfinite(K).all() or not np.isfinite(dist).all():
        raise RuntimeError('Calibration produced invalid parameters; inspect views.')

    # Mean Euclidean reprojection distance (in pixels) for each accepted view.
    all_distances = []
    view_stats = []
    for path, ob, im, rv, tv in zip(accepted, object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(ob, rv, tv, K, dist)
        distances = np.linalg.norm(im.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
        all_distances.extend(distances.tolist())
        view_stats.append((path.name, float(distances.mean()), float(np.sqrt(np.mean(distances**2)))))

    mean_euclidean_px = float(np.mean(all_distances))
    rms_euclidean_px = float(np.sqrt(np.mean(np.square(all_distances))))
    out_npz = result_dir / 'calibration.npz'
    np.savez(out_npz, K=K, dist=dist, width=image_size[0], height=image_size[1],
             square_size_m=square_m, pattern_size=np.array(pattern), rms_api=rms,
             mean_euclidean_px=mean_euclidean_px)

    with (result_dir / 'view_errors.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file_name', 'mean_reprojection_error_px', 'rms_euclidean_reprojection_error_px'])
        w.writerows(view_stats)
    with (result_dir / 'detection_log.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['file_name', 'status', 'detector', 'notes']); w.writerows(records)

    sample = Path(args.sample) if args.sample else accepted[0]
    img = cv2.imread(str(sample))
    if img is None:
        raise FileNotFoundError(f'Sample image unreadable: {sample}')
    if (img.shape[1], img.shape[0]) != image_size:
        raise ValueError('Sample image resolution differs from calibration views; '
                         'reuse same camera/resolution or explicitly scale K.')
    # Keep output uncropped for an apples-to-apples before/after comparison.
    undistorted = cv2.undistort(img, K, dist)
    cv2.imwrite(str(result_dir / 'undistorted.jpg'), undistorted,
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    both = np.concatenate([img, undistorted], axis=1)
    cv2.putText(both, 'BEFORE', (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(both, 'AFTER', (img.shape[1] + 16, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 160, 0), 2)
    cv2.imwrite(str(result_dir / 'before_after.jpg'), both,
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    report = (
        'FCVG101 Task 1A - Camera calibration results\n'
        'Input: '+str(input_folder)+'\n'
        f'Accepted/total: {len(accepted)}/{len(paths)}\n'
        f'Pattern (INNER corners): {pattern[0]} x {pattern[1]}\n'
        f'Square side length: {args.square_mm:g} mm\n'
        f'Image size (width x height): {image_size[0]} x {image_size[1]} px\n'
        f'OpenCV calibration RMS reprojection error: {rms:.6f} px\n'
        f'Mean Euclidean reprojection error: {mean_euclidean_px:.6f} px\n'
        f'RMS Euclidean reprojection error: {rms_euclidean_px:.6f} px\n'
        'K =\n'+str(K)+'\n'
        'dist = '+str(dist.ravel())+'\n'
        'NOTE: Synthetic demo values must not be presented as real-camera measurements.\n'
    )
    (result_dir / 'metrics.txt').write_text(report, encoding='utf-8')
    print('\n'+report)
    print('Saved outputs to:', result_dir.resolve())
    return K, dist, rms


def main():
    p = argparse.ArgumentParser(description='FCVG101 - calibrate camera with checkerboard views')
    p.add_argument('--input', default='data/calibration', help='folder of photos from SAME camera')
    p.add_argument('--output', default='results', help='output folder')
    p.add_argument('--sample', default=None, help='same-camera image for undistortion demo')
    p.add_argument('--cols', type=int, default=9, help='number of INNER corners horizontally')
    p.add_argument('--rows', type=int, default=6, help='number of INNER corners vertically')
    p.add_argument('--square-mm', type=float, default=18.0, help='physical square size in mm')
    args = p.parse_args()
    if args.cols < 3 or args.rows < 3 or args.square_mm <= 0:
        p.error('cols/rows must be >= 3 and square-mm > 0')
    run(args)

if __name__ == '__main__':
    main()
