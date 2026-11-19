# FCVG101 – Task 1A: Camera Calibration & Undistortion

## 1. Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Test the pipeline using synthetic data (NOT real experimental results)

```bash
python generate_synthetic_checkerboards.py
python calibration_task1a.py --input data/calibration --output results/demo --cols 9 --rows 6 --square-mm 18
```

This verifies the code without a physical camera. It does NOT count as actual camera calibration for Coursework.

## 3. Real camera calibration

- Print the 10×7 square checkerboard at 100% scale (18 mm/square for the provided print-ready PDF).
- Take 10–20 sharp, varied views from the same real camera, with unchanged optical configuration and same image resolution.
- Replace the synthetic JPGs in `data/calibration/` with real photographs. **Do not mix synthetic and real images.**
- Run:

```bash
python calibration_task1a.py --input data/calibration --output results/real --cols 9 --rows 6 --square-mm 18
```

To undistort another same-camera photo, add `--sample path/to/photo.jpg`.

## 4. Output evidence

- `results/real/corners/*_corners.jpg`: detected inner corners
- `results/real/calibration.npz`: K, dist, image resolution, calibration RMS
- `results/real/detection_log.csv`: accepted/rejected photos
- `results/real/view_errors.csv`: mean and RMS reprojection error per view (pixels)
- `results/real/metrics.txt`: actual values from images supplied
- `results/real/before_after.jpg`: undistortion side-by-side
- `results/real/undistorted.jpg`: corrected image

A low RMS alone does NOT ensure good calibration: check diversity of camera viewpoints, image coverage, corner overlays, the distortion coefficient sanity, and visually straightened lines.

The provided PNG/PDF board uses squares at 18 mm only when printed **Actual size / 100% / no fit to page**. Verify dimensions with a ruler before setting `--square-mm`.
