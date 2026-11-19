"""FCVG101 Coursework - Task 1.2: Filtering, Gradients & Frequency Domain Analysis.
Implements spatial filtering (Box, Gaussian, Median), quantitative evaluation (MSE & PSNR),
Sobel & Canny edge detection, and 2D FFT with Ideal vs Gaussian LPF (Gibbs ringing analysis).
"""
import cv2
import numpy as np


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, std: float = 25.0, seed: int = 42) -> np.ndarray:
    """Adds zero-mean Gaussian noise to an image."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(mean, std, image.shape)
    noisy = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_salt_and_pepper_noise(image: np.ndarray, amount: float = 0.05, salt_ratio: float = 0.5, seed: int = 42) -> np.ndarray:
    """Adds impulse (Salt & Pepper) noise to an image."""
    rng = np.random.RandomState(seed)
    noisy = image.copy()
    num_pixels = int(amount * image.shape[0] * image.shape[1])
    
    # Salt (white pixels)
    num_salt = int(num_pixels * salt_ratio)
    coords_salt_y = rng.randint(0, image.shape[0], num_salt)
    coords_salt_x = rng.randint(0, image.shape[1], num_salt)
    noisy[coords_salt_y, coords_salt_x] = 255

    # Pepper (black pixels)
    num_pepper = num_pixels - num_salt
    coords_pep_y = rng.randint(0, image.shape[0], num_pepper)
    coords_pep_x = rng.randint(0, image.shape[1], num_pepper)
    noisy[coords_pep_y, coords_pep_x] = 0

    return noisy


def compute_mse(img_clean: np.ndarray, img_filtered: np.ndarray) -> float:
    """Calculates Mean Squared Error between clean reference and filtered image."""
    diff = img_clean.astype(np.float64) - img_filtered.astype(np.float64)
    return float(np.mean(diff ** 2))


def compute_psnr(img_clean: np.ndarray, img_filtered: np.ndarray) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR) in decibels (dB)."""
    mse = compute_mse(img_clean, img_filtered)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return float(20.0 * np.log10(max_pixel / np.sqrt(mse)))


def apply_spatial_filters(image: np.ndarray, ksize: int = 5, gaussian_sigma: float = 1.0) -> dict:
    """Applies Box, Gaussian, and Median filters to an input image."""
    if ksize % 2 == 0:
        ksize += 1
    
    box_filtered = cv2.blur(image, (ksize, ksize))
    gauss_filtered = cv2.GaussianBlur(image, (ksize, ksize), gaussian_sigma)
    median_filtered = cv2.medianBlur(image, ksize)
    
    return {
        'box': box_filtered,
        'gaussian': gauss_filtered,
        'median': median_filtered
    }


def compute_gradients_and_edges(gray: np.ndarray, canny_low: int = 50, canny_high: int = 150):
    """Computes Sobel gradients (Ix, Iy, magnitude, direction) and Canny edges."""
    # Sobel gradients (16-bit signed to prevent underflow)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    direction = np.arctan2(sobel_y, sobel_x)  # in radians [-pi, pi]
    
    # Scale magnitude for visualization (0-255 uint8)
    mag_scaled = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Canny edge detector
    canny_edges = cv2.Canny(gray, canny_low, canny_high)
    
    return {
        'sobel_x': sobel_x,
        'sobel_y': sobel_y,
        'magnitude': magnitude,
        'magnitude_visual': mag_scaled,
        'direction': direction,
        'canny': canny_edges
    }


def compute_fft_2d(gray: np.ndarray):
    """Computes 2D FFT, centered frequency spectrum and log-magnitude spectrum."""
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1.0)
    return fshift, magnitude_spectrum


def create_frequency_masks(shape: tuple, cutoff_d0: float = 40.0):
    """Creates Ideal Low-Pass Filter (ILPF) and Gaussian Low-Pass Filter (GLPF) masks."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    
    # Ideal Low-Pass Filter: step function
    ideal_lpf = (D <= cutoff_d0).astype(np.float64)
    
    # Gaussian Low-Pass Filter: smooth exponential
    gauss_lpf = np.exp(-(D**2) / (2.0 * (cutoff_d0**2)))
    
    # Ideal High-Pass Filter: 1 - ILPF
    ideal_hpf = 1.0 - ideal_lpf
    
    # Gaussian High-Pass Filter: 1 - GLPF
    gauss_hpf = 1.0 - gauss_lpf
    
    return {
        'ideal_lpf': ideal_lpf,
        'gauss_lpf': gauss_lpf,
        'ideal_hpf': ideal_hpf,
        'gauss_hpf': gauss_hpf,
        'distance_matrix': D
    }


def apply_frequency_filter(fshift: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Applies frequency domain filter mask and computes inverse 2D FFT."""
    filtered_fshift = fshift * mask
    f_ishift = np.fft.ifftshift(filtered_fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return np.clip(img_back, 0, 255).astype(np.uint8)


def evaluate_filters_on_noise(clean_gray: np.ndarray, ksize: int = 5) -> dict:
    """Systematically evaluates Box, Gaussian, and Median filters on Gaussian vs S&P noise."""
    noisy_gauss = add_gaussian_noise(clean_gray, std=25.0)
    noisy_sp = add_salt_and_pepper_noise(clean_gray, amount=0.08)
    
    filters_gauss = apply_spatial_filters(noisy_gauss, ksize=ksize)
    filters_sp = apply_spatial_filters(noisy_sp, ksize=ksize)
    
    eval_table = []
    # Baseline noisy
    eval_table.append({
        'noise_type': 'Gaussian Noise (sigma=25)',
        'filter_applied': 'None (Raw Noisy)',
        'mse': compute_mse(clean_gray, noisy_gauss),
        'psnr': compute_psnr(clean_gray, noisy_gauss)
    })
    for fname, filtered_img in filters_gauss.items():
        eval_table.append({
            'noise_type': 'Gaussian Noise (sigma=25)',
            'filter_applied': fname.capitalize(),
            'mse': compute_mse(clean_gray, filtered_img),
            'psnr': compute_psnr(clean_gray, filtered_img)
        })
        
    eval_table.append({
        'noise_type': 'Salt & Pepper Noise (amount=8%)',
        'filter_applied': 'None (Raw Noisy)',
        'mse': compute_mse(clean_gray, noisy_sp),
        'psnr': compute_psnr(clean_gray, noisy_sp)
    })
    for fname, filtered_img in filters_sp.items():
        eval_table.append({
            'noise_type': 'Salt & Pepper Noise (amount=8%)',
            'filter_applied': fname.capitalize(),
            'mse': compute_mse(clean_gray, filtered_img),
            'psnr': compute_psnr(clean_gray, filtered_img)
        })
        
    return {
        'noisy_gauss': noisy_gauss,
        'noisy_sp': noisy_sp,
        'filters_gauss': filters_gauss,
        'filters_sp': filters_sp,
        'metrics_table': eval_table
    }
