from __future__ import annotations

import numpy as np

__all__ = [
    "SHAPE_DIM", "COLOR_DIM",
    "segment_object", "fourier_descriptors", "color_histogram",
    "extract_visual_features",
]

SHAPE_DIM = 23
COLOR_DIM = 63


def segment_object(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    import cv2

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    # Object pixels are either saturated (coloured) or bright against the backdrop.
    _, m1 = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, m2 = cv2.threshold(val, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_or(m1, m2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask, None
    return mask, max(contours, key=cv2.contourArea)


def fourier_descriptors(contour: np.ndarray, n_descriptors: int = SHAPE_DIM,
                        n_resample: int = 256) -> np.ndarray:
    
    pts = np.asarray(contour, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 4:
        return np.zeros(n_descriptors)

    # Resample the contour to a fixed length so the FFT bins are comparable
    # across objects with different perimeter lengths.
    idx = np.linspace(0, pts.shape[0] - 1, n_resample)
    xs = np.interp(idx, np.arange(pts.shape[0]), pts[:, 0])
    ys = np.interp(idx, np.arange(pts.shape[0]), pts[:, 1])

    complex_contour = xs + 1j * ys
    coeffs = np.fft.fft(complex_contour)

    mags = np.abs(coeffs)
    if mags[1] < 1e-12:
        return np.zeros(n_descriptors)
    descriptors = mags[2:2 + n_descriptors] / mags[1]    # drop DC and normalise

    out = np.zeros(n_descriptors)
    out[:len(descriptors)] = descriptors
    return np.clip(out, 0.0, 1.0)


def color_histogram(image_bgr: np.ndarray, mask: np.ndarray | None = None,
                    bins: tuple[int, int, int] = (7, 3, 3)) -> np.ndarray:
   
    import cv2

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, list(bins),
                        [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(float)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def extract_visual_features(image_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Full visual front-end: image -> {shape: (23,), color: (63,)}."""
    mask, contour = segment_object(image_bgr)
    shape = (fourier_descriptors(contour) if contour is not None
             else np.zeros(SHAPE_DIM))
    color = color_histogram(image_bgr, mask)
    return {"shape": shape, "color": color}
