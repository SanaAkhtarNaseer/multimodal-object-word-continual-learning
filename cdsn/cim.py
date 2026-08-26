from __future__ import annotations

import numpy as np

__all__ = ["gaussian_kernel", "cim", "cim_batch", "silverman_bandwidth"]

# kappa_sigma(0) = exp(0) = 1 under the un-normalised Gaussian kernel.
KAPPA_ZERO = 1.0


def gaussian_kernel(e: np.ndarray, sigma: float | np.ndarray) -> np.ndarray:
    
    sigma = np.asarray(sigma, dtype=float)
    # Guard against a degenerate bandwidth collapsing the kernel to a delta.
    sigma = np.maximum(sigma, 1e-12)
    return np.exp(-(np.asarray(e, dtype=float) ** 2) / (2.0 * sigma**2))


def cim(u: np.ndarray, v: np.ndarray, sigma: float | np.ndarray) -> float:
  
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    delta_hat = float(np.mean(gaussian_kernel(u - v, sigma)))
    # Clamp for numerical safety: floating point can push this a hair below 0.
    return float(np.sqrt(max(KAPPA_ZERO - delta_hat, 0.0)))


def cim_batch(u: np.ndarray, V: np.ndarray, sigma: float | np.ndarray) -> np.ndarray:
  
    u = np.asarray(u, dtype=float)
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        V = V.reshape(1, -1)

    diff = V - u[None, :]                       # (K, D)
    sigma_arr = np.asarray(sigma, dtype=float)
    if sigma_arr.ndim == 1:
        sigma_arr = sigma_arr[:, None]          # (K, 1) -> broadcast over D
    delta_hat = np.mean(gaussian_kernel(diff, sigma_arr), axis=1)   # (K,)
    return np.sqrt(np.maximum(KAPPA_ZERO - delta_hat, 0.0))


def silverman_bandwidth(X: np.ndarray) -> float:
   
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, d = X.shape
    if n < 2:
        return 0.1
    std = np.std(X, axis=0, ddof=1)
    std = std[std > 1e-12]
    if std.size == 0:
        return 0.1
    # Silverman: h = (4 / (d + 2))^(1/(d+4)) * n^(-1/(d+4)) * std
    factor = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0)) * n ** (-1.0 / (d + 4.0))
    return float(np.mean(std) * factor)
