import numpy as np


def gaussian_kernel(x, y, sigma):
    """
    Compute element-wise Gaussian kernel similarity.
    """
    return np.exp(-((x - y) ** 2) / (2 * sigma ** 2))


def correntropy(x, y, sigma):
    """
    Compute correntropy between two vectors.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    kernel_values = gaussian_kernel(x, y, sigma)

    return np.mean(kernel_values)


if __name__ == "__main__":

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.2, 2.5, 2.8])

    sigma = 1.0

    similarity = correntropy(x, y, sigma)

    print("Vector x:", x)
    print("Vector y:", y)
    print("Sigma:", sigma)
    print("Correntropy:", similarity)
    