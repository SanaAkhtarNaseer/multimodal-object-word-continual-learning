import numpy as np


def gaussian_kernel(x, y, sigma):
    """
    Compute element-wise Gaussian kernel similarity.
    """
    return np.exp(-((x - y) ** 2) / (2 * sigma ** 2))


def correntropy(x, y, sigma):
    """
    Compute sample correntropy between two vectors.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    kernel_values = gaussian_kernel(x, y, sigma)

    return np.mean(kernel_values)


def cim(x, y, sigma):
    """
    Compute the Correntropy Induced Metric (CIM).
    """

    similarity = correntropy(x, y, sigma)

    distance = np.sqrt(1.0 - similarity)

    return distance


if __name__ == "__main__":

    x = np.array([1.0, 2.0, 3.0])

    y_similar = np.array([1.2, 2.5, 2.8])

    y_different = np.array([5.0, 8.0, 10.0])

    sigma = 1.0

    cim_similar = cim(x, y_similar, sigma)
    cim_different = cim(x, y_different, sigma)

    print("CIM for similar vectors:", cim_similar)
    print("CIM for different vectors:", cim_different)