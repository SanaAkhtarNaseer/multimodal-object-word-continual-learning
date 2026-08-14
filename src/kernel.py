import numpy as np


def gaussian_kernel(x, y, sigma):
    """
    Compute Gaussian kernel similarity between two scalar values.

    sigma : float
        Kernel bandwidth.

    Returns:  Similarity value between 0 and 1.
    """

    difference = x - y

    squared_difference = difference ** 2

    denominator = 2 * (sigma ** 2)

    kernel_value = np.exp(
        -squared_difference / denominator
    )

    return kernel_value


x = 2.0
y = 4.0

sigma_values = [0.5, 1.0, 2.0, 5.0]

for sigma in sigma_values:
    similarity = gaussian_kernel(x, y, sigma)

    print(
        "sigma:", sigma,
        " similarity:", similarity
    )
