import cv2
import numpy as np


SHAPE_DIM = 23

COLOR_BINS_PER_CHANNEL = 21
COLOR_DIM = COLOR_BINS_PER_CHANNEL * 3


# ---------------------------------------------------------
# Foreground segmentation
# ---------------------------------------------------------

def create_object_mask(image):
    """
    Estimate the foreground object using GrabCut.

    This is our reconstruction because the CDSN paper does
    not fully specify the segmentation algorithm used before
    extracting outline and color features.
    """

    if image is None:
        raise ValueError("Input image is None.")

    height, width = image.shape[:2]

    max_side = 400

    if max(height, width) > max_side:
        scale = max_side / max(height, width)

        small_image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        small_image = image.copy()
        scale = 1.0

    small_height, small_width = small_image.shape[:2]    

    if height < 10 or width < 10:
        raise ValueError("Image is too small.")

    mask = np.zeros(
    small_image.shape[:2],
    dtype=np.uint8
)
    background_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    # Leave a small border around the image
    margin_x = max(1, int(small_width * 0.05))
    margin_y = max(1, int(small_height * 0.05))


    rect = (
    margin_x,
    margin_y,
    small_width - (2 * margin_x),
    small_height - (2 * margin_y)
)
    cv2.grabCut(
        small_image,
        mask,
        rect,
        background_model,
        foreground_model,
        1,
        cv2.GC_INIT_WITH_RECT
    )

    # GrabCut labels:
    # 0 = background
    # 1 = foreground
    # 2 = probable background
    # 3 = probable foreground
    binary_mask = np.where(
        (mask == cv2.GC_FGD)
        | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    # Clean small segmentation artifacts
    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    if scale != 1.0:
        binary_mask = cv2.resize(
            binary_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

    return binary_mask


# ---------------------------------------------------------
# Shape feature
# ---------------------------------------------------------

def extract_shape_descriptor(mask):
    """
    Extract a 23-dimensional normalized Fourier descriptor
    from the largest foreground contour.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        raise ValueError(
            "No foreground contour was detected."
        )

    contour = max(
        contours,
        key=cv2.contourArea
    )

    if len(contour) < 4:
        raise ValueError(
            "Detected contour is too small."
        )

    points = contour[:, 0, :].astype(
        np.float64
    )

    # Represent x/y coordinates as complex values
    contour_complex = (
        points[:, 0]
        + 1j * points[:, 1]
    )

    # Translation invariance
    contour_complex -= np.mean(
        contour_complex
    )

    fourier = np.fft.fft(
        contour_complex
    )

    # Ignore DC coefficient
    descriptor = np.abs(
        fourier[1:SHAPE_DIM + 1]
    )

    # Rare short-contour case
    if descriptor.size < SHAPE_DIM:
        descriptor = np.pad(
            descriptor,
            (
                0,
                SHAPE_DIM - descriptor.size
            )
        )

    # Scale normalization
    norm = np.linalg.norm(descriptor)

    if norm > 0:
        descriptor = descriptor / norm

    return descriptor.astype(
        np.float64
    )


# ---------------------------------------------------------
# Color feature
# ---------------------------------------------------------

def extract_color_descriptor(image, mask):
    """
    Produce a 63-dimensional color descriptor.

    Reconstruction choice:
        21 bins for each B, G and R channel

        21 x 3 = 63 dimensions
    """

    descriptors = []

    for channel in range(3):

        histogram = cv2.calcHist(
            [image],
            [channel],
            mask,
            [COLOR_BINS_PER_CHANNEL],
            [0, 256]
        )

        histogram = histogram.flatten().astype(
            np.float64
        )

        total = histogram.sum()

        if total > 0:
            histogram /= total

        descriptors.extend(
            histogram.tolist()
        )

    descriptor = np.asarray(
        descriptors,
        dtype=np.float64
    )

    if descriptor.shape != (COLOR_DIM,):
        raise RuntimeError(
            "Unexpected color descriptor dimension."
        )

    return descriptor


# ---------------------------------------------------------
# Complete extraction
# ---------------------------------------------------------

def extract_image_features(image_path):
    """
    Extract CDSN-compatible features from one real image.

    Returns
    -------
    shape_vector : ndarray (23,)
    color_vector : ndarray (63,)
    mask         : binary object mask
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    mask = create_object_mask(image)

    shape_vector = extract_shape_descriptor(
        mask
    )

    color_vector = extract_color_descriptor(
        image,
        mask
    )

    if shape_vector.shape != (SHAPE_DIM,):
        raise RuntimeError(
            f"Shape should be ({SHAPE_DIM},), "
            f"got {shape_vector.shape}"
        )

    if color_vector.shape != (COLOR_DIM,):
        raise RuntimeError(
            f"Color should be ({COLOR_DIM},), "
            f"got {color_vector.shape}"
        )

    return shape_vector, color_vector, mask