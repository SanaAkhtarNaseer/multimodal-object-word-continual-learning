import os
import sys
import cv2
import kagglehub

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.image_feature_extractor import extract_image_features


DATASET_HANDLE = "kritikseth/fruit-and-vegetable-image-recognition"

dataset_path = kagglehub.dataset_download(DATASET_HANDLE)

apple_dir = os.path.join(
    dataset_path,
    "train",
    "apple"
)

image_files = [
    name
    for name in os.listdir(apple_dir)
    if name.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )
]

if not image_files:
    raise RuntimeError("No apple images found.")

image_path = os.path.join(
    apple_dir,
    sorted(image_files)[0]
)

print("Using image:")
print(image_path)

shape_vector, color_vector, mask = extract_image_features(
    image_path
)

print("\nShape dimension:", shape_vector.shape)
print("Color dimension:", color_vector.shape)

print("\nShape vector:")
print(shape_vector)

print("\nFirst 10 color values:")
print(color_vector[:10])

output_mask = os.path.join(
    "results",
    "apple_test_mask.png"
)

os.makedirs("results", exist_ok=True)

cv2.imwrite(
    output_mask,
    mask
)

print("\nSaved mask:")
print(output_mask)

print("\nFeature extraction test completed.")