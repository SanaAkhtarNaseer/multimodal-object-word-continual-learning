import os
import kagglehub


DATASET_HANDLE = "kritikseth/fruit-and-vegetable-image-recognition"


# Download dataset
dataset_path = kagglehub.dataset_download(DATASET_HANDLE)

print("\nDataset path:")
print(dataset_path)


# Inspect top-level structure
print("\nTop-level contents:")

for item in sorted(os.listdir(dataset_path)):
    full_path = os.path.join(dataset_path, item)

    if os.path.isdir(full_path):
        print("[DIR ]", item)
    else:
        print("[FILE]", item)


# Search one level deeper for class folders
print("\nPossible class folders:")

for root, dirs, files in os.walk(dataset_path):

    image_files = [
        f for f in files
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    ]

    if image_files:
        print(
            os.path.relpath(root, dataset_path),
            "->",
            len(image_files),
            "images"
        )