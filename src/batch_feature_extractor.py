import os
import csv
import kagglehub

from src.image_feature_extractor import extract_image_features



DATASET_HANDLE = "kritikseth/fruit-and-vegetable-image-recognition"

SELECTED_CLASSES = [
    "apple",
    "banana",
    "beetroot",
    "bell pepper",
    "carrot",
    "corn",
    "cucumber",
    "eggplant",
    "garlic",
    "lemon",
    "mango",
    "onion",
    "orange",
    "pear",
    "pineapple",
    "pomegranate",
    "potato",
    "raddish",
    "tomato",
    "watermelon",
]


def extract_split(split_name, output_csv):
    """
    Extract 23-D shape + 63-D color features
    for one Kaggle split.
    """

    dataset_path = kagglehub.dataset_download(
        DATASET_HANDLE
    )

    fieldnames = (
        ["name", "image_path"]
        + [f"shape_{i}" for i in range(23)]
        + [f"color_{i}" for i in range(63)]
    )

    successful = 0
    failed = 0

    os.makedirs(
        os.path.dirname(output_csv),
        exist_ok=True
    )

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for class_name in SELECTED_CLASSES:

            class_dir = os.path.join(
                dataset_path,
                split_name,
                class_name
            )

            if not os.path.isdir(class_dir):
                print(
                    f"[WARNING] Missing class folder: "
                    f"{class_dir}"
                )
                continue

            image_files = sorted(
                [
                    filename
                    for filename in os.listdir(class_dir)
                    if filename.lower().endswith(
                        (
                            ".jpg",
                            ".jpeg",
                            ".png",
                            ".bmp",
                            ".webp",
                        )
                    )
                ]
            )

            print(
                f"\nProcessing {split_name}/{class_name} "
                f"({len(image_files)} images)"
            )

            for filename in image_files:

                image_path = os.path.join(
                    class_dir,
                    filename
                )

                try:
                    shape, color, _ = (
                        extract_image_features(
                            image_path
                        )
                    )

                    row = {
                        "name": class_name,
                        "image_path": image_path,
                    }

                    for i, value in enumerate(shape):
                        row[f"shape_{i}"] = float(value)

                    for i, value in enumerate(color):
                        row[f"color_{i}"] = float(value)

                    writer.writerow(row)

                    successful += 1

                except Exception as error:

                    failed += 1

                    print(
                        "[FAILED]",
                        image_path,
                        "->",
                        error
                    )

    print("\n================================")
    print(f"{split_name.upper()} EXTRACTION SUMMARY")
    print("================================")
    print("Successful:", successful)
    print("Failed:", failed)
    print("Output:", output_csv)

    return successful, failed


if __name__ == "__main__":

    extract_split(
        "train",
        os.path.join(
            "data",
            "features",
            "train_features.csv"
        )
    )

    extract_split(
        "validation",
        os.path.join(
            "data",
            "features",
            "validation_features.csv"
        )
    )

    extract_split(
        "test",
        os.path.join(
            "data",
            "features",
            "test_features.csv"
        )
    )