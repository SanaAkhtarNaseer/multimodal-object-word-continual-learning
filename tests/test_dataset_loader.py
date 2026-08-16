import os
import sys
import csv
import tempfile

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.dataset_loader import (
    load_cdsn_features,
    SHAPE_DIM,
    COLOR_DIM
)


def create_test_csv(path):
    fieldnames = (
        ["name"]
        + [f"shape_{i}" for i in range(SHAPE_DIM)]
        + [f"color_{i}" for i in range(COLOR_DIM)]
    )

    row = {"name": "apple"}

    for i in range(SHAPE_DIM):
        row[f"shape_{i}"] = float(i) / SHAPE_DIM

    for i in range(COLOR_DIM):
        row[f"color_{i}"] = float(i) / COLOR_DIM

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


with tempfile.TemporaryDirectory() as temp_dir:

    csv_path = os.path.join(temp_dir, "test_features.csv")

    create_test_csv(csv_path)

    samples = load_cdsn_features(csv_path)

    print("Number of samples:", len(samples))
    print("Name:", samples[0]["name"])
    print("Shape dimension:", samples[0]["shape"].shape)
    print("Color dimension:", samples[0]["color"].shape)