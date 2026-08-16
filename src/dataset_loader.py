import csv
import numpy as np


SHAPE_DIM = 23
COLOR_DIM = 63


def load_cdsn_features(csv_path):
    """
    Load CDSN feature data.

    Expected CSV columns:

        name
        shape_0 ... shape_22
        color_0 ... color_62

    Returns
    -------
    list of dictionaries:
        {
            "name": str,
            "shape": np.ndarray shape (23,),
            "color": np.ndarray shape (63,)
        }
    """

    samples = []

    with open(csv_path, "r", newline="", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        required_columns = (
            ["name"]
            + [f"shape_{i}" for i in range(SHAPE_DIM)]
            + [f"color_{i}" for i in range(COLOR_DIM)]
        )

        missing = [
            column
            for column in required_columns
            if column not in reader.fieldnames
        ]

        if missing:
            raise ValueError(
                "Missing required CSV columns: "
                + ", ".join(missing)
            )

        for row_number, row in enumerate(reader, start=2):

            try:
                shape = np.array(
                    [
                        float(row[f"shape_{i}"])
                        for i in range(SHAPE_DIM)
                    ],
                    dtype=float
                )

                color = np.array(
                    [
                        float(row[f"color_{i}"])
                        for i in range(COLOR_DIM)
                    ],
                    dtype=float
                )

            except ValueError as error:
                raise ValueError(
                    f"Invalid numeric feature at CSV row {row_number}"
                ) from error

            name = row["name"].strip()

            if not name:
                raise ValueError(
                    f"Missing object name at CSV row {row_number}"
                )

            samples.append(
                {
                    "name": name,
                    "shape": shape,
                    "color": color
                }
            )

    return samples
