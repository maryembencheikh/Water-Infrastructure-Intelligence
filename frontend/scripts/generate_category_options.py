
import json
from pathlib import Path

import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/raw/Training set values.csv"
OUTPUT_DIR = ROOT / "frontend/data"
OUTPUT_PATH = OUTPUT_DIR / "category_options.json"


# Categorical features retained by the preprocessing pipeline
CATEGORICAL_COLUMNS = [
    "basin",
    "region",
    "lga",
    "ward",
    "funder",
    "installer",
    "scheme_management",
    "scheme_name",
    "public_meeting",
    "permit",
    "extraction_type",
    "management",
    "payment",
    "water_quality",
    "quantity",
    "source",
    "waterpoint_type"
]

BOOLEAN_COLUMNS = [
    "public_meeting",
    "permit"
]


def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    categories = {
        "source": "Training set values.csv",
        "fields": {}
    }

    for column in CATEGORICAL_COLUMNS:

        if column not in df.columns:
            raise ValueError(
                f"Column not found in dataset: {column}"
            )

        values = df[column].dropna()

        if column in BOOLEAN_COLUMNS:

            mapping = {
                "true": "Yes",
                "false": "No",
                "1": "Yes",
                "0": "No",
                "yes": "Yes",
                "no": "No"
            }

            normalized = (
                values.astype(str)
                .str.strip()
                .str.lower()
                .map(mapping)
                .dropna()
            )

            counts = normalized.value_counts()

        else:

            normalized = (
                values.astype(str)
                .str.strip()
            )

            normalized = normalized[
                normalized.ne("")
            ]

            counts = normalized.value_counts()

        options = sorted(
            counts.index.tolist(),
            key=str.casefold
        )

        categories["fields"][column] = {
            "options": options,
            "unique_count": len(options),
            "counts": {
                str(value): int(count)
                for value, count in counts.items()
            }
        }

        print(
            f"{column}: {len(options)} available options"
        )
# ==========================================
# LOCATION HIERARCHY
# Region -> LGA -> Ward
# ==========================================

    location_data = df[["region", "lga", "ward"]].copy()

    for column in ["region", "lga", "ward"]:
        location_data[column] = (
            location_data[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    regions = {}

    region_names = sorted(
        location_data["region"].dropna().unique().tolist(),
        key=str.casefold
    )

    for region_name in region_names:

        region_df = location_data[
            location_data["region"] == region_name
        ]

        lga_names = sorted(
            region_df["lga"].dropna().unique().tolist(),
            key=str.casefold
        )

        wards_by_lga = {}

        for lga_name in lga_names:

            lga_df = region_df[
                region_df["lga"] == lga_name
            ]

            wards = sorted(
                lga_df["ward"].dropna().unique().tolist(),
                key=str.casefold
            )

            wards_by_lga[lga_name] = wards

        regions[region_name] = {
            "lgas": lga_names,
            "wards_by_lga": wards_by_lga
        }
        categories["location_hierarchy"] = {
            "regions": regions
    }

    print(
        f"\nLocation hierarchy generated for "
        f"{len(regions)} regions."
    )
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            categories,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nCategory catalog generated successfully.")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
