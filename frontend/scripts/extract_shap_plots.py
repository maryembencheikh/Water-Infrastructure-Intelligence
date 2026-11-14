
import json
import base64
from pathlib import Path


# Project paths
ROOT = Path(__file__).resolve().parents[2]

NOTEBOOK_PATH = (
    ROOT / "notebooks" / "03_machine_learning.ipynb"
)

OUTPUT_DIR = ROOT / "frontend" / "assets" / "shap"


# Notebook cell -> output filename
SHAP_PLOTS = {
    67: "shap_summary_all_classes.png",
    70: "shap_summary_functional.png",
    71: "shap_summary_non_functional.png",
    72: "shap_summary_needs_repair.png",
    74: "shap_waterfall_needs_repair.png"
}


def main():

    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"Notebook not found: {NOTEBOOK_PATH}"
        )

    with open(
        NOTEBOOK_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        notebook = json.load(file)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for cell_index, filename in SHAP_PLOTS.items():

        cell = notebook["cells"][cell_index]

        png_data = None

        for output in cell.get("outputs", []):

            data = output.get("data", {})

            if "image/png" in data:
                png_data = data["image/png"]
                break

        if png_data is None:
            print(
                f"[NOT FOUND] Cell {cell_index}: {filename}"
            )
            continue

        if isinstance(png_data, list):
            png_data = "".join(png_data)

        image_bytes = base64.b64decode(png_data)

        output_path = OUTPUT_DIR / filename

        output_path.write_bytes(image_bytes)

        print(
            f"[EXTRACTED] {filename} "
            f"({len(image_bytes):,} bytes)"
        )

    print("\nSHAP extraction completed.")


if __name__ == "__main__":
    main()
