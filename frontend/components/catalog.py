
import json
from pathlib import Path

import streamlit as st


# Catalog location
FRONTEND_DIR = Path(__file__).resolve().parents[1]

CATALOG_PATH = (
    FRONTEND_DIR / "data" / "category_options.json"
)


@st.cache_data(show_spinner=False)
def load_category_catalog():

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Category catalog not found: {CATALOG_PATH}"
        )

    with open(
        CATALOG_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        catalog = json.load(file)

    if "fields" not in catalog:
        raise ValueError(
            "Invalid category catalog: missing fields."
        )

    if "location_hierarchy" not in catalog:
        raise ValueError(
            "Invalid category catalog: missing location hierarchy."
        )

    return catalog


def get_category_options(catalog, field):

    return catalog["fields"].get(
        field, {}
    ).get("options", [])


def get_location_hierarchy(catalog):

    return catalog["location_hierarchy"]["regions"]
