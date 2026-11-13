
import math
import uuid
from datetime import datetime

import streamlit as st

from components.api_client import predict_water_point
from components.catalog import (
    load_category_catalog,
    get_category_options,
    get_location_hierarchy
)


# ==================================================
# CONFIGURATION
# ==================================================

NOT_SPECIFIED = "Not specified"

st.session_state.setdefault("prediction_result", None)
st.session_state.setdefault("assessment_payload", None)


# ==================================================
# LOAD DATASET CATEGORY CATALOG
# ==================================================

try:
    catalog = load_category_catalog()
    locations = get_location_hierarchy(catalog)

except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(f"Unable to load category catalog: {error}")
    st.stop()


# ==================================================
# REUSABLE HELPERS
# ==================================================

def safe_selectbox(label, options, key, help=None, disabled=False):
    """Searchable dropdown with a missing-value option."""

    options = [NOT_SPECIFIED] + list(options)

    current = st.session_state.get(key, NOT_SPECIFIED)

    # Reset a dependent selection when its parent changes.
    if current not in options:
        st.session_state[key] = NOT_SPECIFIED

    return st.selectbox(
        label,
        options=options,
        key=key,
        help=help,
        disabled=disabled
    )


def category_select(label, field, key, help=None):
    options = get_category_options(catalog, field)

    return safe_selectbox(
        label,
        options,
        key,
        help=help
    )


def optional_text(value):
    value = value.strip()
    return value if value else None


def optional_number(value, label, converter):
    value = value.strip()

    if not value:
        return None

    try:
        number = converter(value)
    except ValueError as error:
        raise ValueError(
            f"{label}: please enter a valid number."
        ) from error

    if isinstance(number, float) and not math.isfinite(number):
        raise ValueError(
            f"{label}: the value must be finite."
        )

    return number


def boolean_value(value):
    if value == NOT_SPECIFIED:
        return None

    return value == "Yes"


# ==================================================
# HEADER
# ==================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-eyebrow">
            INTELLIGENT INFRASTRUCTURE ASSESSMENT
        </div>
        <h1>Water Point Assessment</h1>
        <p>
            Describe a water point using the available
            information. Our machine learning model will
            estimate its operational status.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

st.markdown("### Assessment Workspace")

st.caption(
    "Select values from the available dataset categories. "
    "Leave information unspecified when it is unknown."
)

st.info(
    "You do not need to know the technical dataset column names. "
    "The form uses descriptive labels and guided selections."
)


# ==================================================
# 01 — LOCATION
# ==================================================

st.markdown("#### 01 — Geographic Information")

with st.container(border=True):

    st.markdown("**Where is the water point located?**")

    col1, col2 = st.columns(2)

    with col1:

        basin = category_select(
            "Water Basin",
            "basin",
            "assessment_basin"
        )

        region_options = get_category_options(
            catalog, "region"
        )

        region = safe_selectbox(
            "Region",
            region_options,
            "assessment_region",
            help="Start typing to search for a region."
        )

        region_data = (
            locations.get(region, {})
            if region != NOT_SPECIFIED
            else {}
        )

        lga = safe_selectbox(
            "District / LGA",
            region_data.get("lgas", []),
            "assessment_lga",
            disabled=region == NOT_SPECIFIED,
            help="Available districts depend on the selected region."
        )

        ward_options = region_data.get(
            "wards_by_lga", {}
        ).get(lga, [])

        ward = safe_selectbox(
            "Ward",
            ward_options,
            "assessment_ward",
            disabled=(
                region == NOT_SPECIFIED
                or lga == NOT_SPECIFIED
            ),
            help="Available wards depend on the selected district."
        )

    with col2:

        latitude = st.text_input(
            "Latitude",
            placeholder="Example: -6.8235",
            help="Geographic latitude in decimal degrees."
        )

        longitude = st.text_input(
            "Longitude",
            placeholder="Example: 35.7382",
            help="Geographic longitude in decimal degrees."
        )

        gps_height = st.text_input(
            "Elevation (meters)",
            placeholder="Optional"
        )


# ==================================================
# 02 — TECHNICAL CHARACTERISTICS
# ==================================================

st.write("")

st.markdown("#### 02 — Water Point Characteristics")

with st.container(border=True):

    st.markdown("**Describe the infrastructure.**")

    col1, col2 = st.columns(2)

    with col1:

        waterpoint_type = category_select(
            "Type of Water Point",
            "waterpoint_type",
            "assessment_waterpoint_type"
        )

        extraction_type = category_select(
            "Water Extraction Method",
            "extraction_type",
            "assessment_extraction"
        )

        source = category_select(
            "Water Source",
            "source",
            "assessment_source"
        )

        construction_year = st.text_input(
            "Construction Year",
            placeholder="Example: 2005"
        )

    with col2:

        quantity = category_select(
            "Water Availability",
            "quantity",
            "assessment_quantity"
        )

        water_quality = category_select(
            "Water Quality",
            "water_quality",
            "assessment_quality"
        )

        population = st.text_input(
            "Estimated Population Served",
            placeholder="Example: 250"
        )

        amount_tsh = st.text_input(
            "Available Water Amount",
            placeholder="Optional numeric value",
            help="The original amount_tsh feature from the dataset."
        )


# ==================================================
# 03 — MANAGEMENT
# ==================================================

st.write("")

st.markdown("#### 03 — Management & Operations")

with st.container(border=True):

    st.markdown(
        "**Provide information about the organization "
        "responsible for the water point.**"
    )

    col1, col2 = st.columns(2)

    with col1:

        management = category_select(
            "Management Organization",
            "management",
            "assessment_management"
        )

        scheme_management = category_select(
            "Scheme Management",
            "scheme_management",
            "assessment_scheme_management"
        )

        payment = category_select(
            "Payment Arrangement",
            "payment",
            "assessment_payment"
        )

        public_meeting = category_select(
            "Public Meeting",
            "public_meeting",
            "assessment_public_meeting"
        )

    with col2:

        permit = category_select(
            "Water Point Permit",
            "permit",
            "assessment_permit"
        )

        funder = category_select(
            "Funding Organization",
            "funder",
            "assessment_funder",
            help="Search the organizations found in the training dataset."
        )

        installer = category_select(
            "Installation Organization",
            "installer",
            "assessment_installer"
        )

        scheme_name = category_select(
            "Water Scheme Name",
            "scheme_name",
            "assessment_scheme_name"
        )


# ==================================================
# 04 — ADDITIONAL INFORMATION
# ==================================================

st.write("")

with st.expander("Additional technical information (optional)"):

    st.write(
        "Provide these details only when they are available."
    )

    col1, col2 = st.columns(2)

    with col1:

        num_private = st.text_input(
            "Number of Private Connections",
            placeholder="Optional integer"
        )

        region_code = st.text_input(
            "Region Code",
            placeholder="Optional integer"
        )

    with col2:

        district_code = st.text_input(
            "District Code",
            placeholder="Optional integer"
        )

        date_recorded = st.text_input(
            "Observation Date",
            placeholder="YYYY-MM-DD",
            help="Date on which the water point information was recorded."
        )


# ==================================================
# INFORMATION COMPLETENESS
# ==================================================

st.write("")
st.divider()

st.markdown("### Assessment Completeness")

provided_values = [
    basin, region, lga, ward,
    latitude, longitude, gps_height,
    waterpoint_type, extraction_type, source,
    quantity, water_quality, construction_year,
    population, amount_tsh,
    management, scheme_management, payment,
    public_meeting, permit, funder, installer,
    scheme_name, num_private, region_code,
    district_code, date_recorded
]

provided_count = sum(
    value is not None
    and str(value).strip() != ""
    and value != NOT_SPECIFIED
    for value in provided_values
)

total_fields = len(provided_values)
completion = provided_count / total_fields

col1, col2 = st.columns([3, 1])

with col1:
    st.progress(completion)

with col2:
    st.markdown(
        f"**{provided_count}/{total_fields} fields provided**"
    )

st.caption(
    "This indicator measures the amount of information entered. "
    "It is not a model confidence or prediction accuracy score."
)


# ==================================================
# PREDICTION
# ==================================================

st.write("")

submitted = st.button(
    "Generate Operational Status Prediction",
    type="primary",
    use_container_width=True
)


if submitted:

    st.session_state["prediction_result"] = None
    st.session_state["assessment_payload"] = None

    try:

        # Parse numeric values.
        latitude_value = optional_number(
            latitude, "Latitude", float
        )

        longitude_value = optional_number(
            longitude, "Longitude", float
        )

        gps_height_value = optional_number(
            gps_height, "Elevation", int
        )

        construction_year_value = optional_number(
            construction_year, "Construction Year", int
        )

        population_value = optional_number(
            population, "Population", int
        )

        amount_tsh_value = optional_number(
            amount_tsh, "Available Water Amount", float
        )

        num_private_value = optional_number(
            num_private, "Private Connections", int
        )

        region_code_value = optional_number(
            region_code, "Region Code", int
        )

        district_code_value = optional_number(
            district_code, "District Code", int
        )

        # Validate geographical coordinates.
        if latitude_value is not None:
            if not -90 <= latitude_value <= 90:
                raise ValueError(
                    "Latitude must be between -90 and 90."
                )

        if longitude_value is not None:
            if not -180 <= longitude_value <= 180:
                raise ValueError(
                    "Longitude must be between -180 and 180."
                )

        # Validate non-negative measurements.
        for label, value in [
            ("Population", population_value),
            ("Available Water Amount", amount_tsh_value),
            ("Private Connections", num_private_value)
        ]:
            if value is not None and value < 0:
                raise ValueError(
                    f"{label} cannot be negative."
                )

        # Validate construction year.
        if construction_year_value is not None:
            if not 1800 <= construction_year_value <= datetime.now().year:
                raise ValueError(
                    "Construction year is outside the accepted range."
                )

        # Validate observation date.
        if date_recorded.strip():
            datetime.strptime(
                date_recorded.strip(),
                "%Y-%m-%d"
            )

        # Generate an internal identifier.
        generated_id = uuid.uuid4().int % 9_000_000_000 + 1

        # Construct API payload.
        payload = {
            "id": generated_id,

            "basin": None if basin == NOT_SPECIFIED else basin,
            "region": None if region == NOT_SPECIFIED else region,
            "lga": None if lga == NOT_SPECIFIED else lga,
            "ward": None if ward == NOT_SPECIFIED else ward,

            "latitude": latitude_value,
            "longitude": longitude_value,
            "gps_height": gps_height_value,

            "waterpoint_type": (
                None if waterpoint_type == NOT_SPECIFIED
                else waterpoint_type
            ),

            "extraction_type": (
                None if extraction_type == NOT_SPECIFIED
                else extraction_type
            ),

            "source": None if source == NOT_SPECIFIED else source,
            "quantity": None if quantity == NOT_SPECIFIED else quantity,
            "water_quality": (
                None if water_quality == NOT_SPECIFIED
                else water_quality
            ),

            "construction_year": construction_year_value,
            "population": population_value,
            "amount_tsh": amount_tsh_value,

            "num_private": num_private_value,
            "region_code": region_code_value,
            "district_code": district_code_value,

            "management": (
                None if management == NOT_SPECIFIED else management
            ),

            "scheme_management": (
                None if scheme_management == NOT_SPECIFIED
                else scheme_management
            ),

            "payment": None if payment == NOT_SPECIFIED else payment,

            "public_meeting": boolean_value(public_meeting),
            "permit": boolean_value(permit),

            "funder": None if funder == NOT_SPECIFIED else funder,
            "installer": None if installer == NOT_SPECIFIED else installer,
            "scheme_name": None if scheme_name == NOT_SPECIFIED else scheme_name,

            "date_recorded": optional_text(date_recorded)
        }

        # Send prediction request.
        with st.spinner(
            "Connecting to the prediction engine..."
        ):

            result = predict_water_point(payload)

        # Store results for the Results page.
        st.session_state["prediction_result"] = result
        st.session_state["assessment_payload"] = payload

        st.success(
            "Prediction generated successfully."
        )

        st.metric(
            "Predicted Operational Status",
            result["prediction"].title()
        )

        st.page_link(
            "pages/3_Results.py",
            label="Explore Prediction Results",
            icon="📊"
        )

    except (ValueError, RuntimeError) as error:
        st.error(str(error))

    except Exception as error:
        st.error(
            f"Unexpected error while generating prediction: {error}"
        )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Water Infrastructure Intelligence · "
    "Machine learning predictions support decision-making "
    "but do not replace physical inspections."
)
