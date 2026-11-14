
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ==================================================
# CONFIGURATION
# ==================================================

CLASS_COLORS = {
    "functional": "#0D9488",
    "functional needs repair": "#D99A30",
    "non functional": "#DC5C5C"
}

CLASS_LABELS = {
    "functional": "Functional",
    "functional needs repair": "Functional Needs Repair",
    "non functional": "Non Functional"
}


# ==================================================
# PAGE HEADER
# ==================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-eyebrow">MODEL OUTPUT & ANALYTICS</div>
        <h1>Prediction Results</h1>
        <p>
            Explore the predicted operational status,
            class probabilities and assessment details.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")


# ==================================================
# RETRIEVE SESSION RESULTS
# ==================================================

result = st.session_state.get("prediction_result")
payload = st.session_state.get("assessment_payload")


if not result:

    st.warning(
        "No prediction is available yet. "
        "Complete a water point assessment first."
    )

    st.page_link(
        "pages/2_Assessment.py",
        label="Start a new assessment",
        icon="💧"
    )

    st.stop()


# ==================================================
# VALIDATE RESULT
# ==================================================

prediction = result.get("prediction")
probabilities = result.get("probabilities", {})
waterpoint_id = result.get("id")
api_status = result.get("status", "success")

if prediction not in CLASS_LABELS:
    st.error("The API returned an unknown prediction class.")
    st.stop()

if not isinstance(probabilities, dict):
    st.error("Invalid probability data returned by the API.")
    st.stop()

try:
    probabilities = {
        label: float(value)
        for label, value in probabilities.items()
    }
except (TypeError, ValueError):
    st.error("Unable to interpret the prediction probabilities.")
    st.stop()

if (
    set(probabilities) != set(CLASS_LABELS)
    or any(
        not 0 <= value <= 1
        for value in probabilities.values()
    )
    or abs(sum(probabilities.values()) - 1.0) > 0.001
):
    st.error("The API returned invalid class probabilities.")
    st.stop()

predicted_probability = probabilities[prediction]



# ==================================================
# MAIN PREDICTION CARD
# ==================================================

st.markdown("### Assessment Outcome")

status_label = CLASS_LABELS[prediction]

with st.container(border=True):

    st.caption("PREDICTED OPERATIONAL STATUS")

    st.subheader(status_label)

    st.caption(f"Water Point ID: {waterpoint_id}")

# ==================================================
# KEY PERFORMANCE INDICATORS
# ==================================================

st.markdown("### Prediction Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Predicted Class",
        status_label
    )

with col2:
    st.metric(
        "Highest Class Probability",
        f"{predicted_probability:.2%}"
    )

with col3:
    st.metric(
        "Assessment ID",
        str(waterpoint_id)
    )


# ==================================================
# PROBABILITY VISUALIZATION
# ==================================================

st.divider()

st.markdown("### Class Probability Analysis")

st.caption(
    "The following values represent the probabilities "
    "estimated by the LightGBM classification model."
)

chart_col1, chart_col2 = st.columns([1.5, 1])

labels = list(CLASS_LABELS.keys())

values = [
    probabilities[label]
    for label in labels
]

display_labels = [
    CLASS_LABELS[label]
    for label in labels
]

colors = [
    CLASS_COLORS[label]
    for label in labels
]


# Horizontal probability chart

with chart_col1:

    st.markdown("#### Probability Comparison")

    bar_chart = go.Figure(
        go.Bar(
            x=[value * 100 for value in values],
            y=display_labels,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=0)
            ),
            text=[f"{value:.2%}" for value in values],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Probability: %{x:.2f}%"
                "<extra></extra>"
            )
        )
    )

    bar_chart.update_layout(
        template="plotly_white",
        height=330,
        margin=dict(l=10, r=45, t=20, b=35),
        xaxis=dict(
            title="Probability (%)",
            range=[0, 110],
            gridcolor="#E8EDF2",
            ticksuffix="%"
        ),
        yaxis=dict(
            title="",
            autorange="reversed"
        ),
        showlegend=False,
        font=dict(
            family="Inter, sans-serif",
            color="#19344A"
        )
    )

    st.plotly_chart(
        bar_chart,
        width="stretch"
    )


# Donut chart

with chart_col2:

    st.markdown("#### Probability Distribution")

    donut_chart = go.Figure(
        go.Pie(
            labels=display_labels,
            values=values,
            hole=0.68,
            marker=dict(
                colors=colors,
                line=dict(
                    color="white",
                    width=3
                )
            ),
            textinfo="percent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "%{percent}"
                "<extra></extra>"
            )
        )
    )

    donut_chart.update_layout(
        template="plotly_white",
        height=330,
        margin=dict(l=5, r=5, t=20, b=20),
        showlegend=False,
        annotations=[
            dict(
                text=f"{predicted_probability:.1%}",
                x=0.5,
                y=0.5,
                font=dict(
                    size=25,
                    color="#19344A"
                ),
                showarrow=False
            )
        ]
    )

    st.plotly_chart(
        donut_chart,
        width="stretch"
    )


# ==================================================
# INPUT DATA SUMMARY
# ==================================================

st.divider()

st.markdown("### Assessment Information")

if payload:

    summary_fields = {
        "basin": "Basin",
        "region": "Region",
        "lga": "District / LGA",
        "ward": "Ward",
        "waterpoint_type": "Water Point Type",
        "extraction_type": "Extraction Type",
        "source": "Water Source",
        "quantity": "Water Quantity",
        "water_quality": "Water Quality",
        "management": "Management",
        "payment": "Payment Type",
        "population": "Population Served",
        "construction_year": "Construction Year"
    }

    available_data = []

    for key, label in summary_fields.items():

        value = payload.get(key)

        if value is not None:
            available_data.append(
                {
                    "Characteristic": label,
                    "Provided Value": str(value)
                }
            )

    if available_data:

        summary_df = pd.DataFrame(available_data)

        st.dataframe(
            summary_df,
            width="stretch",
            hide_index=True
        )

    else:
        st.info("No optional assessment information was provided.")

    provided_fields = sum(
        1 for key, value in payload.items()
        if key != "id" and value is not None
    )

    total_optional_fields = sum(
        1 for key in payload
        if key != "id"
    )

    st.caption(
        f"{provided_fields} of {total_optional_fields} "
        "available input fields were provided. "
        "This is not a prediction reliability score."
    )


# ==================================================
# DOWNLOAD REPORT
# ==================================================

st.divider()

st.markdown("### Export Assessment")

report = {
    "assessment_id": waterpoint_id,
    "api_status": api_status,
    "input_data": payload,
    "prediction": prediction,
    "probabilities": probabilities
}

st.download_button(
    label="Download Prediction Report (JSON)",
    data=json.dumps(
        report,
        indent=4,
        ensure_ascii=False
    ),
    file_name=f"water_point_{waterpoint_id}_report.json",
    mime="application/json",
    width="stretch"
)


# ==================================================
# DISCLAIMER
# ==================================================

st.info(
    "Machine learning predictions are estimates based on "
    "the characteristics provided. They do not replace "
    "physical inspection or professional water quality testing."
)
