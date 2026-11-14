
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pathlib import Path
# ==================================================
# MODEL RESULTS — REAL VALIDATION OUTPUT
# ==================================================

ACCURACY = 0.7753367003367003
BALANCED_ACCURACY = 0.7335727311522732
F1_MACRO = 0.6900354604851137
CV_F1_MACRO = 0.6834234810023198

CLASS_NAMES = [
    "Functional",
    "Functional Needs Repair",
    "Non Functional"
]

CONFUSION_MATRIX = [
    [5079, 773, 600],
    [222, 541, 100],
    [717, 257, 3591]
]

CLASS_METRICS = pd.DataFrame({
    "Class": CLASS_NAMES,
    "Precision": [0.84, 0.34, 0.84],
    "Recall": [0.79, 0.63, 0.79],
    "F1-score": [0.81, 0.44, 0.81],
    "Support": [6452, 863, 4565]
})


# ==================================================
# HEADER
# ==================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-eyebrow">
            MACHINE LEARNING & MODEL VALIDATION
        </div>
        <h1>Model Insights</h1>
        <p>
            Evaluate model performance, understand
            classification behavior and explore
            the strengths and limitations of LightGBM.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")


# ==================================================
# MODEL OVERVIEW
# ==================================================

st.markdown("### Model Overview")

overview = [
    ("Algorithm", "LightGBM", "Gradient Boosting"),
    ("Problem", "Multiclass", "Classification"),
    ("Validation", "80/20", "Hold-out evaluation"),
    ("Validation Records", "11,880", "Hold-out validation observations")
]
columns = st.columns(4)

for column, (label, value, description) in zip(
    columns, overview
):
    with column:
        value_class = (
            "metric-value metric-value-compact"
            if label in {"Algorithm", "Problem"}
            else "metric-value"
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="{value_class}">{value}</div>
                <div style="color:#718096;font-size:0.78rem;">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==================================================
# GLOBAL PERFORMANCE METRICS
# ==================================================

st.write("")
st.divider()

st.markdown("### Overall Performance")

st.caption(
    "Metrics calculated on the held-out validation dataset."
)

metric_columns = st.columns(4)

metrics = [
    ("Accuracy", ACCURACY, "Overall correct predictions"),
    ("Balanced Accuracy", BALANCED_ACCURACY, "Average class recall"),
    ("Macro F1-score", F1_MACRO, "Equal class weighting"),
    ("CV Macro F1", CV_F1_MACRO, "Cross-validation score")
]

for column, (label, value, description) in zip(
    metric_columns, metrics
):
    with column:
        st.metric(
            label,
            f"{value:.2%}"
        )
        st.caption(description)


# ==================================================
# CLASSIFICATION PERFORMANCE
# ==================================================

st.write("")
st.divider()

st.markdown("### Classification Performance by Class")

st.caption(
    "Precision, recall and F1-score for each target category."
)

display_metrics = CLASS_METRICS.copy()

for col in ["Precision", "Recall", "F1-score"]:
    display_metrics[col] = display_metrics[col].map(
        lambda value: f"{value:.0%}"
    )

st.dataframe(
    display_metrics,
    width="stretch",
    hide_index=True
)


# ==================================================
# CONFUSION MATRIX
# ==================================================

st.write("")
st.divider()

st.markdown("### Confusion Matrix")

st.caption(
    "Rows represent actual classes; columns represent "
    "predicted classes. Diagonal values are correct predictions."
)

fig = go.Figure(
    data=go.Heatmap(
        z=CONFUSION_MATRIX,
        x=CLASS_NAMES,
        y=CLASS_NAMES,
        colorscale=[
            [0, "#EDF3F6"],
            [0.5, "#72C9BE"],
            [1, "#0B756E"]
        ],
        showscale=True,
        colorbar=dict(title="Records"),
        hovertemplate=(
            "Actual: %{y}<br>"
            "Predicted: %{x}<br>"
            "Records: %{z}<extra></extra>"
        )
    )
)

for row_index, row in enumerate(CONFUSION_MATRIX):
    for col_index, value in enumerate(row):
        fig.add_annotation(
            x=CLASS_NAMES[col_index],
            y=CLASS_NAMES[row_index],
            text=f"<b>{value:,}</b>",
            showarrow=False,
            font=dict(
                color="white" if value > 1800 else "#19344A",
                size=14
            )
        )

fig.update_layout(
    template="plotly_white",
    height=460,
    margin=dict(l=20, r=20, t=25, b=80),
    xaxis_title="Predicted Class",
    yaxis_title="Actual Class",
    font=dict(
        family="Inter, sans-serif",
        color="#19344A"
    )
)

fig.update_yaxes(autorange="reversed")

st.plotly_chart(
    fig,
    width="stretch"
)


# ==================================================
# MODEL DIAGNOSTICS
# ==================================================

st.write("")
st.divider()

st.markdown("### Model Diagnostics")

st.markdown(
    """
    <div class="panel-card">
        <h3 style="font-size:1.1rem;color:#A66C12;">
            Class imbalance and repair detection
        </h3>
        <p style="color:#718096;line-height:1.7;">
            The <b>Functional Needs Repair</b> category
            has the weakest classification performance.
            Its recall is 63%, while its precision is only 34%.
        </p>
        <p style="color:#718096;line-height:1.7;">
            This indicates that the model identifies a
            considerable proportion of repair cases,
            but also assigns this category to many
            water points belonging to other classes.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

st.info(
    "Model performance should be interpreted across all "
    "three classes. Accuracy alone does not describe the "
    "model's ability to identify less frequent categories."
)


# ==================================================
# HYPERPARAMETERS
# ==================================================

st.write("")
st.divider()

st.markdown("### Best Model Configuration")

st.caption(
    f"Best cross-validation Macro F1-score: {CV_F1_MACRO:.4f}"
)

parameters = {
    "Parameter": [
        "colsample_bytree",
        "learning_rate",
        "max_depth",
        "min_child_samples",
        "n_estimators",
        "num_leaves"
    ],
    "Value": [
        "0.6571",
        "0.1337",
        "-1",
        "11",
        "543",
        "49"
    ]
}

st.dataframe(
    pd.DataFrame(parameters),
    width=700,
    hide_index=True
)



# ==================================================
# MODEL EXPLAINABILITY — SHAP
# ==================================================

st.write("")
st.divider()

st.markdown("### Model Explainability")

st.markdown(
    """
    SHAP (SHapley Additive exPlanations) helps us understand
    how individual features influence the predictions
    generated by our LightGBM model.

    The following visualizations were generated during
    the model validation and explainability analysis.
    """
)

SHAP_DIR = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "shap"
)

# ------------------------------------------
# Global SHAP Analysis
# ------------------------------------------

st.markdown("#### Global Feature Impact")

st.caption(
    "Overview of feature contributions across the three "
    "classification categories and 500 sampled observations."
)

global_plot = SHAP_DIR / "shap_summary_all_classes.png"

if global_plot.exists():
    st.image(
        str(global_plot),
        width=700
    )
else:
    st.warning("Global SHAP plot not found.")

st.markdown(
    """
    Each point represents an observation.
    Its horizontal position indicates the SHAP value,
    while its color represents the feature value.
    """
)

# ------------------------------------------
# Class-Specific SHAP Analysis
# ------------------------------------------

st.markdown("#### Class-Specific Feature Impact")

functional_tab, non_functional_tab, repair_tab = st.tabs([
    "Functional",
    "Non Functional",
    "Needs Repair"
])

class_plots = {
    "functional": (
        "shap_summary_functional.png"
    ),
    "non_functional": (
        "shap_summary_non_functional.png"
    ),
    "needs_repair": (
        "shap_summary_needs_repair.png"
    )
}

with functional_tab:

    plot_path = SHAP_DIR / class_plots["functional"]

    if plot_path.exists():
        st.image(
            str(plot_path),
            width=700
        )
    else:
        st.warning("Functional SHAP plot not found.")

with non_functional_tab:

    plot_path = SHAP_DIR / class_plots["non_functional"]

    if plot_path.exists():
        st.image(
            str(plot_path),
            width=700
        )
    else:
        st.warning("Non Functional SHAP plot not found.")

with repair_tab:

    plot_path = SHAP_DIR / class_plots["needs_repair"]

    if plot_path.exists():
        st.image(
            str(plot_path),
            width=700
        )
    else:
        st.warning("Needs Repair SHAP plot not found.")

# ------------------------------------------
# Local SHAP Explanation
# ------------------------------------------

st.markdown("#### Local Prediction Explanation")

st.caption(
    "An individual observation explained using a SHAP "
    "Waterfall Plot."
)

waterfall_plot = (
    SHAP_DIR / "shap_waterfall_needs_repair.png"
)

if waterfall_plot.exists():

    st.info(
        "Example: observation 2 was predicted as "
        "'Functional Needs Repair', while its actual "
        "class was 'Functional'."
    )

    st.image(
        str(waterfall_plot),
        width=700
    )

else:
    st.warning("SHAP Waterfall plot not found.")

# ------------------------------------------
# SHAP Interpretation
# ------------------------------------------

st.markdown("#### How to Interpret SHAP")

with st.expander("Understand SHAP values"):

    st.markdown(
        """
        **Positive SHAP values**

        Push the model output towards the class being explained.

        **Negative SHAP values**

        Push the model output away from the class being explained.

        **Feature values**

        In the summary plots, red generally represents higher
        feature values and blue represents lower feature values.

        **Waterfall Plot**

        Shows how individual feature contributions move
        the model output from its baseline value to the
        output for one specific observation.
        """
    )

    st.caption(
        "SHAP values explain the model's output, not "
        "necessarily causal relationships in the real world."
    )


# ==================================================
# FOOTER
# ==================================================

st.write("")
st.divider()

st.caption(
    "Water Infrastructure Intelligence · "
    "LightGBM Model Evaluation"
)
