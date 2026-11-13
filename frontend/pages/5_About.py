
import streamlit as st

from textwrap import dedent

# ==================================================
# HERO SECTION
# ==================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-eyebrow">PROJECT DOCUMENTATION</div>
        <h1>About Water Infrastructure Intelligence</h1>
        <p>
            An end-to-end machine learning solution designed
            to assess water infrastructure functionality
            through data-driven predictions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")


# ==================================================
# PROJECT OVERVIEW
# ==================================================

st.markdown("### Project Overview")

st.markdown(
    """
    Water infrastructure is essential for communities,
    public health and sustainable development.

    However, maintaining water points and identifying
    infrastructure requiring intervention can be challenging,
    particularly when operational information is incomplete
    or distributed across multiple sources.

    **Water Infrastructure Intelligence** uses historical
    water point data and machine learning to estimate
    whether a water point is functional, requires repair,
    or is non-functional.

    The platform combines data preprocessing, predictive
    modeling, explainability and API-based deployment
    into an integrated application.
    """
)


# ==================================================
# BUSINESS PROBLEM
# ==================================================

st.divider()

st.markdown("### The Business Problem")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="panel-card" style="min-height:210px;">
            <h3 style="font-size:1.1rem;">
                The Challenge
            </h3>
            <p style="color:#718096;line-height:1.7;">
                Water point functionality is influenced by
                several factors, including geographical
                conditions, technical infrastructure,
                water characteristics and management practices.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="panel-card" style="min-height:210px;">
            <h3 style="font-size:1.1rem;">
                The Approach
            </h3>
            <p style="color:#718096;line-height:1.7;">
                A trained LightGBM classification pipeline
                analyzes available water point characteristics
                and estimates the probability of each
                operational status.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# DATASET
# ==================================================

st.divider()

st.markdown("### Dataset Overview")

st.markdown(
    """
    The project uses the **Pump it Up: Data Mining the Water Table**
    dataset from DrivenData.

    It contains historical information about water points
    in Tanzania, including geographical, technical,
    management and water-related characteristics.
    """
)

col1, col2, col3 = st.columns(3)

dataset_metrics = [
    ("59,400", "Training Records"),
    ("14,850", "Competition Test Records"),
    ("40", "Original Input Features")
]

for column, (value, label) in zip(
    [col1, col2, col3],
    dataset_metrics
):
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.write("")

st.markdown("#### Target Classes")

target_classes = [
    (
        "Functional",
        "The water point is operational.",
        "#0D9488"
    ),
    (
        "Functional Needs Repair",
        "The water point is operational but requires repair.",
        "#D99A30"
    ),
    (
        "Non Functional",
        "The water point is not operational.",
        "#DC5C5C"
    )
]

for name, description, color in target_classes:

    st.markdown(
        f"""
        <div style="
            display:flex;
            align-items:center;
            gap:14px;
            background:white;
            border:1px solid #E5EBF1;
            border-radius:10px;
            padding:13px 16px;
            margin-bottom:10px;
        ">
            <div style="
                width:5px;
                height:42px;
                background:{color};
                border-radius:5px;
            "></div>
            <div>
                <div style="
                    font-weight:700;
                    color:#172B40;
                ">{name}</div>
                <div style="
                    color:#718096;
                    font-size:0.85rem;
                    margin-top:3px;
                ">{description}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# METHODOLOGY
# ==================================================

st.divider()

st.markdown("### Data Science Methodology")

methodology = [
    (
        "01",
        "Data Understanding & EDA",
        "Exploration of the dataset, target distribution,"
        " missing values and data quality."
    ),
    (
        "02",
        "Data Preprocessing",
        "Missing value treatment, feature engineering,"
        " categorical encoding and preprocessing pipelines."
    ),
    (
        "03",
        "Machine Learning",
        "Model development, comparison, hyperparameter"
        " optimization and LightGBM selection."
    ),
    (
        "04",
        "Validation & Explainability",
        "Model evaluation using classification metrics,"
        " confusion matrix and SHAP analysis."
    ),
    (
        "05",
        "Deployment & Operationalization",
        "FastAPI backend, automated API tests, Docker"
        " containerization and Render deployment."
    ),
    (
        "06",
        "User Interface & Delivery",
        "Interactive Streamlit frontend, prediction"
        " visualization and project documentation."
    )
]



for number, title, description in methodology:

    with st.container(border=True):

        number_col, content_col = st.columns([0.10, 0.90])

        with number_col:
            st.markdown(f"### {number}")

        with content_col:
            st.markdown(f"**{title}**")
            st.caption(description)

# ==================================================
# TECHNICAL ARCHITECTURE
# ==================================================

st.divider()

st.markdown("### Technical Architecture")

architecture = [
    ("User Interface", "Streamlit"),
    ("Backend API", "FastAPI"),
    ("Prediction Engine", "LightGBM + Scikit-learn"),
    ("API Communication", "HTTP / REST"),
    ("Containerization", "Docker"),
    ("Backend Hosting", "Render"),
    ("Version Control", "Git + GitHub")
]

st.dataframe(
    {
        "Component": [item[0] for item in architecture],
        "Technology": [item[1] for item in architecture]
    },
    use_container_width=True,
    hide_index=True
)


# ==================================================
# PROJECT RESOURCES
# ==================================================

st.divider()

st.markdown("### Project Resources")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="panel-card">
            <h3 style="font-size:1.1rem;">Source Code</h3>
            <p style="color:#718096;">
                Explore the project's implementation,
                notebooks and documentation.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.link_button(
        "View GitHub Repository",
        "https://github.com/maryembencheikh/Water-Infrastructure-Intelligence",
        use_container_width=True
    )

with col2:
    st.markdown(
        """
        <div class="panel-card">
            <h3 style="font-size:1.1rem;">Backend API</h3>
            <p style="color:#718096;">
                Explore and test the prediction API
                through its interactive documentation.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.link_button(
        "Explore API Documentation",
        "https://water-infrastructure-intelligence.onrender.com/docs",
        use_container_width=True
    )


# ==================================================
# DISCLAIMER
# ==================================================

st.write("")
st.divider()

st.warning(
    "Machine learning predictions are estimates based on "
    "historical data. They should support, not replace, "
    "physical inspections and professional assessments."
)


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#718096;
        font-size:0.8rem;
        padding:20px 0;
    ">
        WATER INFRASTRUCTURE INTELLIGENCE<br>
        End-to-End Data Science & Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
