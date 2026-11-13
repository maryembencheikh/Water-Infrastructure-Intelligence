
import streamlit as st
import pandas as pd


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-eyebrow">
            MACHINE LEARNING · WATER INFRASTRUCTURE
        </div>
        <h1>Water Infrastructure Intelligence</h1>
        <p>
            Transforming water point data into actionable
            infrastructure intelligence through machine learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

st.markdown(
    "### Platform Overview"
)

st.markdown(
    """
    This platform uses machine learning to assess the
    operational status of water points based on their
    technical, geographical and management characteristics.
    """
)


# ==========================================
# KEY PROJECT INDICATORS
# ==========================================

st.write("")

col1, col2, col3, col4 = st.columns(4)

metrics = [
    ("TRAINING RECORDS", "59,400", "Labeled observations"),
    ("TEST RECORDS", "14,850", "Unlabeled observations"),
    ("TARGET CLASSES", "03", "Classification categories"),
    ("ML ALGORITHM", "LightGBM", "Gradient boosting")
]

for column, (label, value, description) in zip(
    [col1, col2, col3, col4],
    metrics
):
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div style="
                    color:#718096;
                    font-size:0.78rem;
                    margin-top:6px;
                ">{description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# DATASET OVERVIEW
# ==========================================

st.write("")
st.divider()

st.markdown("### Water Point Status Distribution")

st.caption(
    "Distribution of the target variable in the "
    "labeled training dataset."
)

distribution = pd.DataFrame({
    "Water Point Status": [
        "Functional",
        "Non Functional",
        "Functional Needs Repair"
    ],
    "Share (%)": [
        54.31,
        38.42,
        7.27
    ]
})

left, right = st.columns([1.4, 1])

with left:

    st.markdown(
        '<div class="panel-card">',
        unsafe_allow_html=True
    )

    st.bar_chart(
        distribution.set_index("Water Point Status"),
        horizontal=True,
        x_label="Share (%)",
        height=280
    )

    st.markdown("</div>", unsafe_allow_html=True)

with right:

    st.markdown("#### Target Categories")

    for _, row in distribution.iterrows():

        st.markdown(
            f"**{row['Water Point Status']}**"
        )

        st.progress(
            float(row["Share (%)"] / 100)
        )

        st.caption(f"{row['Share (%)']:.2f}%")



# ==========================================
# PLATFORM CAPABILITIES
# ==========================================

st.write("")
st.divider()

st.markdown("### What Does This Platform Do?")

capabilities = [
    (
        "01",
        "Data Assessment",
        "Collect water point characteristics, including "
        "location, technical features and management information."
    ),
    (
        "02",
        "ML Prediction",
        "Use a trained LightGBM classification pipeline "
        "to estimate the operational status of a water point."
    ),
    (
        "03",
        "Prediction Insights",
        "Explore the predicted class and probabilities "
        "associated with the three possible outcomes."
    )
]

col1, col2, col3 = st.columns(3)

for column, (number, title, description) in zip(
    [col1, col2, col3],
    capabilities
):
    with column:
        with st.container(border=True):

            st.caption(number)

            st.markdown(f"#### {title}")

            st.write(description)



# ==========================================
# TECHNOLOGY STACK
# ==========================================

st.write("")
st.divider()

st.markdown("### Technology Stack")

technologies = [
    ("Python", "Programming"),
    ("Pandas", "Data Processing"),
    ("Scikit-learn", "ML Pipeline"),
    ("LightGBM", "Classification"),
    ("FastAPI", "Backend API"),
    ("Docker", "Containerization"),
    ("Render", "Cloud Hosting"),
    ("Streamlit", "Frontend")
]

columns = st.columns(4)

for index, (technology, role) in enumerate(technologies):

    with columns[index % 4]:

        st.markdown(
            f"""
            <div class="metric-card"
                 style="margin-bottom:12px;">
                <div style="
                    color:#172B40;
                    font-weight:700;
                ">{technology}</div>
                <div style="
                    color:#718096;
                    font-size:0.78rem;
                    margin-top:5px;
                ">{role}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# FOOTER
# ==========================================

st.write("")
st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#718096;
        font-size:0.8rem;
        padding:12px;
    ">
        WATER INFRASTRUCTURE INTELLIGENCE<br>
        Machine Learning · Data Science · Applied AI
    </div>
    """,
    unsafe_allow_html=True
)
