import os
import streamlit as st
import pandas as pd
import plotly.express as px

from ml_engine import run_unsupervised_pipeline, FEATURE_COLS

DEFAULT_CSV = "intellirehab_unsupervised_features.csv"

st.set_page_config(page_title="SIH26113 | Movement Pattern Discovery", layout="wide")

st.title("Movement Pattern Discovery for Rehabilitation & Ergonomics")
st.caption("100% Unsupervised Learning Pipeline (INT396) | Real IntelliRehabDS Features")

# Sidebar Hyperparameters & Data Input
st.sidebar.header("Unsupervised Hyperparameters")
n_clusters = st.sidebar.slider("Number of Clusters (K / Ward)", 2, 6, 3)
eps = st.sidebar.slider("DBSCAN Epsilon (eps)", 0.3, 3.0, 1.1, 0.1)
min_samples = st.sidebar.slider("DBSCAN Min Samples", 2, 15, 5)
contamination = st.sidebar.slider("Isolation Forest Contamination", 0.02, 0.20, 0.07, 0.01)
color_by = st.sidebar.selectbox(
    "Color Projections By",
    ["Cluster_KMeans", "Cluster_Hierarchical", "Cluster_DBSCAN", "Anomaly_Flag"],
)

uploaded_file = st.sidebar.file_uploader("Upload Unlabeled Features CSV (Optional)", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
elif os.path.exists(DEFAULT_CSV):
    raw_df = pd.read_csv(DEFAULT_CSV)
else:
    st.error(
        f"'{DEFAULT_CSV}' nahi mila! Pehle `python feature_extractor.py` run karo "
        "ya sidebar se features CSV upload karo."
    )
    st.stop()

try:
    results_df, metrics_df, elbow_df, pca_var = run_unsupervised_pipeline(
        raw_df,
        n_clusters=n_clusters,
        eps=eps,
        min_samples=min_samples,
        contamination=contamination,
    )
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()

# Section 1: Internal Validation & Elbow Curve
st.subheader("1. Internal Cluster Validation & Elbow Method (WCSS)")
col1, col2 = st.columns([1.4, 1])

with col1:
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.info(
        f"**PCA Explained Variance:** PC1 = {pca_var[0]*100:.1f}%, PC2 = {pca_var[1]*100:.1f}% "
        f"(Total = {sum(pca_var)*100:.1f}%)"
    )

with col2:
    fig_elbow = px.line(
        elbow_df,
        x="K",
        y="WCSS_Inertia",
        markers=True,
        title="Elbow Curve (WCSS vs K)",
    )
    fig_elbow.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_elbow, use_container_width=True)

# Section 2: Dimensionality Reduction Projections
st.subheader("2. Unsupervised Projections: UMAP vs t-SNE vs PCA")
tab1, tab2, tab3 = st.tabs(["UMAP Projection", "t-SNE Projection", "PCA Projection"])
hover_cols = [
    "Rep_ID",
    "Range_of_Motion_deg",
    "Mean_Angular_Vel",
    "Max_Angular_Vel",
    "Movement_Jerk",
    "Joint_Symmetry_Index",
    "Rep_Variability",
]

with tab1:
    st.plotly_chart(
        px.scatter(results_df, x="UMAP_1", y="UMAP_2", color=color_by, hover_data=hover_cols),
        use_container_width=True,
    )

with tab2:
    st.plotly_chart(
        px.scatter(results_df, x="TSNE_1", y="TSNE_2", color=color_by, hover_data=hover_cols),
        use_container_width=True,
    )

with tab3:
    st.plotly_chart(
        px.scatter(results_df, x="PCA_1", y="PCA_2", color=color_by, hover_data=hover_cols),
        use_container_width=True,
    )

# Section 3: Cluster Profiles & Flagged Anomalies
st.subheader("3. Discovered Cluster Profiles & Flagged Unusual Repetitions")
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Cluster Feature Mean Summary**")
    profile_df = results_df.groupby(color_by)[FEATURE_COLS].mean().round(4)
    st.dataframe(profile_df, use_container_width=True)

with c2:
    st.markdown("**Flagged Unusual Repetitions (Isolation Forest)**")
    flagged_df = results_df[results_df["Anomaly_Flag"] == "Unusual / Flagged"][
        ["Rep_ID", "Range_of_Motion_deg", "Mean_Angular_Vel", "Movement_Jerk", "Joint_Symmetry_Index"]
    ]
    st.dataframe(flagged_df, use_container_width=True, hide_index=True)

st.warning(
    "Limitation & Clinical Note: Flagged repetitions and clusters reflect statistical movement patterns "
    "and deviations—they do not represent an automated medical or ergonomic diagnosis."
)

csv_bytes = results_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Unsupervised Results (CSV)",
    data=csv_bytes,
    file_name="sih26113_unsupervised_results.csv",
    mime="text/csv",
)