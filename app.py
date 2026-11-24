import os
import tempfile
import pandas as pd
import plotly.express as px
import streamlit as st

from ml_pipeline_v2 import run_full_ml_pipeline
from phase2_features import build_phase2_features

st.set_page_config(
    page_title="IntelliRehabDS — Unsupervised Kinematics Studio",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2.2rem;}
        div[data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(150, 165, 185, 0.18);
            padding: 0.75rem 1rem;
            border-radius: 10px;
        }
        .insight-card {
            background-color: rgba(30, 136, 229, 0.08);
            border-left: 4px solid #38bdf8;
            padding: 0.85rem 1.1rem;
            border-radius: 6px;
            margin-bottom: 0.55rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Running Phase 1–12 Unsupervised ML Pipeline...")
def _cached_pipeline(
    phase2_csv: str,
    scaler_type: str,
    prune_collinear: bool,
    contamination: float,
    lof_neighbors: int,
    random_state: int,
):
    return run_full_ml_pipeline(
        phase2_csv=phase2_csv,
        scaler_type=scaler_type,
        prune_collinear=prune_collinear,
        collinear_threshold=0.95,
        contamination=contamination,
        lof_neighbors=lof_neighbors,
        random_state=random_state,
    )


def main():
    st.title("IntelliRehabDS — Unsupervised Kinematics & Anomaly Studio")
    st.caption(
        "Unlabeled 3D Skeleton Motion Discovery • Multi-Model Clustering • Explainable Anomaly Ranking (INT396)"
    )

    # --- Sidebar ---
    with st.sidebar:
        st.header("Pipeline Controls")
        scaler_type = st.selectbox("Feature Scaler", options=["robust", "standard"], index=0)
        prune_collinear = st.checkbox("Prune Redundant Features (|r| ≥ 0.95)", value=True)
        contamination = st.slider("Anomaly Contamination", min_value=0.01, max_value=0.20, value=0.07, step=0.01)
        lof_neighbors = st.slider("LOF Neighbors", min_value=5, max_value=50, value=20, step=5)
        random_state = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1)

        st.divider()
        st.subheader("Projection & Dataset")
        color_by = st.selectbox(
            "Color Projections By",
            options=[
                "Cluster_KMeans",
                "Cluster_GMM",
                "Cluster_DBSCAN",
                "Cluster_Hierarchical",
                "Anomaly_Status",
            ],
            index=0,
        )
        uploaded_csv = st.file_uploader("Upload Kinematic CSV (Optional)", type=["csv"])

    csv_to_use = "phase2_kinematic_features.csv"
    if uploaded_csv is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded_csv.getvalue())
        tmp.close()
        csv_to_use = tmp.name

    if not os.path.exists(csv_to_use):
        if os.path.exists("phase2_kinematic_features.csv"):
            csv_to_use = "phase2_kinematic_features.csv"
        else:
            with st.spinner("Extracting 3D kinematic features from skeleton data..."):
                build_phase2_features(output_csv="phase2_kinematic_features.csv")
            csv_to_use = "phase2_kinematic_features.csv"

    bundle = _cached_pipeline(
        phase2_csv=csv_to_use,
        scaler_type=scaler_type,
        prune_collinear=prune_collinear,
        contamination=float(contamination),
        lof_neighbors=int(lof_neighbors),
        random_state=int(random_state),
    )

    results_df = bundle["results_df"].copy()
    scorecard_df = bundle["scorecard"].copy()
    scree_df = bundle["scree"].copy()
    fidelity_df = bundle["fidelity"].copy()
    prof_summary = bundle["profile_summary"].copy()
    prof_med = bundle["profile_median"].copy()
    anom_top = bundle["anomalies_top"].copy()
    anom_summ = bundle["anomaly_summary"]
    insights = bundle["insights"]
    tuned_params = bundle["tuned_params"]

    # --- Top KPI Row ---
    pca_2d_cum = (
        f"{float(scree_df['Cumulative_Variance'].iloc[1]) * 100:.1f}%"
        if len(scree_df) >= 2
        else "N/A"
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Repetitions Analyzed", f"{len(results_df):,}")
    k2.metric("Active 3D Features", f"{len(bundle['prep']['active_features'])}")
    k3.metric("Consensus Anomalies (IF+LOF)", f"{anom_summ.get('consensus_flagged', 0)}")
    k4.metric("PCA 2D Cumulative Variance", pca_2d_cum)

    st.write("")

    # --- Main 4-Tab Layout ---
    tab_ov, tab_eval, tab_proj, tab_anom = st.tabs(
        [
            "Overview & Insights",
            "Model Scorecard & PCA",
            "2D Projections",
            "Profiles & Anomalies",
        ]
    )

    # TAB 1: Overview & Insights
    with tab_ov:
        st.subheader("Automated Data-Driven Insights (Phase 11)")
        for line in insights:
            st.markdown(f"<div class='insight-card'>{line}</div>", unsafe_allow_html=True)

        st.subheader("Tuned Hyperparameters Summary")
        p1, p2, p3, p4 = st.columns(4)
        p1.info(f"**K-Means++**\n\nOptimal k = `{tuned_params['kmeans']['n_clusters']}`\n\nSil = `{tuned_params['kmeans']['silhouette']}`")
        p2.info(f"**Hierarchical**\n\nLinkage = `{tuned_params['hierarchical']['linkage']}`\n\nSil = `{tuned_params['hierarchical']['silhouette']}`")
        p3.info(f"**DBSCAN**\n\neps = `{tuned_params['dbscan']['eps']}` | min = `{tuned_params['dbscan']['min_samples']}`\n\nNoise = `{tuned_params['dbscan']['noise_pct']}%`")
        p4.info(f"**GMM**\n\nComponents = `{tuned_params['gmm']['n_components']}`\n\nCov = `{tuned_params['gmm']['covariance_type']}`")

    # TAB 2: Model Scorecard & PCA
    with tab_eval:
        c_left, c_right = st.columns([1.35, 1.0])
        with c_left:
            st.subheader("Comparative Clustering Scorecard")
            st.dataframe(scorecard_df, use_container_width=True, hide_index=True)

            st.subheader("Original vs. 2D Embedding Fidelity")
            st.dataframe(fidelity_df, use_container_width=True, hide_index=True)

        with c_right:
            st.subheader("PCA Scree & Cumulative Variance")
            fig_scree = px.bar(
                scree_df,
                x="PC",
                y="Explained_Variance_Ratio",
                text="Cumulative_Variance",
                labels={"Explained_Variance_Ratio": "Explained Variance Ratio"},
            )
            fig_scree.update_traces(textposition="outside")
            fig_scree.update_layout(height=360, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig_scree, use_container_width=True)

    # TAB 3: 2D Projections
    with tab_proj:
        plot_df = results_df.copy()
        plot_df[color_by] = plot_df[color_by].astype(str)
        hover_cols = [c for c in ["Rep_ID", "Anomaly_Severity", "Top_Driving_Features"] if c in plot_df.columns]

        sub_u, sub_t, sub_p = st.tabs(["UMAP", "t-SNE", "PCA"])

        with sub_u:
            fig_u = px.scatter(
                plot_df,
                x="UMAP_1",
                y="UMAP_2",
                color=color_by,
                hover_data=hover_cols,
                title=f"UMAP 2D Projection • Colored by {color_by}",
            )
            fig_u.update_traces(marker=dict(size=5.5, opacity=0.82))
            fig_u.update_layout(height=450)
            st.plotly_chart(fig_u, use_container_width=True)

        with sub_t:
            fig_t = px.scatter(
                plot_df,
                x="TSNE_1",
                y="TSNE_2",
                color=color_by,
                hover_data=hover_cols,
                title=f"t-SNE 2D Projection • Colored by {color_by}",
            )
            fig_t.update_traces(marker=dict(size=5.5, opacity=0.82))
            fig_t.update_layout(height=450)
            st.plotly_chart(fig_t, use_container_width=True)

        with sub_p:
            fig_p = px.scatter(
                plot_df,
                x="PCA_1",
                y="PCA_2",
                color=color_by,
                hover_data=hover_cols,
                title=f"PCA 2D Projection • Colored by {color_by}",
            )
            fig_p.update_traces(marker=dict(size=5.5, opacity=0.82))
            fig_p.update_layout(height=450)
            st.plotly_chart(fig_p, use_container_width=True)

    # TAB 4: Profiles & Anomalies
    with tab_anom:
        left_p, right_a = st.columns([1.0, 1.35])

        with left_p:
            st.subheader("Cluster Characteristics (Data-Driven)")
            if not prof_summary.empty:
                st.dataframe(prof_summary, use_container_width=True, hide_index=True)

            st.subheader("Cluster Feature Medians")
            if not prof_med.empty:
                st.dataframe(prof_med, use_container_width=True)

        with right_a:
            st.subheader(
                f"Flagged Repetitions (Any: {anom_summ.get('any_flagged', 0)} | Consensus: {anom_summ.get('consensus_flagged', 0)})"
            )
            if not anom_top.empty:
                selected_rep = st.selectbox(
                    "Inspect Flagged Repetition",
                    options=anom_top["Rep_ID"].tolist(),
                    index=0,
                )
                row_sel = anom_top[anom_top["Rep_ID"] == selected_rep].iloc[0]
                st.success(
                    f"**{row_sel['Rep_ID']}** • Rank `#{int(row_sel['Severity_Rank'])}` • "
                    f"Severity `{row_sel['Severity_Score']}` • {row_sel['Status']}\n\n"
                    f"**Top Driving Features:** {row_sel['Top_Driving_Features']}"
                )

                show_cols = [
                    c
                    for c in [
                        "Severity_Rank",
                        "Rep_ID",
                        "Cluster",
                        "Status",
                        "Severity_Score",
                        "Cluster_Rel_Z_Norm",
                        "Top_Driving_Features",
                    ]
                    if c in anom_top.columns
                ]
                st.dataframe(anom_top[show_cols].head(40), use_container_width=True, hide_index=True)

    st.divider()
    st.warning(
        "Limitation & Clinical Note: Cluster groups, severity rankings, and anomaly flags reflect statistical "
        "kinematic patterns and deviations in unlabeled motion features—they do not constitute an automated "
        "medical, clinical, or ergonomic diagnosis."
    )

    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Full Unsupervised Results (CSV)",
        data=csv_bytes,
        file_name="intellirehabds_unsupervised_results_v2.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()