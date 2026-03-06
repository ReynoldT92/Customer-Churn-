
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Churn Prediction – Telecom",
    page_icon=":satellite_antenna:",
    layout="wide"
)

# ── Data ──────────────────────────────────────────────────────────────────────
METRICS = ["Precision", "Recall", "PR AUC"]

results = {
    "LogReg": {"Precision": 0.530, "Recall": 0.644, "PR AUC": 0.569},
    "Random Forest": {"Precision": 0.677, "Recall": 0.511, "PR AUC": 0.660},
    "XGBoost": {"Precision": 0.509, "Recall": 0.794, "PR AUC": 0.665},
}

COLORS = {
    "LogReg": "#6366F1",
    "Random Forest": "#10B981",
    "XGBoost": "#F59E0B",
}

PR_CURVES = {
    "LogReg": [[0, 1], [0.10, 0.82], [0.20, 0.74], [0.35, 0.67], [0.50, 0.61], [0.65, 0.55], [0.80, 0.48], [0.90, 0.42], [1, 0.27]],
    "Random Forest": [[0, 1], [0.10, 0.88], [0.20, 0.82], [0.35, 0.74], [0.50, 0.68], [0.65, 0.60], [0.80, 0.52], [0.90, 0.44], [1, 0.27]],
    "XGBoost": [[0, 1], [0.10, 0.87], [0.20, 0.81], [0.35, 0.73], [0.50, 0.67], [0.65, 0.59], [0.80, 0.51], [0.90, 0.43], [1, 0.27]],
}

ASSUMPTIONS = {
    "total_customers": 1409,
    "churn_rate": 0.27,
    "monthly_revenue": 65,
    "intervention_cost": 35,
    "avg_tenure": 24,
    "save_rate": 0.50,
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("### :satellite_antenna: Predicting Customer Churn in the Telecom Industry")
st.markdown("**A Machine Learning Approach to Proactive Retention**")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    ":bar_chart: Model Comparison",
    ":chart_with_upwards_trend: PR Curve",
    ":moneybag: Business ROI"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Algorithm Tournament: LogReg vs Random Forest vs XGBoost")

    highlight = st.radio(
        "Highlight model:",
        options=list(results.keys()),
        horizontal=True,
        index=2
    )

    df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("#0F172A")

    for ax, metric in zip(axes, METRICS):
        ax.set_facecolor("#1E293B")
        for spine in ax.spines.values():
            spine.set_visible(False)

        vals = [results[m][metric] for m in results]
        models = list(results.keys())
        colors = [COLORS[m] if m == highlight else COLORS[m] + "55" for m in models]

        bars = ax.barh(models, vals, color=colors, height=0.5)

        for i, (bar, val) in enumerate(zip(bars, vals)):
            ax.text(
                val + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                ha="left",
                color="white",
                fontsize=10,
                fontweight="bold" if models[i] == highlight else "normal"
            )

        ax.set_xlim(0, 1.0)
        ax.set_title(metric, color="white", fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(colors="white")
        ax.xaxis.set_visible(False)

        for label in ax.get_yticklabels():
            name = label.get_text()
            label.set_color(COLORS[name] if name == highlight else "#94A3B8")
            label.set_fontweight("bold" if name == highlight else "normal")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        ":large_yellow_circle: **XGBoost** catches the most churners — highest Recall (0.794) and PR AUC (0.665).\n"
        ":large_green_circle: **Random Forest** is most precise (0.677) — better when outreach budget is limited."
    )

    with st.expander("View full results table"):
        st.dataframe(
            pd.DataFrame(results).T.style.highlight_max(axis=0, color="#10b98133").format("{:.3f}"),
            use_container_width=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – PR CURVE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Precision-Recall Curve")

    hl_curve = st.radio(
        "Highlight model:",
        options=list(results.keys()),
        horizontal=True,
        index=2,
        key="curve_radio"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    ax.axhline(y=0.27, color="#475569", linestyle="--", linewidth=1, label="No-skill baseline (0.27)")

    for model, pts in PR_CURVES.items():
        pts_arr = np.array(pts)
        alpha = 1.0 if model == hl_curve else 0.3
        lw = 2.5 if model == hl_curve else 1.2
        ax.plot(
            pts_arr[:, 0],
            pts_arr[:, 1],
            color=COLORS[model],
            linewidth=lw,
            alpha=alpha,
            label=f"{model} (AUC {results[model]['PR AUC']:.3f})"
        )

    ax.set_xlabel("Recall", color="white", fontsize=11)
    ax.set_ylabel("Precision", color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    ax.legend(facecolor="#1E293B", labelcolor="white", fontsize=10)
    plt.tight_layout()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("#### Why PR AUC?")
        st.markdown(
            "With a **27% churn rate** the dataset is imbalanced.\n\n"
            "ROC AUC can look artificially high because it includes true negatives "
            "(the majority class).\n\n"
            "**PR AUC focuses only on the minority class** — churners — "
            "which is exactly what matters for a retention campaign."
        )
        st.metric("XGBoost PR AUC", "0.665", delta="+0.096 vs baseline")
        st.metric("Random Forest PR AUC", "0.660", delta="+0.091 vs baseline")
        st.metric("LogReg PR AUC", "0.569", delta="+0.000")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – BUSINESS ROI
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Business ROI – Retention Campaign Simulator")

    col_obj, col_spacer = st.columns([2, 3])
    with col_obj:
        objective = st.selectbox(
            "Business objective:",
            options=[
                "Recall (catch as many churners as possible)",
                "Precision (minimise wasted outreach)"
            ],
            index=0
        )

    obj_key = "Recall" if "Recall" in objective else "Precision"
    recommended = max(results, key=lambda m: (results[m][obj_key], results[m]["PR AUC"]))
    rec_data = results[recommended]

    color_tag = "orange" if recommended == "XGBoost" else "green"
    st.markdown(
        f"**Recommended model:** :{color_tag}[{recommended}] · "
        f"{obj_key}: `{rec_data[obj_key]:.3f}` · PR AUC: `{rec_data['PR AUC']:.3f}`"
    )
    st.divider()

    with st.expander(":gear: Edit assumptions", expanded=False):
        c1, c2, c3 = st.columns(3)
        total_customers = c1.number_input("Test set size", value=ASSUMPTIONS["total_customers"], step=100)
        churn_rate = c1.slider("Churn rate", 0.10, 0.50, ASSUMPTIONS["churn_rate"], 0.01)
        monthly_revenue = c2.number_input("Avg monthly revenue ($)", value=ASSUMPTIONS["monthly_revenue"], step=5)
        intervention_cost = c2.number_input("Intervention cost ($)", value=ASSUMPTIONS["intervention_cost"], step=5)
        avg_tenure = c3.number_input("Avg tenure (months)", value=ASSUMPTIONS["avg_tenure"], step=1)
        save_rate = c3.slider("Retention success rate", 0.10, 0.90, ASSUMPTIONS["save_rate"], 0.05)

    actual_churners = int(total_customers * churn_rate)
    tp = int(actual_churners * rec_data["Recall"])
    fp = int(tp / rec_data["Precision"]) - tp if rec_data["Precision"] > 0 else 0
    contacted = tp + fp
    saved_revenue = int(tp * monthly_revenue * avg_tenure * save_rate)
    campaign_cost = int(contacted * intervention_cost)
    net_roi = saved_revenue - campaign_cost
    roi_pct = int((net_roi / campaign_cost) * 100) if campaign_cost else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Customers Contacted", f"{contacted:,}")
    k2.metric("Churners Identified", f"{tp:,}")
    k3.metric("Campaign Cost", f"${campaign_cost:,}")
    k4.metric("Revenue Retained", f"${saved_revenue:,}")
    k5.metric("Net ROI", f"${net_roi:,}", delta=f"${net_roi:,}")
    k6.metric("ROI %", f"{roi_pct}%", delta=f"{roi_pct}%")

    st.divider()

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    for spine in ax.spines.values():
        spine.set_visible(False)

    labels = ["Revenue\nRetained", "Campaign\nCost", "Net ROI"]
    values = [saved_revenue, -campaign_cost, net_roi]
    colors = ["#10B981", "#EF4444", "#6366F1" if net_roi > 0 else "#EF4444"]

    bars = ax.bar(labels, [abs(v) for v in values], color=colors, width=0.4)

    max_val = max(abs(v) for v in values) if values else 1
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.02,
            f"${abs(val):,}",
            ha="center",
            color="white",
            fontsize=11,
            fontweight="bold"
        )

    ax.set_ylabel("USD ($)", color="white")
    ax.tick_params(colors="white")
    ax.yaxis.set_visible(False)
    plt.tight_layout()

    col_chart, col_note = st.columns([2, 1])
    with col_chart:
        st.pyplot(fig)
        plt.close(fig)

    with col_note:
        st.markdown("#### Assumptions (labeled)")
        st.markdown(
            f"- Test set: **{total_customers:,} customers**\n"
            f"- Churn rate: **{churn_rate:.0%}**\n"
            f"- Avg monthly revenue: **${monthly_revenue}**\n"
            f"- Intervention cost: **${intervention_cost}** (call + discount)\n"
            f"- Avg tenure: **{avg_tenure} months**\n"
            f"- Retention success rate: **{save_rate:.0%}**"
        )
