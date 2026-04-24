from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_DIR = PROJECT_ROOT / "models"

st.set_page_config(page_title="PSO Fraud Detection", layout="wide", page_icon="💳")
st.title("💳 PSO Feature Selection — Credit Card Fraud Detection")
st.caption("Kaggle creditcard.csv | Binary PSO + Random Forest")


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_json(file_path: Path):
    if not file_path.exists():
        return None
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ── Load artifacts ─────────────────────────────────────────────────────────────
summary = load_json(REPORT_DIR / "project_summary.json")
metrics_summary = load_json(REPORT_DIR / "metrics_summary.json")
predictions_path = REPORT_DIR / "predictions.csv"
pso_history_path = REPORT_DIR / "pso_history.csv"
importance_path = REPORT_DIR / "feature_importance.csv"

if summary is None or metrics_summary is None or not predictions_path.exists():
    st.warning(
        "Artifacts not found. Run `python run_training.py` first, "
        "then relaunch: `streamlit run app/dashboard.py`"
    )
    st.stop()

predictions_df = pd.read_csv(predictions_path)
pso_history_df = pd.read_csv(pso_history_path) if pso_history_path.exists() else pd.DataFrame()
importance_df = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_overview, tab_pso, tab_threshold, tab_predict = st.tabs(
    ["📊 Overview", "🔬 PSO Analysis", "🎚️ Threshold Tuning", "🔍 Predict Transaction"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    data_summary = summary["data_summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{data_summary['rows']:,}")
    c2.metric("Fraud Cases", f"{data_summary['fraud_cases']:,}")
    c3.metric("Fraud Ratio", f"{100 * data_summary['fraud_ratio']:.3f}%")
    c4.metric("PSO Selected Features", summary["selected_feature_count"])

    st.subheader("Model Comparison")
    comparison_df = pd.DataFrame(
        [
            {"Model": "Baseline (Logistic)", **metrics_summary["baseline"]},
            {"Model": "PSO + Random Forest", **metrics_summary["pso_model"]},
        ]
    )
    show_cols = ["Model", "roc_auc", "pr_auc", "precision", "recall", "f1", "balanced_accuracy", "mcc", "selected_feature_count"]
    st.dataframe(comparison_df[[c for c in show_cols if c in comparison_df.columns]].round(4), use_container_width=True)

    img1, img2 = st.columns(2)
    if (FIGURE_DIR / "baseline_confusion_matrix.png").exists():
        img1.image(str(FIGURE_DIR / "baseline_confusion_matrix.png"), caption="Baseline Confusion Matrix")
    if (FIGURE_DIR / "pso_confusion_matrix.png").exists():
        img2.image(str(FIGURE_DIR / "pso_confusion_matrix.png"), caption="PSO Model Confusion Matrix")

    rc1, rc2 = st.columns(2)
    if (FIGURE_DIR / "roc_comparison.png").exists():
        rc1.image(str(FIGURE_DIR / "roc_comparison.png"), caption="ROC Curve Comparison")
    if (FIGURE_DIR / "pr_comparison.png").exists():
        rc2.image(str(FIGURE_DIR / "pr_comparison.png"), caption="PR Curve Comparison")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PSO Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab_pso:
    st.subheader("PSO Convergence")

    converged_at = summary.get("pso_converged_at")
    if converged_at:
        st.info(f"⚡ Early stopping triggered at iteration **{converged_at}** — PSO converged.")

    if not pso_history_df.empty:
        fig = px.line(
            pso_history_df, x="iteration", y="best_score",
            markers=True, title="Best Fitness per Iteration",
            labels={"best_score": "Best Fitness Score", "iteration": "Iteration"},
        )
        fig.update_traces(line_color="#2ca02c")
        st.plotly_chart(fig, use_container_width=True)
    elif (FIGURE_DIR / "pso_history.png").exists():
        st.image(str(FIGURE_DIR / "pso_history.png"))

    st.subheader("Selected Features")
    selected = summary["selected_features"]
    st.success(f"**{len(selected)} features selected** from 30 original features ({100*(1-len(selected)/30):.0f}% reduction)")
    st.write(selected)

    if not importance_df.empty:
        bar = px.bar(
            importance_df.head(15),
            x="importance", y="feature", orientation="h",
            title="Top Feature Importances (PSO Model)",
            color="importance", color_continuous_scale="Viridis",
        )
        bar.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Threshold Tuning
# ══════════════════════════════════════════════════════════════════════════════
with tab_threshold:
    st.subheader("Live Threshold Tuning")
    st.caption("Adjust the decision threshold and see how metrics change in real-time.")

    threshold = st.slider(
        "Probability Threshold", 0.05, 0.95,
        float(summary["pso_threshold"]), 0.01,
    )

    y_true = predictions_df["y_true"]
    y_proba = predictions_df["pso_proba"]
    y_pred = (y_proba >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.4f}")
    m2.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.4f}")
    m3.metric("F1 Score", f"{f1_score(y_true, y_pred, zero_division=0):.4f}")
    m4.metric("PR-AUC", f"{average_precision_score(y_true, y_proba):.4f}")

    st.markdown("**Confusion Matrix at current threshold:**")
    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual Normal", "Actual Fraud"],
        columns=["Predicted Normal", "Predicted Fraud"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.caption(f"FP = {fp} (false alarms) | FN = {fn} (missed frauds) | TP = {tp} | TN = {tn}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Predict Transaction
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("🔍 Live Transaction Fraud Predictor")
    st.caption("Enter feature values to get a real-time fraud prediction from the trained PSO model.")

    model_path = MODEL_DIR / "pso_selected_model.joblib"

    if not model_path.exists():
        st.warning("Trained model not found. Run training first.")
    else:
        import joblib
        bundle = joblib.load(model_path)
        features = bundle["features"]
        pipeline = bundle["pipeline"]
        threshold_val = float(bundle["threshold"])

        st.info(f"Model uses **{len(features)} features**: {', '.join(features)}")

        with st.form("predict_form"):
            st.markdown("**Enter feature values** (V-features are PCA components; unknown = leave at 0.0):")
            cols = st.columns(3)
            inputs = {}
            for i, feat in enumerate(features):
                default = 0.0
                if feat == "Amount":
                    default = 100.0
                inputs[feat] = cols[i % 3].number_input(feat, value=default, format="%.4f", step=0.01)

            submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

        if submitted:
            row = pd.DataFrame([inputs])[features]
            proba = float(pipeline.predict_proba(row)[0, 1])
            prediction = int(proba >= threshold_val)

            st.divider()
            if prediction == 1:
                st.error(f"🚨 **FRAUD DETECTED** — Probability: `{proba:.4f}` (threshold: `{threshold_val:.4f}`)")
            else:
                st.success(f"✅ **LEGITIMATE** — Probability: `{proba:.4f}` (threshold: `{threshold_val:.4f}`)")

            confidence = abs(proba - threshold_val) / max(threshold_val, 1 - threshold_val)
            st.progress(min(confidence, 1.0), text=f"Confidence: {min(confidence, 1.0):.1%}")

        st.divider()
        st.markdown("**Quick test with known fraud patterns** (high V14 negative = typical fraud signal):")
        if st.button("Load example fraud pattern"):
            st.session_state["fraud_example"] = True
            st.info("Set V14 = -10.0, V12 = -8.0, V10 = -5.0, Amount = 1.0 and click Predict.")
