# app.py
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

# ============== Page config ==============
st.set_page_config(
    page_title="Respiratory Failure Risk Calculator (SVM)",
    layout="wide",
)

# ============== Paths (same directory as app.py) ==============
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "svm_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
BG_PATH = BASE_DIR / "shap_background.pkl"

# Feature order (must match training)
FEATURE_COLS = ["Age", "PaO2", "PF_ratio", "pneumonia", "ISS"]


# ============== Utilities ==============
def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_assets():
    missing = []
    for p in [MODEL_PATH, SCALER_PATH]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("Missing required file(s):\n" + "\n".join(missing))

    model = load_pickle(MODEL_PATH)
    scaler = load_pickle(SCALER_PATH)

    bg = None
    if BG_PATH.exists():
        bg = load_pickle(BG_PATH)

    return model, scaler, bg


def safe_predict_proba(model, X_scaled: pd.DataFrame) -> float:
    """Return positive-class probability."""
    proba = model.predict_proba(X_scaled)
    return float(proba[0, 1])


def build_single_case_df(age, pao2, pf_ratio, pneumonia01, iss) -> pd.DataFrame:
    df = pd.DataFrame([{
        "Age": float(age),
        "PaO2": float(pao2),
        "PF_ratio": float(pf_ratio),
        "pneumonia": int(pneumonia01),
        "ISS": float(iss),
    }])
    return df[FEATURE_COLS]


def scale_features(scaler, X: pd.DataFrame) -> pd.DataFrame:
    X_np = scaler.transform(X.values)
    return pd.DataFrame(X_np, columns=FEATURE_COLS)


def _prepare_bg(bg):
    """Ensure background is a 2D numpy array with correct feature order."""
    if bg is None:
        return np.zeros((50, len(FEATURE_COLS)), dtype=float)

    if isinstance(bg, pd.DataFrame):
        bg_arr = bg[FEATURE_COLS].to_numpy()
    else:
        bg_arr = np.asarray(bg)
        if bg_arr.ndim == 1:
            bg_arr = bg_arr.reshape(1, -1)

    if bg_arr.ndim != 2 or bg_arr.shape[1] != len(FEATURE_COLS):
        bg_arr = np.zeros((50, len(FEATURE_COLS)), dtype=float)

    return bg_arr


@st.cache_resource
def build_shap_explainer(_model, bg):
    """
    Key fix:
    - Use parameter name `_model` so Streamlit does NOT hash the model object,
      avoiding: Cannot hash argument 'model' (sklearn SVC).
    """
    import shap

    bg_arr = _prepare_bg(bg)
    explainer = shap.Explainer(_model.predict_proba, bg_arr)
    return explainer


def _extract_positive_class_explanation(sv, X_scaled: pd.DataFrame):
    """
    Robustly extract a single-case explanation for the positive class (class=1),
    compatible with common SHAP output shapes:
    - values: (1, n_features, 2) and base_values: (1, 2)
    - values: (1, n_features) and base_values: (1,)
    """
    import shap

    values = getattr(sv, "values", None)
    base_values = getattr(sv, "base_values", None)

    if values is None or base_values is None:
        raise ValueError("Unexpected SHAP output: missing values/base_values.")

    values = np.asarray(values)
    base_values = np.asarray(base_values)

    data_row = X_scaled.iloc[0].values

    # Binary classification output: (1, n_features, 2)
    if values.ndim == 3 and values.shape[-1] == 2:
        v = values[0, :, 1]
        if base_values.ndim >= 2 and base_values.shape[-1] == 2:
            bv = float(base_values[0, 1])
        else:
            bv = float(np.ravel(base_values)[0])

    # Single output: (1, n_features)
    elif values.ndim == 2 and values.shape[0] == 1:
        v = values[0, :]
        bv = float(np.ravel(base_values)[0])

    else:
        raise ValueError(
            f"Unexpected SHAP shapes: values={values.shape}, base_values={base_values.shape}"
        )

    exp = shap.Explanation(
        values=v,
        base_values=bv,
        data=data_row,
        feature_names=FEATURE_COLS
    )
    return exp


def render_shap_waterfall(explainer, X_scaled: pd.DataFrame):
    """Single-case SHAP waterfall plot (matplotlib figure)."""
    import shap

    sv = explainer(X_scaled)
    exp = _extract_positive_class_explanation(sv, X_scaled)

    fig = plt.figure(figsize=(8.5, 5.2))
    shap.plots.waterfall(exp, max_display=len(FEATURE_COLS), show=False)
    plt.tight_layout()
    return fig


def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# ============== UI ==============
st.title("🫁 Respiratory Failure Risk Calculator (SVM)")
st.caption("Enter clinical variables → get individual risk probability + optional single-case SHAP waterfall explanation.")
st.info("For research and decision support only. Not a substitute for clinical judgment.")

# Sidebar inputs
st.sidebar.header("Input features")

age = st.sidebar.number_input("Age (years)", min_value=0.0, max_value=120.0, value=60.0, step=1.0)
pao2 = st.sidebar.number_input("PaO₂ (mmHg)", min_value=0.0, max_value=800.0, value=82.0, step=1.0)
pf_ratio = st.sidebar.number_input("PF ratio (PaO₂/FiO₂)", min_value=0.0, max_value=1000.0, value=250.0, step=1.0)

pneumonia01 = st.sidebar.selectbox("Pneumonia (0=No, 1=Yes)", options=[0, 1], index=1)

iss = st.sidebar.number_input("ISS (Injury Severity Score)", min_value=0.0, max_value=75.0, value=26.0, step=1.0)

st.sidebar.markdown("---")

pt = st.sidebar.slider("Decision threshold (pt)", min_value=0.05, max_value=0.95, value=0.50, step=0.01)
st.sidebar.caption("Tip (paper-friendly): consider pt = 0.20 / 0.40 / 0.60 as three reference thresholds.")

# Main layout
col_left, col_right = st.columns([1.05, 1.0], gap="large")

# ============== Main workflow ==============
try:
    model, scaler, bg = load_assets()
    X_raw = build_single_case_df(age, pao2, pf_ratio, pneumonia01, iss)
    X_scaled = scale_features(scaler, X_raw)
    prob = safe_predict_proba(model, X_scaled)

    pred_label = int(prob >= pt)
    risk_text = "High risk" if pred_label == 1 else "Low risk"

    cbr = pt / (1 - pt)

    with col_left:
        st.subheader("Prediction")

        st.markdown("**Predicted risk (probability)**")
        st.metric(label="", value=f"{prob:.3f}")

        if pred_label == 1:
            st.error(f"Decision (pt={pt:.2f}): **{risk_text}**")
        else:
            st.success(f"Decision (pt={pt:.2f}): **{risk_text}**")

        st.caption(f"Cost:Benefit ratio = pt/(1-pt) = {cbr:.3f}")

        st.markdown("**Raw input (unscaled):**")
        st.dataframe(X_raw, use_container_width=True)

        st.download_button(
            label="Download this case (CSV)",
            data=to_csv_download(X_raw),
            file_name="svm_single_case.csv",
            mime="text/csv"
        )

    with col_right:
        st.subheader("Single-case explanation (SHAP)")

        # Recommended default: off (cloud can be slow)
        enable_shap = st.toggle("Enable SHAP waterfall explanation", value=False)

        if not enable_shap:
            st.info("SHAP explanation is off. Turn it on to compute a single-case SHAP waterfall plot (may be slow in cloud environments).")
        else:
            try:
                explainer = build_shap_explainer(model, bg)
                fig = render_shap_waterfall(explainer, X_scaled)
                st.pyplot(fig, clear_figure=True)
                st.caption("Interpretation: red increases predicted risk; blue decreases predicted risk (relative to the baseline).")
            except Exception as e:
                st.warning(
                    "Failed to generate SHAP explanation (this does NOT affect the risk prediction). "
                    "Common reasons: shap/numba/llvmlite build issues on cloud, or computation timeout."
                )
                st.code(str(e))

except Exception as e:
    st.error("App failed to start. Please check runtime/requirements and that required .pkl files exist.")
    st.code(str(e))
