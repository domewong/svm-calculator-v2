# app.py
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

# ============== 配置 ==============
st.set_page_config(
    page_title="Respiratory Failure Risk Calculator (SVM)",
    layout="wide",
)

# ============== 资源路径（与 app.py 同目录） ==============
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "svm_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
BG_PATH = BASE_DIR / "shap_background.pkl"

# 你的特征顺序（必须与训练时一致）
FEATURE_COLS = ["Age", "PaO2", "PF_ratio", "pneumonia", "ISS"]


# ============== 工具函数 ==============
def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_assets():
    # 检查文件是否齐全
    missing = []
    for p in [MODEL_PATH, SCALER_PATH]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("Missing required file(s):\n" + "\n".join(missing))

    model = load_pickle(MODEL_PATH)
    scaler = load_pickle(SCALER_PATH)

    # background 不是必须（用于 SHAP）
    bg = None
    if BG_PATH.exists():
        bg = load_pickle(BG_PATH)

    return model, scaler, bg


def safe_predict_proba(model, X_scaled: pd.DataFrame) -> float:
    """返回正类概率"""
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
    """把 background 处理成 (n, n_features) 的 numpy array，并确保列顺序一致"""
    if bg is None:
        return np.zeros((50, len(FEATURE_COLS)), dtype=float)

    if isinstance(bg, pd.DataFrame):
        bg_arr = bg[FEATURE_COLS].to_numpy()
    else:
        bg_arr = np.asarray(bg)
        if bg_arr.ndim == 1:
            bg_arr = bg_arr.reshape(1, -1)

    # 防御：列数不匹配则退化
    if bg_arr.ndim != 2 or bg_arr.shape[1] != len(FEATURE_COLS):
        bg_arr = np.zeros((50, len(FEATURE_COLS)), dtype=float)

    return bg_arr


@st.cache_resource
def build_shap_explainer(_model, bg):
    """
    ✅ 核心修复：
    - 参数名用 _model：Streamlit 不会对它做 hash，避免 Cannot hash argument 'model'
    """
    import shap

    bg_arr = _prepare_bg(bg)

    # 用 predict_proba 作为 black-box 解释入口
    explainer = shap.Explainer(_model.predict_proba, bg_arr)
    return explainer


def _extract_positive_class_explanation(sv, X_scaled: pd.DataFrame):
    """
    兼容不同 SHAP 输出形态：
    - 常见二分类：values shape (1, n_features, 2) / base_values shape (1,2)
    - 也可能：values shape (1, n_features) / base_values shape (1,)
    返回：shap.Explanation（用于 waterfall）
    """
    import shap

    values = getattr(sv, "values", None)
    base_values = getattr(sv, "base_values", None)

    if values is None or base_values is None:
        raise ValueError("Unexpected SHAP output: missing values/base_values.")

    values = np.asarray(values)
    base_values = np.asarray(base_values)

    # data：用于显示特征值
    data_row = X_scaled.iloc[0].values

    # Case A: (1, n_features, 2) -> 取正类 index=1
    if values.ndim == 3 and values.shape[-1] == 2:
        v = values[0, :, 1]
        if base_values.ndim >= 2 and base_values.shape[-1] == 2:
            bv = float(base_values[0, 1])
        else:
            # 兜底：取第一个
            bv = float(np.ravel(base_values)[0])

    # Case B: (1, n_features) -> 直接用
    elif values.ndim == 2 and values.shape[0] == 1:
        v = values[0, :]
        bv = float(np.ravel(base_values)[0])

    else:
        raise ValueError(f"Unexpected SHAP values shape: {values.shape}, base_values shape: {base_values.shape}")

    exp = shap.Explanation(
        values=v,
        base_values=bv,
        data=data_row,
        feature_names=FEATURE_COLS
    )
    return exp


def render_shap_waterfall(explainer, X_scaled: pd.DataFrame):
    """
    生成单病例 SHAP waterfall
    返回 matplotlib figure
    """
    import shap  # 放内部：云端装包失败时不影响主预测

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
st.caption("输入临床变量 → 输出个体风险（概率） + 单病例 SHAP 解释（waterfall）")
st.info("提示：该工具用于科研展示与辅助决策，不替代临床医生判断。")

# 侧边栏输入
st.sidebar.header("Input features")

age = st.sidebar.number_input("Age (years)", min_value=0.0, max_value=120.0, value=60.0, step=1.0)
pao2 = st.sidebar.number_input("PaO₂ (mmHg)", min_value=0.0, max_value=800.0, value=82.0, step=1.0)
pf_ratio = st.sidebar.number_input("PF ratio (PaO₂/FiO₂)", min_value=0.0, max_value=1000.0, value=250.0, step=1.0)

pneumonia01 = st.sidebar.selectbox("Pneumonia (0=No, 1=Yes)", options=[0, 1], index=1)

iss = st.sidebar.number_input("ISS (Injury Severity Score)", min_value=0.0, max_value=75.0, value=26.0, step=1.0)

st.sidebar.markdown("---")

# 阈值
pt = st.sidebar.slider("Decision threshold (pt)", min_value=0.05, max_value=0.95, value=0.54, step=0.01)
st.sidebar.caption("建议用于论文阈值解释：pt=0.20 / 0.40 / 0.60（三档）")

# 主区布局
col_left, col_right = st.columns([1.05, 1.0], gap="large")

# ============== 主流程：加载模型并预测 ==============
try:
    model, scaler, bg = load_assets()
    X_raw = build_single_case_df(age, pao2, pf_ratio, pneumonia01, iss)
    X_scaled = scale_features(scaler, X_raw)
    prob = safe_predict_proba(model, X_scaled)

    # 预测标签
    pred_label = int(prob >= pt)
    risk_text = "High risk" if pred_label == 1 else "Low risk"

    # cost:benefit = pt/(1-pt)
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

        st.markdown("**Raw input:**")
        st.dataframe(X_raw, use_container_width=True)

        st.download_button(
            label="Download this case (CSV)",
            data=to_csv_download(X_raw),
            file_name="svm_single_case.csv",
            mime="text/csv"
        )

    # ============== SHAP（右侧） ==============
    with col_right:
        st.subheader("Single-case SHAP (waterfall)")

        # ✅ 建议默认关闭：云端 SHAP 可能慢/依赖不稳，用户需要再开
        enable_shap = st.toggle("Enable SHAP explanation", value=False)

        if not enable_shap:
            st.info("已关闭 SHAP 解释。开启后将计算单病例 SHAP waterfall。")
        else:
            try:
                explainer = build_shap_explainer(model, bg)
                fig = render_shap_waterfall(explainer, X_scaled)
                st.pyplot(fig, clear_figure=True)
                st.caption("说明：红色条表示提高预测风险，蓝色条表示降低预测风险（相对于基线）。")
            except Exception as e:
                st.warning(
                    "SHAP 解释生成失败（不影响概率输出）。常见原因：云端环境下 shap/numba/llvmlite 依赖构建失败或计算超时。"
                )
                st.code(str(e))

except Exception as e:
    st.error("App 启动失败：请检查 requirements/runtime 与 pkl 文件是否齐全。")
    st.code(str(e))
