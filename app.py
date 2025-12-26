import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

# ======================
# 基本配置
# ======================
st.set_page_config(
    page_title="Respiratory Failure Risk Calculator (SVM)",
    page_icon="🫁",
    layout="wide"
)

FEATURE_COLS = ["Age", "PaO2", "PF_ratio", "pneumonia", "ISS"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "svm_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
BG_PATH = os.path.join(BASE_DIR, "shap_background.pkl")

# ======================
# 资源加载
# ======================
@st.cache_resource
def load_assets():
    for p in [MODEL_PATH, SCALER_PATH, BG_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {os.path.basename(p)} (expected in repo root)")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(BG_PATH, "rb") as f:
        bg = pickle.load(f)

    # SHAP：用 KernelExplainer（对 SVM 概率输出较稳）
    # bg 建议是 DataFrame 或 ndarray，形状 (n_bg, n_features)
    bg = np.array(bg)
    explainer = shap.KernelExplainer(model.predict_proba, bg, link="logit")
    return model, scaler, explainer, bg

# ======================
# SHAP 绘图：单样本 waterfall（修复你现在的报错）
# ======================
def plot_shap_waterfall(explainer, x_scaled_df, feature_names):
    """
    x_scaled_df: (1, n_features) DataFrame
    """
    # KernelExplainer 返回 list（每个类别一个）或 Explanation
    shap_values = explainer.shap_values(x_scaled_df, nsamples=200)

    # 二分类：通常 shap_values 是 [class0, class1]
    if isinstance(shap_values, list):
        sv = shap_values[1]  # 取正类（事件=1）
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        sv = shap_values
        base_value = explainer.expected_value

    # sv 可能是 (1, n_features)，取第一行，保证是一维
    sv_1d = np.array(sv)[0, :]

    # 用 shap.Explanation 保证 waterfall 接受“单解释”
    exp = shap.Explanation(
        values=sv_1d,
        base_values=base_value,
        data=x_scaled_df.iloc[0, :].values,
        feature_names=feature_names
    )

    fig = plt.figure(figsize=(8, 4.8), dpi=150)
    shap.plots.waterfall(exp, max_display=len(feature_names), show=False)
    plt.tight_layout()
    return fig

# ======================
# 页面标题
# ======================
st.title("🫁 Respiratory Failure Risk Calculator (SVM)")
st.caption("输入临床变量 → 输出个体风险（概率） + 单例 SHAP 解释（waterfall）")
st.info("提示：该工具用于科研展示与辅助决策，不替代临床医生判断。")

# ======================
# 侧边栏输入
# ======================
with st.sidebar:
    st.header("Input features")

    age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=60.0, step=1.0)
    pao2 = st.number_input("PaO₂ (mmHg)", min_value=0.0, max_value=800.0, value=82.0, step=1.0)
    pf_ratio = st.number_input("PF ratio (PaO₂/FiO₂)", min_value=0.0, max_value=1000.0, value=250.0, step=1.0)

    # ✅ 你要求：只显示 Pneumonia，0=No，1=Yes
    pneumonia = st.selectbox("Pneumonia (0=No, 1=Yes)", options=[0, 1], index=1)

    iss = st.number_input("ISS (Injury Severity Score)", min_value=0.0, max_value=75.0, value=26.0, step=1.0)

    st.divider()
    pt_custom = st.slider("Decision threshold (pt)", min_value=0.05, max_value=0.95, value=0.40, step=0.01)
    st.caption("建议用于论文阈值解释：pt=0.20 / 0.40 / 0.60（三档）")

# ======================
# 主区：预测 + SHAP
# ======================
try:
    model, scaler, explainer, bg = load_assets()

    raw = pd.DataFrame([{
        "Age": age,
        "PaO2": pao2,
        "PF_ratio": pf_ratio,
        "pneumonia": pneumonia,
        "ISS": iss
    }])

    # 标准化
    x_scaled_np = scaler.transform(raw[FEATURE_COLS])
    x_scaled = pd.DataFrame(x_scaled_np, columns=FEATURE_COLS)

    # 预测概率
    prob = float(model.predict_proba(x_scaled)[0, 1])
    pred_label = int(prob >= pt_custom)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Prediction")
        st.metric("Predicted risk (probability)", f"{prob:.3f}")

        if pred_label == 1:
            st.error(f"Decision (pt={pt_custom:.2f}): High risk")
        else:
            st.success(f"Decision (pt={pt_custom:.2f}): Low risk")

        cost_benefit = pt_custom / (1 - pt_custom)
        st.caption(f"Cost:Benefit ratio = pt/(1-pt) = {cost_benefit:.3f}")

        st.write("Raw input:")
        st.dataframe(raw, use_container_width=True)

        # 下载当前病例
        csv_bytes = raw.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download this case (CSV)", data=csv_bytes, file_name="single_case.csv", mime="text/csv")

    with right:
        st.subheader("Single-case SHAP (waterfall)")

        # 生成 SHAP waterfall
        fig = plot_shap_waterfall(explainer, x_scaled, FEATURE_COLS)
        st.pyplot(fig, use_container_width=True)

except Exception as e:
    st.error("App 启动失败：请检查 requirements/runtime 与 pkl 文件是否齐全。")
    st.exception(e)
