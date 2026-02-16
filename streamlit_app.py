import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from src.eda import basic_eda
from src.insight_generator import generate_insights
# -------------------------------------------------
# ELITE FEATURE 4: AI-STYLE EXECUTIVE NARRATOR
# -------------------------------------------------
def generate_ai_executive_summary(data, quality_score, risks):

    rows, cols = data.shape

    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    categorical_cols = data.select_dtypes(include="object").columns.tolist()

    summary = ""

    # Dataset scale
    if rows > 10000:
        summary += "The dataset represents a large-scale dataset suitable for robust analytical modeling. "
    elif rows > 1000:
        summary += "The dataset size is moderately large, providing sufficient data for meaningful analysis. "
    else:
        summary += "The dataset is relatively small, and insights should be interpreted cautiously. "

    # Feature composition
    summary += f"It contains {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features. "

    # Data quality interpretation
    if quality_score > 85:
        summary += "Data quality appears strong with minimal structural concerns. "
    elif quality_score > 70:
        summary += "Data quality is acceptable but may benefit from targeted cleaning. "
    else:
        summary += "Data quality issues may significantly impact analytical reliability. "

    # Risk awareness
    if risks:
        summary += "Several structural risks were identified that could influence statistical interpretation. "
    else:
        summary += "No major structural risks were detected. "

    # Correlation insight
    corr_explanation = generate_correlation_insight(data)
    if corr_explanation:
        summary += corr_explanation

    summary += "Overall, the dataset demonstrates actionable analytical potential."

    return summary

# -------------------------------------------------
# CORRELATION INSIGHT FUNCTION
# -------------------------------------------------
def generate_correlation_insight(data):
    numeric_cols = data.select_dtypes(include="number").columns
    if len(numeric_cols) < 2:
        return ""
    
    corr = data[numeric_cols].corr().abs()
    np.fill_diagonal(corr.values, 0)
    max_corr = corr.max().max()
    
    if max_corr > 0.8:
        return "Strong correlations exist between key variables, suggesting potential multicollinearity. "
    elif max_corr > 0.5:
        return "Moderate correlations detected, indicating some relationships between variables. "
    else:
        return "Variables appear largely independent with weak correlations. "

# -------------------------------------------------
# ELITE FEATURE 2: INTELLIGENT RISK DETECTION
# -------------------------------------------------
def detect_structural_risks(data):

    risks = []

    numeric_cols = data.select_dtypes(include="number").columns
    categorical_cols = data.select_dtypes(include="object").columns

    # 1️⃣ High Missing Columns
    missing_percent = (data.isnull().mean()) * 100
    for col in missing_percent.index:
        if missing_percent[col] > 20:
            risks.append(f"⚠ Column '{col}' has {missing_percent[col]:.1f}% missing values.")

    # 2️⃣ Severe Category Imbalance
    for col in categorical_cols:
        top_ratio = data[col].value_counts(normalize=True).max()
        if top_ratio > 0.75:
            risks.append(f"⚠ '{col}' is highly imbalanced ({top_ratio*100:.1f}% in one category).")

    # 3️⃣ Outlier Heavy Columns
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outlier_ratio = ((data[col] < lower) | (data[col] > upper)).mean()

        if outlier_ratio > 0.10:
            risks.append(f"⚠ '{col}' contains {outlier_ratio*100:.1f}% extreme outliers.")

    # 4️⃣ Near Constant Columns
    for col in numeric_cols:
        if data[col].std() < 0.01:
            risks.append(f"⚠ '{col}' has near-zero variance (may be uninformative).")

    # 5️⃣ Multicollinearity
    if len(numeric_cols) > 1:
        corr = data[numeric_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)
        if corr.max().max() > 0.9:
            risks.append("⚠ Strong multicollinearity detected between numeric variables.")

    return risks

# -------------------------------------------------
# ELITE FEATURE 1: SMART DATA QUALITY ENGINE
# -------------------------------------------------
def calculate_data_quality_score(data):

    score = 100
    total_cells = data.shape[0] * data.shape[1]

    # 1️⃣ Missing Values Penalty
    missing_percent = (data.isnull().sum().sum() / total_cells) * 100
    score -= missing_percent * 0.5  # weighted penalty

    # 2️⃣ Duplicate Rows Penalty
    duplicate_percent = (data.duplicated().sum() / len(data)) * 100
    score -= duplicate_percent * 0.7

    # 3️⃣ Outlier Penalty (numeric only)
    numeric_cols = data.select_dtypes(include="number").columns

    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outlier_ratio = ((data[col] < lower) | (data[col] > upper)).mean()
        score -= outlier_ratio * 10

    # 4️⃣ Multicollinearity Penalty
    if len(numeric_cols) > 1:
        corr = data[numeric_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)
        if corr.max().max() > 0.9:
            score -= 8

    return max(round(score, 2), 0)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Insight Engine",
    page_icon="🚀",
    layout="wide"
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("⚙ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
st.sidebar.markdown("---")
st.sidebar.caption("AI-Powered Automated Analytics Tool")

# -------------------------------------------------
# PREMIUM HEADER
# -------------------------------------------------
st.markdown("""
<h1 style='text-align: center;
background: linear-gradient(90deg, #00c6ff, #0072ff);
-webkit-background-clip: text;
color: transparent;
font-weight: 800;'>
AI Insight Engine
</h1>
""", unsafe_allow_html=True)

st.caption("Smart statistical analysis & automated insight generation")
st.markdown("---")

# -------------------------------------------------
# MAIN APP
# -------------------------------------------------
if uploaded_file:

    data = pd.read_csv(uploaded_file)
    eda = basic_eda(data)

    # -------------------------------------------------
    # KPI SECTION
    # -------------------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", eda["shape"][0])
    col2.metric("Columns", eda["shape"][1])
    col3.metric("Missing Values", data.isnull().sum().sum())

    st.markdown("---")
    # -------------------------------
# DATA QUALITY SECTION
# -------------------------------
    quality_score = calculate_data_quality_score(data)

    st.subheader("📊 Data Quality Score")

    if quality_score > 85:
        st.success(f"{quality_score}/100 — Excellent Data Quality")
    elif quality_score > 70:
        st.warning(f"{quality_score}/100 — Moderate Quality")
    else:
        st.error(f"{quality_score}/100 — Needs Cleaning")
    # -------------------------------
# RISK DETECTION SECTION
# -------------------------------
    st.subheader("🚨 Structural Risk Detection")

    risks = detect_structural_risks(data)

    if risks:
        for r in risks:
            st.warning(r)
        else:
            st.success("No major structural risks detected.")

    # -------------------------------------------------
    # EXECUTIVE SUMMARY
    # -------------------------------------------------
    st.subheader("📄 Executive Summary")
    summary_text = generate_ai_executive_summary(data, quality_score, risks)
    st.info(summary_text)

    st.markdown("---")

    # -------------------------------------------------
    # INTELLIGENT VISUAL ANALYTICS
    # -------------------------------------------------
    st.subheader("📊 Intelligent Visual Analytics")

    numeric_cols = data.select_dtypes(include="number").columns.tolist()

    if numeric_cols:

        selected_col = st.selectbox("Select Numeric Column", numeric_cols)

        colA, colB = st.columns(2)

        mean_val = data[selected_col].mean()
        median_val = data[selected_col].median()
        std_val = data[selected_col].std()
        min_val = data[selected_col].min()
        max_val = data[selected_col].max()

        # ADAPTIVE CHART LOGIC
        if len(data) < 30:
            with colA:
                freq_data = data[selected_col].value_counts().sort_index()

                fig_bar = px.bar(
                    x=freq_data.values,
                    y=freq_data.index,
                    orientation='h',
                    title=f"{selected_col} Frequency Distribution",
                    template="plotly_dark"
                )

                fig_bar.update_layout(
                    yaxis_title=selected_col,
                    xaxis_title="Frequency"
                )

                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            with colA:
                fig_hist = px.histogram(
                    data,
                    x=selected_col,
                    nbins=20,
                    title=f"{selected_col} Distribution",
                    template="plotly_dark"
                )

                fig_hist.add_vline(
                    x=median_val,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Median"
                )

                st.plotly_chart(fig_hist, use_container_width=True)

        # BOX PLOT
        with colB:
            fig_box = px.box(
                data,
                y=selected_col,
                title=f"{selected_col} Spread & Outliers",
                template="plotly_dark"
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # INTERPRETATION
        st.markdown("### 🧠 Interpretation")

        explanation = f"""
- **Range:** {min_val:.2f} to {max_val:.2f}
- **Mean:** {mean_val:.2f}
- **Median:** {median_val:.2f}
- **Standard Deviation:** {std_val:.2f}
"""

        if std_val > mean_val * 0.5:
            explanation += "\n- High variability detected."

        if abs(mean_val - median_val) > std_val * 0.3:
            explanation += "\n- Distribution may be skewed."

        st.markdown(explanation)

    st.markdown("---")

    # -------------------------------------------------
    # CORRELATION INTELLIGENCE
    # -------------------------------------------------
    st.subheader("🔥 Correlation Intelligence")

    if len(numeric_cols) > 1:

        corr = data[numeric_cols].corr()

        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu",
                zmin=-1,
                zmax=1
            )
        )

        fig_corr.update_layout(template="plotly_dark", height=500)

        st.plotly_chart(fig_corr, use_container_width=True)

        # RANKING
        st.markdown("### 📈 Strongest Relationships")

        corr_abs = corr.abs()
        np.fill_diagonal(corr_abs.values, 0)

        pairs = []

        for i in range(len(corr_abs.columns)):
            for j in range(i+1, len(corr_abs.columns)):
                pairs.append((
                    corr_abs.columns[i] + " & " + corr_abs.columns[j],
                    corr_abs.iloc[i, j]
                ))

        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

        if pairs_sorted:
            pair_names = [p[0] for p in pairs_sorted]
            pair_values = [p[1] for p in pairs_sorted]

            fig_rank = px.bar(
                x=pair_values,
                y=pair_names,
                orientation="h",
                title="Correlation Strength Ranking",
                template="plotly_dark"
            )

            st.plotly_chart(fig_rank, use_container_width=True)

            strongest_pair, strongest_value = pairs_sorted[0]

            if strongest_value > 0.8:
                st.success("Very strong relationship detected.")
            elif strongest_value > 0.5:
                st.warning("Moderate relationship detected.")
            else:
                st.info("Weak relationship detected.")

    st.markdown("---")

    # -------------------------------------------------
    # AI INSIGHTS
    # -------------------------------------------------
    st.subheader("🤖 AI-Generated Insights")

    insights = generate_insights(data)

    for insight in insights:
        st.markdown(f"• {insight}")

    st.download_button(
        label="📥 Download Insight Report",
        data="\n".join(insights),
        file_name="ai_insight_report.txt",
        mime="text/plain"
    )

    st.success("Analysis Completed Successfully 🚀")
