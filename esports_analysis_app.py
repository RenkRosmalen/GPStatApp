import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Esports Datateam Visualisation Tool")

# -------------------------
# Navigation Tabs
# -------------------------
tabs = st.tabs([
    "Single Dataset Analysis",
    "Comparative T-test (Two Datasets)"
])

# =========================
# TAB 1 — SINGLE DATASET
# =========================
with tabs[0]:
    st.sidebar.header("1. Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # -------------------------
        # Descriptive Stats
        # -------------------------
        st.sidebar.header("2. Descriptive Statistics")
        st.subheader("Descriptive Statistics")
        selected_col = st.selectbox("Select a column for statistics", numeric_cols)
        if selected_col:
            desc = df[selected_col].describe()
            mode = df[selected_col].mode().iloc[0] if not df[selected_col].mode().empty else None
            st.write(f"**Mean:** {desc['mean']:.2f}")
            st.write(f"**Median:** {df[selected_col].median():.2f}")
            st.write(f"**Mode:** {mode}")
            st.write(f"**Standard Deviation:** {desc['std']:.2f}")
            st.write(f"**Min/Max:** {desc['min']} / {desc['max']}")
            st.write(f"**Count (n):** {int(desc['count'])}")

        # -------------------------
        # Frequency / % Change
        # -------------------------
        st.sidebar.header("3. Visualize Frequency / % Change")
        st.subheader("Frequency or Trend Plot")
        time_col = st.selectbox("Time/Match Column (for trend)", df.columns)
        value_col = st.selectbox("Numeric Column (for % change)", numeric_cols)

        if time_col and value_col:
            trend_df = df[[time_col, value_col]].dropna()
            trend_df = trend_df.sort_values(time_col)
            trend_df['% Change'] = trend_df[value_col].pct_change() * 100

            fig = px.line(trend_df, x=time_col, y=value_col, title=f"{value_col} over time")
            st.plotly_chart(fig, use_container_width=True)

            fig_change = px.bar(trend_df, x=time_col, y="% Change", title=f"% Change in {value_col}")
            st.plotly_chart(fig_change, use_container_width=True)

    else:
        st.info("Upload a CSV file to get started.")


# =========================
# TAB 2 — COMPARATIVE T-TEST
# =========================
with tabs[1]:
    st.sidebar.header("Upload Two Datasets for Comparison")

    file1 = st.sidebar.file_uploader("Dataset 1", type="csv", key="file1")
    file2 = st.sidebar.file_uploader("Dataset 2", type="csv", key="file2")

    if file1 and file2:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        st.subheader("Data Previews")
        st.write("**Dataset 1 Preview:**")
        st.dataframe(df1.head())
        st.write("**Dataset 2 Preview:**")
        st.dataframe(df2.head())

        numeric_cols1 = df1.select_dtypes(include=np.number).columns.tolist()
        numeric_cols2 = df2.select_dtypes(include=np.number).columns.tolist()

        # Find common numeric columns
        common_cols = list(set(numeric_cols1).intersection(numeric_cols2))

        if common_cols:
            test_col = st.selectbox("Select a common numeric column to compare", common_cols)
            paired = st.checkbox("Paired T-test (same subjects across datasets)?")

            if test_col:
                data1 = df1[test_col].dropna()
                data2 = df2[test_col].dropna()

                if paired:
                    # Paired test requires equal lengths
                    if len(data1) == len(data2):
                        t_stat, p_val = stats.ttest_rel(data1, data2)
                        test_type = "Paired T-test"
                        # Cohen's d for paired samples
                        diff = data1 - data2
                        cohens_d = diff.mean() / diff.std(ddof=1)
                    else:
                        st.warning("Paired T-test requires datasets of equal length (same participants).")
                        t_stat = p_val = cohens_d = None
                else:
                    t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
                    test_type = "Independent T-test"
                    # Cohen's d for independent samples
                    n1, n2 = len(data1), len(data2)
                    pooled_std = np.sqrt(((n1 - 1)*np.var(data1, ddof=1) + (n2 - 1)*np.var(data2, ddof=1)) / (n1 + n2 - 2))
                    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std

                if t_stat is not None:
                    st.subheader(f"{test_type} Results")
                    st.write(f"**Test on '{test_col}'**")
                    st.write(f"t-statistic = {t_stat:.4f}")
                    st.write(f"p-value = {p_val:.4f}")

                    # Determine significance
                    alpha = 0.05
                    if p_val < alpha:
                        st.success(f"✅ The difference is statistically **significant** (p < {alpha}).")
                    else:
                        st.info(f"❌ The difference is **not statistically significant** (p ≥ {alpha}).")

                    # Interpret Cohen's d
                    if cohens_d is not None:
                        st.write(f"**Cohen’s d = {cohens_d:.3f}**")

                        if abs(cohens_d) < 0.2:
                            effect = "negligible"
                        elif abs(cohens_d) < 0.5:
                            effect = "small"
                        elif abs(cohens_d) < 0.8:
                            effect = "medium"
                        else:
                            effect = "large"

                        st.write(f"Effect size interpretation: **{effect.capitalize()} effect**")

                    # Summary metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean (Dataset 1)", f"{np.mean(data1):.2f}")
                        st.metric("Std Dev (Dataset 1)", f"{np.std(data1, ddof=1):.2f}")
                    with col2:
                        st.metric("Mean (Dataset 2)", f"{np.mean(data2):.2f}")
                        st.metric("Std Dev (Dataset 2)", f"{np.std(data2, ddof=1):.2f}")

                    # Visualization
                    compare_df = pd.DataFrame({
                        "Dataset": ["Dataset 1"] * len(data1) + ["Dataset 2"] * len(data2),
                        test_col: list(data1) + list(data2)
                    })

                    fig = px.box(compare_df, x="Dataset", y=test_col,
                                 title=f"Distribution of {test_col} across datasets ({test_type})")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No common numeric columns found between the two datasets.")
    else:
        st.info("Upload two CSV files to compare.")
