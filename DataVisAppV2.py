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
    "📂 Single Dataset Analysis",
    "📊 Comparative T-test (Two Datasets)"
])

# =========================
# TAB 1 — SINGLE DATASET
# =========================
with tabs[0]:
    st.sidebar.header("1. Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Raw Data Preview")
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # -------------------------
        # Descriptive Stats
        # -------------------------
        st.sidebar.header("2. Descriptive Statistics")
        st.subheader("📈 Descriptive Statistics")
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
        st.subheader("📊 Frequency or Trend Plot")
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

        # -------------------------
        # Inferential Stats
        # -------------------------
        st.sidebar.header("4. Statistical Tests")
        st.subheader("🧪 Inferential Analysis")

        stat_test = st.selectbox("Select a test", ["T-test", "Correlation", "Chi-square"])

        if stat_test == "T-test":
            test_type = st.radio("Select T-test type", ["Independent T-test", "Paired T-test"])

            if test_type == "Independent T-test":
                group_col = st.selectbox("Group column (2 categories)", df.columns)
                test_col = st.selectbox("Numeric column", numeric_cols)

                if group_col and test_col:
                    groups = df[group_col].dropna().unique()
                    if len(groups) == 2:
                        data1 = df[df[group_col] == groups[0]][test_col]
                        data2 = df[df[group_col] == groups[1]][test_col]
                        t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
                        st.write(f"**T-test between {groups[0]} and {groups[1]}**")
                        st.write(f"t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")

                        # Cohen’s d
                        mean1, mean2 = data1.mean(), data2.mean()
                        pooled_std = np.sqrt(((data1.std() ** 2) + (data2.std() ** 2)) / 2)
                        cohens_d = (mean1 - mean2) / pooled_std
                        st.write(f"**Effect Size (Cohen’s d):** {cohens_d:.3f}")

                        if abs(cohens_d) < 0.2:
                            effect_label = "negligible"
                        elif abs(cohens_d) < 0.5:
                            effect_label = "small"
                        elif abs(cohens_d) < 0.8:
                            effect_label = "medium"
                        else:
                            effect_label = "large"
                        st.write(f"This indicates a **{effect_label}** effect size.")

                        # Interpretation
                        alpha = 0.05
                        if p_val < alpha:
                            st.success(
                                f"✅ The difference between {groups[0]} and {groups[1]} is **statistically significant** "
                                f"(p < {alpha}). This suggests {test_col} differs meaningfully between the two groups."
                            )
                        else:
                            st.info(
                                f"ℹ️ The difference between {groups[0]} and {groups[1]} is **not statistically significant** "
                                f"(p = {p_val:.4f} ≥ {alpha}). There’s not enough evidence to say {test_col} differs between groups."
                            )

            elif test_type == "Paired T-test":
                st.write("Upload **two datasets** with the same participants (before/after or condition A/B).")
                file1 = st.file_uploader("Dataset 1 (e.g., Before)", type="csv", key="paired1")
                file2 = st.file_uploader("Dataset 2 (e.g., After)", type="csv", key="paired2")

                if file1 and file2:
                    df1 = pd.read_csv(file1)
                    df2 = pd.read_csv(file2)

                    # Auto-handle duplicate column names
                    df1 = df1.loc[:, ~df1.columns.duplicated()]
                    df2 = df2.loc[:, ~df2.columns.duplicated()]

                    common_cols = [c for c in df1.columns if
                                   c in df2.columns and df1[c].dtype in [np.float64, np.int64]]
                    test_col = st.selectbox("Numeric column to compare", common_cols)

                    if test_col:
                        data1, data2 = df1[test_col], df2[test_col]
                        t_stat, p_val = stats.ttest_rel(data1, data2)
                        st.write(f"**Paired T-test on {test_col}**")
                        st.write(f"t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")

                        # Cohen’s d for paired samples
                        diff = data1 - data2
                        cohens_d = diff.mean() / diff.std(ddof=1)
                        st.write(f"**Effect Size (Cohen’s d):** {cohens_d:.3f}")

                        if abs(cohens_d) < 0.2:
                            effect_label = "negligible"
                        elif abs(cohens_d) < 0.5:
                            effect_label = "small"
                        elif abs(cohens_d) < 0.8:
                            effect_label = "medium"
                        else:
                            effect_label = "large"
                        st.write(f"This indicates a **{effect_label}** effect size.")

                        # Interpretation
                        alpha = 0.05
                        if p_val < alpha:
                            st.success(
                                f"✅ The change in **{test_col}** between the two conditions is **statistically significant** "
                                f"(p < {alpha}). This suggests a meaningful difference between datasets."
                            )
                        else:
                            st.info(
                                f"ℹ️ The change in **{test_col}** is **not statistically significant** "
                                f"(p = {p_val:.4f} ≥ {alpha}). There’s no clear evidence of a difference between conditions."
                            )


        elif stat_test == "Correlation":
            col1 = st.selectbox("Column 1", numeric_cols)
            col2 = st.selectbox("Column 2", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

            if col1 and col2:
                corr, p = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                st.write(f"**Correlation between {col1} and {col2}:** r = {corr:.4f}, p = {p:.4f}")
                fig = px.scatter(df, x=col1, y=col2, trendline="ols")
                st.plotly_chart(fig, use_container_width=True)

        elif stat_test == "Chi-square":
            cat_cols = df.select_dtypes(include='object').columns.tolist()
            if len(cat_cols) >= 2:
                cat1 = st.selectbox("Categorical Column 1", cat_cols)
                cat2 = st.selectbox("Categorical Column 2", cat_cols[::-1])

                if cat1 and cat2:
                    contingency = pd.crosstab(df[cat1], df[cat2])
                    chi2, p, dof, expected = stats.chi2_contingency(contingency)
                    st.write(f"Chi-square statistic: {chi2:.4f}, p-value: {p:.4f}")
                    st.write("Contingency Table:")
                    st.dataframe(contingency)

        # -------------------------
        # Qualitative Placeholder
        # -------------------------
        st.sidebar.header("5. (Optional) Qualitative Themes")
        st.subheader("Theme & Subtheme Visuals (Coming Soon)")
        st.markdown("This section could visualize coded qualitative data like quotes, themes, and subthemes using a tree diagram or card grid.")

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

        st.subheader("📁 Data Previews")
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
                    else:
                        st.warning("Paired T-test requires datasets of equal length (same participants).")
                        t_stat = p_val = None
                else:
                    t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
                    test_type = "Independent T-test"

                if t_stat is not None:
                    st.subheader(f"🧪 {test_type} Results")
                    st.write(f"**Test on '{test_col}'**")
                    st.write(f"t-statistic = {t_stat:.4f}")
                    st.write(f"p-value = {p_val:.4f}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean (Dataset 1)", f"{np.mean(data1):.2f}")
                        st.metric("Std Dev (Dataset 1)", f"{np.std(data1):.2f}")
                    with col2:
                        st.metric("Mean (Dataset 2)", f"{np.mean(data2):.2f}")
                        st.metric("Std Dev (Dataset 2)", f"{np.std(data2):.2f}")

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
