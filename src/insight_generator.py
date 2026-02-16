import numpy as np

def generate_insights(data):
    insights = []

    numeric_data = data.select_dtypes(include=["number"])
    categorical_data = data.select_dtypes(include=["object"])

    # -------------------------------
    # 1️⃣ Missing Value Analysis
    # -------------------------------
    missing = data.isnull().sum()
    total_rows = data.shape[0]

    for col in missing.index:
        if missing[col] > 0:
            percent = (missing[col] / total_rows) * 100
            insights.append(
                f"Column '{col}' has {percent:.2f}% missing values."
            )

    # -------------------------------
    # 2️⃣ Numeric Analysis
    # -------------------------------
    for col in numeric_data.columns:
        mean_val = numeric_data[col].mean()
        std_val = numeric_data[col].std()
        min_val = numeric_data[col].min()
        max_val = numeric_data[col].max()
        skew_val = numeric_data[col].skew()

        insights.append(
            f"'{col}' ranges from {min_val:.2f} to {max_val:.2f}, with an average of {mean_val:.2f}."
        )

        # Variance check
        if std_val > mean_val * 0.5:
            insights.append(
                f"'{col}' shows high variability."
            )

        # Skewness detection
        if skew_val > 0.5:
            insights.append(
                f"'{col}' is positively skewed."
            )
        elif skew_val < -0.5:
            insights.append(
                f"'{col}' is negatively skewed."
            )

        # IQR-based Outlier Detection
        Q1 = numeric_data[col].quantile(0.25)
        Q3 = numeric_data[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = numeric_data[(numeric_data[col] < lower_bound) |
                                (numeric_data[col] > upper_bound)]

        if not outliers.empty:
            insights.append(
                f"Potential outliers detected in '{col}'."
            )

    # -------------------------------
    # 3️⃣ Strongest Correlation
    # -------------------------------
    if numeric_data.shape[1] > 1:
        corr_matrix = numeric_data.corr().abs()
        corr_matrix = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool))

        max_corr = corr_matrix.max().max()

        if max_corr and not np.isnan(max_corr):
            strongest_pair = np.where(corr_matrix == max_corr)
            col1 = numeric_data.columns[strongest_pair[0][0]]
            col2 = numeric_data.columns[strongest_pair[1][0]]

            insights.append(
                f"Strong relationship found between '{col1}' and '{col2}' (correlation = {max_corr:.2f})."
            )

    # -------------------------------
    # 4️⃣ Categorical Analysis
    # -------------------------------
    for col in categorical_data.columns:
        top_category = data[col].value_counts().idxmax()
        count = data[col].value_counts().max()

        insights.append(
            f"Most common category in '{col}' is '{top_category}' ({count} occurrences)."
        )

    # -------------------------------
    # 5️⃣ Overall Summary
    # -------------------------------
    insights.append(
        "Overall, the dataset shows clear numerical trends and categorical distribution patterns."
    )

    return insights
