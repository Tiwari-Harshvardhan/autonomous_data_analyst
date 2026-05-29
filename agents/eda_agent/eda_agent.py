import os
import json
import uuid
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from google.adk.agents import Agent


BASE_STORAGE_DIR = "storage"
EDA_DATA_DIR  = os.path.join(BASE_STORAGE_DIR, "eda_data")
PROFILE_DIR   = os.path.join(BASE_STORAGE_DIR, "profiles")
METADATA_DIR  = os.path.join(BASE_STORAGE_DIR, "metadata")
PLOT_DIR      = os.path.join(BASE_STORAGE_DIR, "plots")
LOG_DIR       = os.path.join(BASE_STORAGE_DIR, "logs")

for d in (EDA_DATA_DIR, PROFILE_DIR, METADATA_DIR, PLOT_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)


class EDAState(BaseModel):
    input_dataframe_path: str
    profile_path: Optional[str] = None
    plot_paths: List[str] = Field(default_factory=list)
    metadata_path: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    quality_report: Optional[Dict[str, Any]] = None


def _unique_path(directory: str, prefix: str, extension: str) -> str:
    return os.path.join(directory, f"{prefix}_{uuid.uuid4()}.{extension}")


def save_profile(profile: Dict[str, Any]) -> str:
    path = _unique_path(PROFILE_DIR, "profile", "json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    return path


def save_metadata(state: EDAState) -> str:
    path = _unique_path(METADATA_DIR, "eda_metadata", "json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.model_dump(), f, indent=2, ensure_ascii=False)
    return path


def _save_figure(prefix: str) -> str:
    """Saves the current matplotlib figure to disk and closes it."""
    path = _unique_path(PLOT_DIR, prefix, "png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    return path



def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns a data quality snapshot: shape, missing values, duplicates,
    dtypes, and numeric summary statistics.
    """
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_summary": (
            df.describe(include=[np.number]).to_dict()
            if not df.select_dtypes(include=[np.number]).empty
            else {}
        ),
    }


def descriptive_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns describe() results for numeric and categorical columns separately.
    """
    result: Dict[str, Any] = {}

    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        result["numeric"] = numeric.describe().to_dict()

    categorical = df.select_dtypes(include=["object", "category"])
    if not categorical.empty:
        result["categorical"] = categorical.describe().to_dict()

    return result


def visualisation_tool(df: pd.DataFrame) -> List[str]:
    """
    Plots a KDE (numeric columns) or bar chart (categorical columns) for each
    column and saves each figure to disk. Returns a list of saved file paths.
    """
    saved_paths = []

    for col in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_data = df[col].dropna()
            if numeric_data.empty:
                plt.close(fig)
                continue
            numeric_data.plot(kind="kde", ax=ax, title=f"Distribution — {col}")
            ax.set_xlabel(col)
        else:
            counts = df[col].value_counts().head(20)
            if counts.empty:
                plt.close(fig)
                continue
            counts.plot(kind="bar", ax=ax, title=f"Value counts — {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")

        path = _save_figure(f"plot_{col}")
        saved_paths.append(path)

    return saved_paths


def outlier_detector(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Uses the IQR method to identify outliers in every numeric column.
    Returns a dict mapping column name → list of outlier values.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {"error": "No numeric columns found"}

    report: Dict[str, Any] = {}
    for col in numeric_df.columns:
        q1 = numeric_df[col].quantile(0.25)
        q3 = numeric_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr 

        mask = (numeric_df[col] < lower) | (numeric_df[col] > upper)  # fixed missing parens
        outliers = numeric_df.loc[mask, col]
        report[col] = {
            "count": int(outliers.count()),
            "values": outliers.tolist(),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
        }

    return report


def skewness_detector(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns skewness for every numeric column where |skew| > 0.5,
    along with a human-readable label (positive / negative skew).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {"error": "No numeric columns found"}

    report: Dict[str, Any] = {}
    for col in numeric_df.columns:
        skew = numeric_df[col].skew()
        if abs(skew) > 0.5:
            report[col] = {
                "skewness": round(skew, 4),
                "direction": "positive" if skew > 0 else "negative",
            }

    return report


def categorical_column_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    For each categorical column, returns value frequencies and the number
    of unique values.
    """
    categorical_df = df.select_dtypes(include=["object", "category"])
    if categorical_df.empty:
        return {"error": "No categorical columns found"}

    report: Dict[str, Any] = {}
    for col in categorical_df.columns:
        report[col] = {
            "unique_count": int(df[col].nunique()),
            "frequencies": df[col].value_counts().to_dict(), 
        }
    return report


def categorical_missing(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns null counts and null percentages for all categorical columns.
    """
    categorical_df = df.select_dtypes(include=["object", "category"])
    if categorical_df.empty:
        return {"error": "No categorical columns found"}

    report: Dict[str, Any] = {}
    for col in categorical_df.columns:
        null_count = int(df[col].isnull().sum())
        report[col] = {
            "null_count": null_count,
            "null_pct": round(null_count / len(df) * 100, 2),
        }
    return report


def bivariate_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    "Runs pairwise analysis across all column combinations"
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    all_cols = num_cols + cat_cols

    report: Dict[str, Any] = {}

    pairs = [(all_cols[i], all_cols[j])
             for i in range(len(all_cols))
             for j in range(i + 1, len(all_cols))]

    for col1, col2 in pairs: 
        key = f"{col1} , {col2}"

        if col1 in num_cols and col2 in num_cols:
            corr = df[col1].corr(df[col2])
            report[key] = {"type": "num-num", "pearson_r": round(corr, 4)}

        elif col1 in num_cols and col2 in cat_cols:
            # mean of numeric column grouped by the categorical column
            group_means = df.groupby(col2)[col1].mean().round(4).to_dict()
            report[key] = {"type": "num-cat", "group_means": group_means}

        elif col1 in cat_cols and col2 in num_cols:
            group_means = df.groupby(col1)[col2].mean().round(4).to_dict()
            report[key] = {"type": "cat-num", "group_means": group_means}

        else:
            ct = pd.crosstab(df[col1], df[col2]).to_dict()
            report[key] = {"type": "cat-cat", "contingency_table": ct}

    return report


def execute_eda(csv_path: str) -> EDAState:
    """
    Loads a cleaned CSV, runs the full EDA suite, saves all artifacts to
    disk, and returns the populated EDAState.
    """
    state = EDAState(input_dataframe_path=csv_path)
    state.logs.append(f"Loading data from: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")
    state.logs.append(f"Loaded DataFrame: {df.shape[0]} rows x {df.shape[1]} cols")

    # profile
    profile = profile_dataframe(df)
    profile['skewness'] = skewness_detector(df)
    state.profile_path = save_profile(profile)
    state.quality_report = profile
    state.logs.append(f"Profile saved: {state.profile_path}")

    # descriptive stats 
    stats = descriptive_statistics(df)
    state.logs.append(f"Descriptive stats computed for {len(stats)} column groups")

    # visualisations
    plot_paths = visualisation_tool(df)
    state.plot_paths = plot_paths
    state.logs.append(f"Saved {len(plot_paths)} plot(s) to {PLOT_DIR}")

    # outliers
    outliers = outlier_detector(df)
    state.logs.append(f"Outlier detection complete: {len(outliers)} column(s) checked")

    # skewness
    skew = skewness_detector(df)
    state.logs.append(f"Skewed columns (|skew|>0.5): {list(skew.keys())}")

    # categorical analysis
    cat_analysis = categorical_column_analysis(df)
    cat_nulls = categorical_missing(df)
    state.logs.append("Categorical analysis complete")

    # bivariate
    bivariate = bivariate_analysis(df)
    state.logs.append(f"Bivariate analysis complete: {len(bivariate)} pair(s)")

    # Persist full metadata
    state.metadata_path = save_metadata(state)
    state.logs.append(f"Metadata saved: {state.metadata_path}")

    print(f"\nEDA complete. Artifacts saved to '{BASE_STORAGE_DIR}/'")
    print(f"  Profile  : {state.profile_path}")
    print(f"  Plots    : {len(state.plot_paths)} file(s) in {PLOT_DIR}")
    print(f"  Metadata : {state.metadata_path}")

    return state


def run_eda_on_csv(csv_path: str) -> dict:
    """ADK-compatible tool. Runs the full EDA pipeline and returns the state."""
    state = execute_eda(csv_path)
    return state.model_dump()


eda_agent = Agent(
    name="eda_agent",
    model="gemini-2.0-flash",
    description=(
        "Exploratory data analysis agent. Given a cleaned CSV file, it profiles "
        "the data, detects outliers and skewness, analyses categorical columns, "
        "runs bivariate analysis, generates visualisation plots, and saves all "
        "artifacts to disk."
    ),
    instruction="""
    You are an expert EDA agent. When given a CSV file path:

    1. Call run_eda_on_csv with that path to execute the full EDA pipeline.
    2. Summarize the quality report: row/column counts, missing value rates,
       duplicate rows, and column dtypes.
    3. Highlight any columns with significant outliers (IQR method).
    4. Report skewed columns and their direction (positive/negative).
    5. Summarize categorical column frequencies and missing rates.
    6. Report the strongest bivariate relationships found (high Pearson r,
       notable group mean differences, or skewed contingency tables).
    7. Confirm paths where all artifacts were saved (profile, plots, metadata).

    Rules:
    - Never fabricate statistics or invent data.
    - If a step fails, log the error and continue with the remaining steps.
    - Always end with a plain-language summary a non-technical stakeholder
      could understand.
    """,
    tools=[run_eda_on_csv],
)

# Entry point
if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "storage/dataframes/cleaned_example.csv"
    final_state = execute_eda(csv_path)
    print(final_state.model_dump_json(indent=2))