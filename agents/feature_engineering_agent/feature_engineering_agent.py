import os
import json
import uuid
import pandas as pd
import numpy as np
from scipy import stats

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from google.adk.agents import Agent

BASE_STORAGE_DIR = "storage"
FEATURED_DATA_DIR = os.path.join(BASE_STORAGE_DIR, "eda_data")
PROFILE_DIR = os.path.join(BASE_STORAGE_DIR, "profiles")
METADATA_DIR = os.path.join(BASE_STORAGE_DIR, "metadata")
LOG_DIR = os.path.join(BASE_STORAGE_DIR, "logs")

for d in (FEATURED_DATA_DIR, PROFILE_DIR, METADATA_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

def _sanitize_for_json(obj: Any) -> Any:
    """Recursively converts NaN, Inf, tuples and other non-JSON-serializable values to JSON-safe values."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.number):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)

class FeatureEngineeringState(BaseModel):
    input_dataframe_path: str
    engineered_dataframe_path: Optional[str] = None
    profile_path: Optional[str] = None
    metadata_path: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    quality_report: Optional[Dict[str, Any]] = None

def _unique_path(directory: str, prefix: str, extension: str) -> str:
    return os.path.join(directory, f"{prefix}_{uuid.uuid4()}.{extension}")

def save_profile(profile: Dict[str, Any]) -> str:
    path = _unique_path(PROFILE_DIR, "profile", "json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(profile), f, indent=2, ensure_ascii=False)
    return path

def save_metadata(state: FeatureEngineeringState) -> str:
    path = _unique_path(METADATA_DIR, "feature_engineering_metadata", "json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(state.model_dump()), f, indent=2, ensure_ascii=False)
    return path

def feature_scaling(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        if df[col].std() != 0:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def normalization(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max != col_min:
            df[col] = (df[col] - col_min) / (col_max - col_min)
    return df

def correct_skewness(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        skew = df_clean[col].skew()
        if skew > 0.5:
            if (df_clean[col] > 0).all():
                df_clean[col] = stats.boxcox(df_clean[col])[0]
        elif skew < -0.5:
            max_val = df_clean[col].max()
            reflected_data = (max_val + 1) - df_clean[col]
            df_clean[col] = stats.boxcox(reflected_data)[0]
    return df_clean

def process_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            continue
        converted = pd.to_datetime(df_clean[col], errors='coerce')
        if pd.api.types.is_datetime64_any_dtype(converted):
            df_clean[col] = converted
            df_clean[f'{col}_day'] = df_clean[col].dt.day
            df_clean[f'{col}_month'] = df_clean[col].dt.month
            df_clean[f'{col}_year'] = df_clean[col].dt.year
    return df_clean

def execute_feature_engineering(csv_path: str) -> FeatureEngineeringState:
    state = FeatureEngineeringState(input_dataframe_path=csv_path)
    state.logs.append(f"[INIT] Started feature engineering pipeline for: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")
    initial_shape = df.shape
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    state.logs.append(f"[LOAD] DataFrame loaded successfully. Shape: {initial_shape[0]} rows x {initial_shape[1]} cols. Numeric features detected: {numeric_cols}")

    profile_data = {
        "shape": list(initial_shape),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict()
    }
    state.profile_path = save_profile(profile_data)
    state.logs.append(f"[PROFILE] Initial data structure profiled and written to disk: {state.profile_path}")

    # Capture original skew metrics for deep logging
    pre_skew = {col: float(df[col].skew()) for col in numeric_cols}
    df = correct_skewness(df)
    post_skew = {col: float(df[col].skew()) for col in numeric_cols}
    state.logs.append(f"[SKEW] Variance correction applied. Pre-skewness: {pre_skew} -> Post-skewness: {post_skew}")

    df = feature_scaling(df)
    state.logs.append(f"[SCALE] Standardization (Z-Score) applied to numerical fields: {numeric_cols}")

    initial_cols = set(df.columns)
    df = process_date_columns(df)
    new_cols = list(set(df.columns) - initial_cols)
    state.logs.append(f"[DATETIME] Evaluated feature types. Generated engineering sub-components: {new_cols if new_cols else 'None discovered'}")

    output_csv_path = _unique_path(FEATURED_DATA_DIR, "engineered_data", "csv")
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    state.engineered_dataframe_path = output_csv_path
    state.logs.append(f"[PERSIST] Transformed dataset saved. Final distribution shape: {df.shape[0]} rows x {df.shape[1]} cols. Destination path: {output_csv_path}")

    state.metadata_path = save_metadata(state)
    state.logs.append(f"[METADATA] Run trace and logs cataloged successfully at: {state.metadata_path}")

    print(f"Feature engineering complete. Artifacts saved to '{BASE_STORAGE_DIR}'")
    return state

def run_feature_engineering_on_csv(csv_path: str) -> dict:
    state = execute_feature_engineering(csv_path)
    return _sanitize_for_json(state.model_dump())

feature_engineering_agent = Agent(
    name="feature_engineering_agent",
    model="gemini-2.0-flash",
    description="Automated system targeting data transformation, mathematical scaling, and feature derivation workflows.",
    instruction="""
    You accept tabular datasets (.csv paths) to standardise scale discrepancies and resolve statistical skewness.
    Always execute the pipeline function `run_feature_engineering_on_csv` when passed file paths to automate cleaning steps.
    Review runtime logging details returned from state tracking variables to verify if date transformations or skew changes succeeded.
    Communicate final processing outputs directly to users by summarizing the storage paths for profiles and output CSV sheets.
    Do not apply subsequent min-max normalizations on metrics that have already passed standardization bounds.
    """,
    tools=[run_feature_engineering_on_csv]
)

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv)>1 else "storage/dataframes/cleaned_example.csv"
    final_state = execute_feature_engineering(csv_path)
    print(final_state.model_dump_json(indent=2))
