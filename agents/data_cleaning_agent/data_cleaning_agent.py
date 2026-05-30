import os
import json
import uuid
import pandas as pd
import numpy as np

from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from google.adk.agents import Agent

#storage setup
BASE_STORAGE_DIR = "storage"
CLEANED_DATA_DIR = os.path.join(BASE_STORAGE_DIR, "cleaned_data")
PROFILE_DIR = os.path.join(BASE_STORAGE_DIR,"profiles")
METADATA_DIR = os.path.join(BASE_STORAGE_DIR,"metadata")
LOG_DIR = os.path.join(BASE_STORAGE_DIR,"logs")

for d in (CLEANED_DATA_DIR,PROFILE_DIR,METADATA_DIR,LOG_DIR):
    os.makedirs(d, exist_ok=True)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively converts NaN, Inf, and other non-JSON-serializable values to None."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.number):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return float(obj)
    return obj


class CleaningState(BaseModel):
    input_dataframe_path: str
    cleaned_dataframe_path: Optional[str] = None
    profile_path: Optional[str] = None
    metadata_path: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    quality_report: Optional[Dict[str, Any]] = None


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:

    profile = {

        "rows": len(df),

        "columns": len(df.columns),

        "missing_values": (
            df.isnull().sum().to_dict()
        ),

        "duplicates": int(df.duplicated().sum()),

        "dtypes": (
            df.dtypes.astype(str).to_dict()
        ),

        "numeric_summary": (
            df.describe(include=[np.number])
            .to_dict()
            if not df.select_dtypes(
                include=[np.number]
            ).empty
            else {}
        )
    }

    return _sanitize_for_json(profile)



def save_profile(profile: Dict[str, Any]) -> str:

    path = os.path.join(PROFILE_DIR,f"profile_{uuid.uuid4()}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(profile),f,indent=4,ensure_ascii=False)
    return path


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype in ["int64","float64"]:
            mean_value = df[col].mean()
            std = df[col].std()
            if abs(mean_value) <= (3 * std):
                df[col] = df[col].fillna(mean_value)
        else:
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(
                    mode[0]
                )
    return df


#remove duplicates
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

#standardize text
def standardize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    text_columns = df.select_dtypes(include=["object"]).columns
    for col in text_columns:
        df[col] = (df[col].astype(str).str.strip().str.lower())
    return df

def validate_numeric_ranges(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

#save cleaned dataframe
def save_cleaned_dataframe(df: pd.DataFrame) -> str:
    path = os.path.join(CLEANED_DATA_DIR,f"cleaned_{uuid.uuid4()}.csv")
    df.to_csv(path,index=False,encoding="utf-8")
    return path

#save metadata
def save_metadata(state: CleaningState) -> str:
    path = os.path.join(METADATA_DIR,f"cleaning_metadata_{uuid.uuid4()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(state.model_dump()),f,indent=4,ensure_ascii=False)
    return path

#main cleaning pipeline
def execute_cleaning_pipeline(dataframe_path: str) -> CleaningState:
    state = CleaningState(input_dataframe_path=dataframe_path)
    state.logs.append("Loading dataframe")

    df = pd.read_csv(dataframe_path)
    state.logs.append(f"Loaded dataframe with shape {df.shape}")

    profile = profile_dataframe(df)
    state.profile_path = save_profile(profile)
    state.logs.append("Data profiling completed")

    df = handle_missing_values(df)
    state.logs.append("Missing values handled")

    df = remove_duplicates(df)
    state.logs.append("Duplicates removed")

    df = standardize_text_columns(df)
    state.logs.append("Text standardized")

    df = validate_numeric_ranges(df)
    state.logs.append("Numeric validation completed")

    cleaned_path = save_cleaned_dataframe(df)
    state.cleaned_dataframe_path = cleaned_path
    state.logs.append(f"Cleaned dataframe saved at {cleaned_path}")

    
    state.quality_report = {
        "final_rows": len(df),
        "final_columns": len(df.columns),
        "missing_after_cleaning": (df.isnull().sum().to_dict()),
        "duplicates_after_cleaning": int(df.duplicated().sum())
    }

    
#save metadata
    metadata_path = save_metadata(state)
    state.metadata_path = metadata_path
    state.logs.append(
        f"Metadata saved at {metadata_path}"
    )
    return state

def clean_dataframe_tool(dataframe_path: str) -> dict:
    result = execute_cleaning_pipeline(dataframe_path)
    return _sanitize_for_json(result.model_dump())

data_cleaning_agent = Agent(
    model="gemini-2.0-flash",
    name="data_cleaning_agent",
    description='Cleans, validates, profiles and standardizes extracted datasets before EDA and modelling',
    instruction="""
    You are a professional data cleaning agent.

    Responsibilities:

    1. Load extracted datasets
    2. Profile the dataset
    3. Detect missing values
    4. Remove duplicates
    5. Standardize categorical values
    6. Validate numeric ranges
    7. Save cleaned artifacts
    8. Generate quality reports

    Rules:

    - Never hallucinate values
    - Never silently remove important columns
    - Always preserve metadata
    - Always return artifact paths
    """,

    tools=[clean_dataframe_tool],
)

if __name__ == "__main__":
    sample_dataframe_path = (
        "storage/dataframes/extracted_sample.csv"
    )

    final_state = execute_cleaning_pipeline(
        sample_dataframe_path
    )

    print(
        final_state.model_dump_json(indent=4)
    )


