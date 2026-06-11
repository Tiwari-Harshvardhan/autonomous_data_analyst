import os
import json
import re
import uuid
import requests
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


# ---------------------------------------------------------------------------
# Numeric string recovery
# ---------------------------------------------------------------------------

_NUMERIC_STRIP_RE = re.compile(
    r"[\$\£\€\₹\¥,%]"
    r"|(\s*(km\u00b2?|mi\u00b2?|sq\s*km|lbs?|kg|m\u00b2?|mph|kph|ft|in|cm|mm|ha|ac|oz|ml|l|gb|mb|tb))\b",
    re.I,
)
_COMMA_RE = re.compile(r",")


def recover_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects columns that look numeric but are stored as strings due to
    HTML text extraction (commas in numbers, currency symbols, unit suffixes,
    percentage signs, etc.) and coerces them to float.

    A column is coerced only when >=60% of its non-null values successfully
    parse after stripping the known non-numeric characters.

    Examples handled:
      '1,234,567'  -> 1234567.0
      '£51.77'     -> 51.77
      '45.3%'      -> 45.3
      '9,596 km²'  -> 9596.0
    """
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().astype(str).str.strip()
        if sample.empty:
            continue
        cleaned = sample.str.replace(_NUMERIC_STRIP_RE, "", regex=True)
        cleaned = cleaned.str.replace(",", "", regex=False)
        numeric = pd.to_numeric(cleaned, errors="coerce")
        success_rate = numeric.notna().mean()
        if success_rate >= 0.60:
            full_cleaned = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(_NUMERIC_STRIP_RE, "", regex=True)
                .str.replace(",", "", regex=False)
            )
            df[col] = pd.to_numeric(full_cleaned, errors="coerce")
    return df


# ---------------------------------------------------------------------------
# LLM outlier arbiter
# ---------------------------------------------------------------------------

_OUTLIER_ARBITER_PROMPT = """
You are a data quality specialist inside an automated data analysis pipeline.

A column named '{col}' in a dataset about '{context}' has outlier values detected
via the IQR method (1.5x rule).

Column statistics:
  - Mean   : {mean:.4g}
  - Std    : {std:.4g}
  - Min    : {min:.4g}
  - Max    : {max:.4g}
  - Q1     : {q1:.4g}
  - Q3     : {q3:.4g}
  - IQR lower fence : {lower:.4g}
  - IQR upper fence : {upper:.4g}

Sample outlier values: {outlier_sample}

Question: Does keeping these outliers affect the data analysis in a NEGATIVE manner?

Answer with ONLY a JSON object — no explanation, no markdown:
{{
  "keep_outliers": true | false,
  "reason": "one sentence"
}}

Rules:
- If the outliers represent REAL and MEANINGFUL extreme values (e.g., largest
  countries by area, most populous nations, richest companies, record-breaking
  athletes), answer keep_outliers=true.
- If the outliers are likely data entry errors, corruption, or unit-mismatch
  artifacts (e.g., a price of 999999 in a dataset where all other prices are
  $10-$50), answer keep_outliers=false.
"""


def _ask_gemini_keep_outlier(
    col: str,
    series: pd.Series,
    lower: float,
    upper: float,
    context: str = "unknown",
) -> bool:
    """
    Asks Gemini whether the IQR-detected outliers in `series` should be KEPT.
    Returns True  → keep outliers (do not remove).
    Returns False → remove outliers (apply IQR filter).
    Defaults to True on any API failure (safe fallback: preserve data).
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return True  # no key → keep by default

    outlier_mask = (series < lower) | (series > upper)
    outlier_vals = series[outlier_mask].dropna().tolist()[:10]
    if not outlier_vals:
        return True

    prompt = _OUTLIER_ARBITER_PROMPT.format(
        col=col,
        context=context,
        mean=float(series.mean()),
        std=float(series.std()),
        min=float(series.min()),
        max=float(series.max()),
        q1=float(series.quantile(0.25)),
        q3=float(series.quantile(0.75)),
        lower=lower,
        upper=upper,
        outlier_sample=outlier_vals,
    )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 256},
    }

    try:
        resp = requests.post(url, json=body, timeout=20)
        resp.raise_for_status()
        raw = (
            resp.json()
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        return bool(parsed.get("keep_outliers", True))
    except Exception:
        return True  # safe fallback: preserve data on any failure


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

def validate_numeric_ranges(
    df: pd.DataFrame,
    context: str = "unknown",
) -> pd.DataFrame:
    """
    IQR-based outlier handling with a Gemini LLM arbiter.

    For each numeric column the IQR fences are computed. If any values fall
    outside the fences, Gemini is asked:
      'Does keeping this outlier affect the data in a negative manner?'

    - If Gemini says keep → the column is left untouched.
    - If Gemini says remove → rows outside the fences are dropped.

    On any API failure the safe default is to KEEP outliers.
    Internal metadata columns (starting with '_') are always skipped.
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col.startswith("_"):  # preserve internal metadata
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        has_outliers = ((df[col] < lower) | (df[col] > upper)).any()
        if not has_outliers:
            continue

        # Ask Gemini whether these outliers should be removed
        keep = _ask_gemini_keep_outlier(
            col=col,
            series=df[col].dropna(),
            lower=lower,
            upper=upper,
            context=context,
        )

        if not keep:
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
def execute_cleaning_pipeline(
    dataframe_path: str,
    context: str = "unknown",
) -> CleaningState:
    """
    Full data cleaning pipeline.

    Args:
        dataframe_path: Path to the raw extracted CSV.
        context: Free-text description of the dataset (e.g. 'countries of the
                 world') passed to the LLM outlier arbiter so it can make a
                 semantically informed keep/remove decision.
    """
    state = CleaningState(input_dataframe_path=dataframe_path)
    state.logs.append("Loading dataframe")

    df = pd.read_csv(dataframe_path)
    state.logs.append(f"Loaded dataframe with shape {df.shape}")

    profile = profile_dataframe(df)
    state.profile_path = save_profile(profile)
    state.logs.append("Data profiling completed")

    # Recover numeric values stored as strings (commas, currency symbols, units)
    df = recover_numeric_columns(df)
    state.logs.append("Numeric string recovery applied.")

    df = handle_missing_values(df)
    state.logs.append("Missing values handled")

    df = remove_duplicates(df)
    state.logs.append("Duplicates removed")

    df = standardize_text_columns(df)
    state.logs.append("Text standardized")

    df = validate_numeric_ranges(df, context=context)
    state.logs.append("Numeric validation completed (LLM outlier arbiter applied)")

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

def clean_dataframe_tool(
    dataframe_path: str,
    context: str = "unknown",
) -> dict:
    """
    ADK tool wrapper.

    Args:
        dataframe_path: Path to the CSV produced by the extraction agent.
        context: Short description of the dataset used by the LLM outlier
                 arbiter (e.g. 'countries of the world with population and area').
    """
    result = execute_cleaning_pipeline(dataframe_path, context=context)
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
    3. Recover numeric columns that were extracted as strings (commas, currency
       symbols, percentage signs, units like km², lbs, etc.)
    4. Detect missing values
    5. Remove duplicates
    6. Standardize categorical values
    7. Validate numeric ranges using IQR — consult Gemini to decide whether
       detected outliers are meaningful extremes (keep) or data errors (remove)
    8. Save cleaned artifacts
    9. Generate quality reports

    Rules:

    - Never hallucinate values
    - Never silently remove important columns
    - Always preserve metadata
    - Always return artifact paths
    - When calling clean_dataframe_tool, pass a short `context` string
      describing what the dataset is about so the outlier arbiter can make
      an informed decision (e.g. 'countries of the world with population
      and land area data').
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


