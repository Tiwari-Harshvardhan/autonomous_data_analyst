import os
import json
import uuid
import re
import requests
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr

from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from google.adk.agents import Agent


# ---------------------------------------------------------------------------
# Storage setup
# ---------------------------------------------------------------------------

BASE_STORAGE_DIR   = "storage"
FEATURED_DATA_DIR  = os.path.join(BASE_STORAGE_DIR, "eda_data")
PROFILE_DIR        = os.path.join(BASE_STORAGE_DIR, "profiles")
METADATA_DIR       = os.path.join(BASE_STORAGE_DIR, "metadata")
LOG_DIR            = os.path.join(BASE_STORAGE_DIR, "logs")

for d in (FEATURED_DATA_DIR, PROFILE_DIR, METADATA_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _unique_path(directory: str, prefix: str, extension: str) -> str:
    return os.path.join(directory, f"{prefix}_{uuid.uuid4()}.{extension}")


# ---------------------------------------------------------------------------
# Numeric string recovery (pre-pass safety net)
# ---------------------------------------------------------------------------

_FE_STRIP_RE = re.compile(
    r"[\$\£\€\₹\¥,%]"
    r"|(\s*(km\u00b2?|mi\u00b2?|sq\s*km|lbs?|kg|m\u00b2?|mph|kph|ft|in|cm|mm|ha|ac|oz|ml|l|gb|mb|tb))\b",
    re.I,
)


def _recover_numeric_strings_fe(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Safety-net numeric coercion for the Feature Engineering pipeline.

    Applies the same logic as the cleaning agent's recover_numeric_columns()
    but as an independent pre-pass, in case numeric-looking strings slipped
    through or the cleaning agent was skipped.

    Returns (df_with_coerced_cols, list_of_recovered_column_names).
    """
    recovered: List[str] = []
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().astype(str).str.strip()
        if sample.empty:
            continue
        cleaned = (
            sample
            .str.replace(_FE_STRIP_RE, "", regex=True)
            .str.replace(",", "", regex=False)
        )
        parsed = pd.to_numeric(cleaned, errors="coerce")
        if parsed.notna().mean() >= 0.60:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip()
                    .str.replace(_FE_STRIP_RE, "", regex=True)
                    .str.replace(",", "", regex=False),
                errors="coerce",
            )
            recovered.append(col)
    return df, recovered


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively converts NaN, Inf, tuples and numpy scalars to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(i) for i in obj]
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, np.number):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class FeatureEngineeringState(BaseModel):
    input_dataframe_path: str
    engineered_dataframe_path: Optional[str] = None
    profile_path: Optional[str] = None
    metadata_path: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    quality_report: Optional[Dict[str, Any]] = None
    analysis_summary: Optional[str] = None
    interaction_features_added: List[str] = Field(default_factory=list)
    datetime_features_added: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

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


def preserve_original_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    #store original numeric values before scaling
    for col in df.select_dtypes(include=[np.number]).columns:
        df[f"__original__{col}"] = df[col]
    return df

def feature_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score standardization for all numeric columns with non-zero std."""
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].std() != 0:
            df[f"{col}_zscore"] = (df[col] - df[col].mean()) / df[col].std()
    return df


def normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalization for numeric columns where max != min."""
    for col in df.select_dtypes(include=[np.number]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max != col_min:
            df[col] = (df[col] - col_min) / (col_max - col_min)
    return df


def correct_skewness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Box-Cox correction for columns with |skew| > 0.5.
    Positive skew: direct Box-Cox (requires all-positive values).
    Negative skew: reflect the column first, then apply Box-Cox.
    """
    df_out = df.copy()
    for col in df_out.select_dtypes(include=[np.number]).columns:
        skew = df_out[col].skew()
        try:
            if skew > 0.5 and (df_out[col] > 0).all():
                df_out[col] = stats.boxcox(df_out[col])[0]
            elif skew < -0.5:
                reflected = (df_out[col].max() + 1) - df_out[col]
                df_out[col] = stats.boxcox(reflected)[0]
        except Exception:
            pass  # leave column untouched if Box-Cox fails (e.g. constant values)
    return df_out


def expand_datetime_column(df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Expands a single parsed datetime column into up to 9 derived signals.
    Only adds 'hour' if the column actually contains sub-day time data.
    Returns the modified DataFrame and a list of new column names added.
    """
    new_cols: List[str] = []
    prefix = col

    # Guard: convert to datetime if not already, coercing bad values to NaT
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Drop the column if everything parsed to NaT — nothing useful to expand
    if df[col].isna().all():
        return df, new_cols

    # 1-3. day / month / year
    df[f"{prefix}_day"]   = df[col].dt.day
    df[f"{prefix}_month"] = df[col].dt.month
    df[f"{prefix}_year"]  = df[col].dt.year
    new_cols += [f"{prefix}_day", f"{prefix}_month", f"{prefix}_year"]

    # 4. day_of_week  (Monday=0 … Sunday=6)
    df[f"{prefix}_day_of_week"] = df[col].dt.dayofweek
    new_cols.append(f"{prefix}_day_of_week")

    # 5. is_weekend  (Saturday=5, Sunday=6)
    df[f"{prefix}_is_weekend"] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
    new_cols.append(f"{prefix}_is_weekend")

    # 6. is_month_start / is_month_end
    df[f"{prefix}_is_month_start"] = df[col].dt.is_month_start.astype(int)
    df[f"{prefix}_is_month_end"]   = df[col].dt.is_month_end.astype(int)
    new_cols += [f"{prefix}_is_month_start", f"{prefix}_is_month_end"]

    # 7. quarter  (1–4)
    df[f"{prefix}_quarter"] = df[col].dt.quarter
    new_cols.append(f"{prefix}_quarter")

    # 8. hour — only extract when the column carries genuine time data
    #    A column is time-bearing when at least one non-midnight hour exists.
    has_time = (df[col].dt.hour != 0).any()
    if has_time:
        df[f"{prefix}_hour"] = df[col].dt.hour
        new_cols.append(f"{prefix}_hour")

    # 9. days_since_min_date — continuous timeline tracker
    min_date = df[col].min()
    df[f"{prefix}_days_since_min"] = (df[col] - min_date).dt.days
    new_cols.append(f"{prefix}_days_since_min")

    return df, new_cols


def process_date_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Scans every non-numeric column, attempts datetime parsing, and runs
    expand_datetime_column on any that succeed.
    Returns the enriched DataFrame and a flat list of all new column names.
    """
    all_new_cols: List[str] = []

    for col in list(df.columns):
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df, new_cols = expand_datetime_column(df, col)
            all_new_cols.extend(new_cols)
            continue
        # Try parsing — only proceed if at least 50 % of values convert
        converted = pd.to_datetime(df[col], errors="coerce")
        non_null_pct = converted.notna().mean()
        if non_null_pct >= 0.5:
            df[col] = converted
            df, new_cols = expand_datetime_column(df, col)
            all_new_cols.extend(new_cols)

    return df, all_new_cols

# ---------------------------------------------------------------------------
# Step 1 — Semantic Intent Tagging & Metadata Payload
# ---------------------------------------------------------------------------

# Heuristic keyword maps used to assign broad domain tags to column names.
# These are intentionally simple — the LLM does the nuanced reasoning.
_DOMAIN_TAG_RULES: Dict[str, List[str]] = {
    "Currency":   ["price", "cost", "revenue", "sales", "amount", "fee",
                   "salary", "wage", "income", "spend", "budget", "total"],
    "Count":      ["count", "qty", "quantity", "num", "number", "units",
                   "frequency", "volume", "n_"],
    "Weight":     ["weight", "mass", "kg", "lb", "gram"],
    "Rate":       ["rate", "ratio", "pct", "percent", "proportion",
                   "share", "fraction"],
    "Score":      ["score", "rating", "rank", "grade", "index"],
    "Identifier": ["id", "uuid", "code", "key", "ref", "sku", "serial"],
    "Date":       ["date", "time", "year", "month", "day", "timestamp",
                   "created", "updated"],
    "Target":     ["target", "label", "churn", "default", "fraud",
                   "outcome", "result", "y"],
    "Geography":  ["city", "country", "region", "state", "zip", "lat",
                   "lon", "location"],
}


def _tag_column_domain(col_name: str) -> str:
    lower = col_name.lower()
    for tag, keywords in _DOMAIN_TAG_RULES.items():
        if any(kw in lower for kw in keywords):
            return tag
    return "Unknown"


def build_metadata_payload(
    df: pd.DataFrame,
    target_variable: Optional[str] = None,
    max_sample_values: int = 5,
) -> Dict[str, Any]:
    """
    Step 1: Packages per-column metadata into a dict that gets sent to the LLM.

    For each column we capture:
    - dtype string
    - domain tag (heuristic)
    - sample of up to `max_sample_values` representative values
    - basic stats for numeric columns (mean, std, min, max)
    - null rate
    """
    payload: Dict[str, Any] = {"columns": {}, "shape": list(df.shape)}

    for col in df.columns:
        is_target = (col == target_variable)
        domain = "Target" if is_target else _tag_column_domain(col)
        dtype_str = str(df[col].dtype)

        # Sample values — top frequent for categoricals, evenly spaced for numerics
        if pd.api.types.is_numeric_dtype(df[col]):
            sample_vals = df[col].dropna().quantile([0, 0.25, 0.5, 0.75, 1.0]).tolist()
            stats_block = {
                "mean": float(df[col].mean()),
                "std":  float(df[col].std()),
                "min":  float(df[col].min()),
                "max":  float(df[col].max()),
            }
        else:
            sample_vals = df[col].value_counts().head(max_sample_values).index.tolist()
            stats_block = {}

        payload["columns"][col] = {
            "dtype":       dtype_str,
            "domain_tag":  domain,
            "null_rate":   round(float(df[col].isna().mean()), 4),
            "sample_vals": [str(v) for v in sample_vals],
            "stats":       _sanitize_for_json(stats_block),
            "is_target":   is_target,
        }

    if target_variable:
        payload["target_variable"] = target_variable

    return payload


# ---------------------------------------------------------------------------
# Step 2 — LLM Prompt for Formula Generation
# ---------------------------------------------------------------------------

FEATURE_PROPOSAL_SYSTEM_PROMPT = """
You are a senior Feature Engineering specialist embedded inside an automated ML pipeline.

You will receive a JSON metadata payload describing the columns of a pandas DataFrame.
Each column entry contains its dtype, a domain tag (e.g. Currency, Count, Rate),
sample values, basic statistics, and a flag indicating whether it is the target variable.

Your task is to propose ONE new interaction feature that:
  1. Combines exactly two non-target columns mathematically.
  2. Is likely to have stronger predictive signal than either base column alone.
  3. Has a clear, domain-grounded business rationale.

Rules you MUST follow:
  - NEVER use the target variable in the formula.
  - Only use column names that appear in the metadata payload.
  - The formula must be valid executable pandas syntax using only these operators:
    +  -  *  /  **  (  )
    and these pandas functions: .abs()  .clip()  .log()  np.log1p()  np.sqrt()
  - Prefer ratios (A / B) and products (A * B) over sums — they often capture
    non-linear relationships that individual columns miss.
  - If two columns share a Currency + Count pairing (e.g. Total_Price and Quantity),
    propose a unit-price ratio. If there is a Rate + Count pairing, propose a
    weighted volume. Use domain logic.

Respond ONLY with a single JSON object — no markdown fences, no explanation outside JSON:
{
  "new_column_name": "<snake_case_name>",
  "formula": "<valid pandas expression using df['col_a'] and df['col_b']>",
  "col_a": "<first base column name>",
  "col_b": "<second base column name>",
  "business_rationale": "<one or two sentences explaining why this feature is useful>"
}
"""


def _call_gemini(prompt: str, metadata_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Sends the metadata payload to Gemini via the Anthropic-compatible messages
    endpoint and returns the parsed JSON proposal, or None on failure.

    The API key is read from the GEMINI_API_KEY environment variable.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    user_message = (
        "Here is the DataFrame metadata payload. Propose one interaction feature.\n\n"
        + json.dumps(metadata_payload, indent=2)
    )
    body = {
        "system_instruction": {"parts": [{"text": FEATURE_PROPOSAL_SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": user_message}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
    }

    resp = requests.post(url, json=body, timeout=30)
    resp.raise_for_status()

    raw_text = (
        resp.json()
        .get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
        .strip()
    )

    # Strip accidental markdown fences if the model misbehaves
    raw_text = re.sub(r"^```json\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    return json.loads(raw_text)


# ---------------------------------------------------------------------------
# Step 3 — Statistical Validation Safety Net
# ---------------------------------------------------------------------------

def validate_and_append_feature(
    df: pd.DataFrame,
    proposal: Dict[str, Any],
    target_variable: Optional[str] = None,
    min_correlation_gain: float = 0.02,
) -> Tuple[pd.DataFrame, bool, str]:
    """
    Validates an LLM-proposed feature before adding it to the DataFrame.

    Checks performed (in order):
      1. Data leakage — formula must not reference the target variable.
      2. Safe execution — formula is evaluated in a restricted namespace;
         divide-by-zero and NaN-dominated results are rejected.
      3. Variance check — a constant column (std == 0) is rejected.
      4. Correlation gain — if a target is provided, the new feature must
         correlate with the target more strongly than both base columns
         individually (by at least `min_correlation_gain`).

    Returns:
      (df, accepted: bool, reason: str)
    """
    new_col   = proposal.get("new_column_name", "")
    formula   = proposal.get("formula", "")
    col_a     = proposal.get("col_a", "")
    col_b     = proposal.get("col_b", "")

    if not new_col or not formula:
        return df, False, "Proposal missing new_column_name or formula."

    # --- Check 1: data leakage ---
    if target_variable and target_variable in formula:
        return df, False, (
            f"Data leakage rejected: formula references target '{target_variable}'."
        )

    # --- Check 2: safe execution ---
    allowed_namespace = {"df": df, "np": np}
    try:
        new_series = eval(formula, {"__builtins__": {}}, allowed_namespace)  # noqa: S307
    except Exception as e:
        return df, False, f"Formula execution failed: {e}"

    if not isinstance(new_series, pd.Series):
        return df, False, "Formula did not return a pandas Series."

    nan_rate = new_series.isna().mean()
    if nan_rate > 0.5:
        return df, False, f"Too many NaNs in new feature ({nan_rate:.1%}) — rejected."

    # --- Check 3: variance check ---
    if new_series.std() == 0:
        return df, False, "New feature is constant (zero variance) — rejected."

    # --- Check 4: correlation gain against target ---
    if target_variable and target_variable in df.columns:
        target = df[target_variable].dropna()

        def _corr(series: pd.Series) -> float:
            aligned = series.reindex(target.index).dropna()
            tgt_aligned = target.reindex(aligned.index)
            if len(aligned) < 10:
                return 0.0
            try:
                # Use Spearman for robustness against outliers
                r, _ = spearmanr(aligned, tgt_aligned)
                return abs(float(r))
            except Exception:
                return 0.0

        corr_new  = _corr(new_series)
        corr_a    = _corr(df[col_a]) if col_a in df.columns else 0.0
        corr_b    = _corr(df[col_b]) if col_b in df.columns else 0.0
        best_base = max(corr_a, corr_b)

        if corr_new < best_base + min_correlation_gain:
            return df, False, (
                f"Correlation gain insufficient: new={corr_new:.4f}, "
                f"best_base={best_base:.4f} (threshold +{min_correlation_gain})."
            )

    # --- All checks passed: append ---
    df[new_col] = new_series
    return df, True, f"Feature '{new_col}' validated and added."


# ---------------------------------------------------------------------------
# Feature interaction orchestrator
# ---------------------------------------------------------------------------

def run_feature_interaction(
    df: pd.DataFrame,
    logs: List[str],
    target_variable: Optional[str] = None,
    max_proposals: int = 3,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Runs the full Semantic Reasoner → LLM Proposal → Statistical Validation loop
    up to `max_proposals` times, each time re-building the metadata payload so
    the LLM sees features added by earlier iterations.

    Returns the enriched DataFrame and a list of added column names.
    """
    added_features: List[str] = []

    for attempt in range(1, max_proposals + 1):
        logs.append(f"[interaction] Attempt {attempt}/{max_proposals}")

        # Step 1 — build fresh metadata so the LLM sees already-added features
        payload = build_metadata_payload(df, target_variable=target_variable)

        # Step 2 — ask the LLM for a proposal
        try:
            proposal = _call_gemini(FEATURE_PROPOSAL_SYSTEM_PROMPT, payload)
        except Exception as e:
            logs.append(f"[interaction] LLM call failed: {e}")
            break

        if not proposal:
            logs.append("[interaction] LLM returned empty proposal — stopping.")
            break

        logs.append(
            f"[interaction] Proposal: '{proposal.get('new_column_name')}' "
            f"= {proposal.get('formula')}"
        )

        # Step 3 — validate and conditionally append
        df, accepted, reason = validate_and_append_feature(
            df, proposal, target_variable=target_variable
        )
        logs.append(f"[interaction] {'Accepted' if accepted else 'Rejected'}: {reason}")

        if accepted:
            added_features.append(proposal["new_column_name"])

    return df, added_features


# ===========================================================================
# Main pipeline
# ===========================================================================

def execute_feature_engineering(
    csv_path: str,
    target_variable: Optional[str] = None,
) -> FeatureEngineeringState:
    """
    Full feature engineering pipeline. Reads a cleaned CSV, applies all
    transforms, and writes the engineered CSV to FEATURED_DATA_DIR.

    The output path is stored in state.engineered_dataframe_path — this is
    the only value downstream agents (EDA, visualization, orchestrator) depend on.
    """
    state = FeatureEngineeringState(input_dataframe_path=csv_path)
    state.logs.append(f"Started feature engineering pipeline for: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")

    # Pre-pass: recover numeric values stored as strings
    df, recovered_cols = _recover_numeric_strings_fe(df)
    if recovered_cols:
        state.logs.append(
            f"Numeric string recovery (FE pre-pass): coerced columns {recovered_cols}"
        )
    else:
        state.logs.append("Numeric string recovery (FE pre-pass): no string-numeric columns found.")

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    state.logs.append(
        f"Loaded DataFrame: {df.shape[0]} rows × {df.shape[1]} cols. "
        f"Numeric columns: {numeric_cols}"
    )

    # Save initial profile
    profile_data = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
    }
    state.profile_path = save_profile(profile_data)

    # 1 — skewness correction (before scaling so scaling sees corrected values)
    pre_skew = {col: float(df[col].skew()) for col in numeric_cols}
    df = correct_skewness(df)
    post_skew = {col: float(df[col].skew()) for col in df.select_dtypes(include=[np.number]).columns}
    state.logs.append("Skewness correction applied.")

    # 2 — z-score feature scaling
    df = preserve_original_numeric_cols(df)
    df = feature_scaling(df)
    state.logs.append("Z-score scaling applied to numeric features.")

    # 3 — UPGRADE A: advanced datetime expansion
    df, datetime_cols = process_date_columns(df)
    state.datetime_features_added = datetime_cols
    state.logs.append(
        f"DateTime expansion complete. New columns: "
        f"{datetime_cols if datetime_cols else 'none'}"
    )

    # 4 — UPGRADE B: semantic feature interaction
    df, interaction_cols = run_feature_interaction(
        df, state.logs, target_variable=target_variable
    )
    state.interaction_features_added = interaction_cols
    state.logs.append(
        f"Feature interaction complete. Added: "
        f"{interaction_cols if interaction_cols else 'none'}"
    )

    # Save engineered DataFrame — downstream agents read this path
    output_path = _unique_path(FEATURED_DATA_DIR, "engineered_data", "csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    state.engineered_dataframe_path = output_path
    state.logs.append(
        f"Engineered DataFrame saved: {df.shape[0]} rows × {df.shape[1]} cols → {output_path}"
    )

    state.quality_report = {
        "pre_skewness":  _sanitize_for_json(pre_skew),
        "post_skewness": _sanitize_for_json(post_skew),
        "datetime_features_added": datetime_cols,
        "interaction_features_added": interaction_cols,
        "final_shape": list(df.shape),
    }

    state.analysis_summary = (
        f"Skewness corrected, z-score scaling applied. "
        f"DateTime expansion added {len(datetime_cols)} signals. "
        f"LLM-proposed interaction features accepted: {len(interaction_cols)}."
    )

    state.metadata_path = save_metadata(state)
    state.logs.append("Metadata saved.")

    return state


def run_feature_engineering_on_csv(
    csv_path: str,
    target_variable: Optional[str] = None,
) -> dict:
    """ADK tool wrapper — returns JSON-safe dict for the orchestrator."""
    state = execute_feature_engineering(csv_path, target_variable=target_variable)
    return _sanitize_for_json(state.model_dump())


# ---------------------------------------------------------------------------
# ADK agent
# ---------------------------------------------------------------------------

feature_engineering_agent = Agent(
    name="feature_engineering_agent",
    model="gemini-2.0-flash",
    description=(
        "Transforms cleaned tabular data through skewness correction, z-score "
        "scaling, advanced datetime expansion, and LLM-guided feature interaction "
        "with statistical validation."
    ),
    instruction="""
    You accept a CSV file path and an optional target_variable column name.

    Always call run_feature_engineering_on_csv to execute the full pipeline.

    After the pipeline completes, report:
    1. Which datetime columns were detected and how many signals were extracted.
    2. Which interaction features the LLM proposed, and how many passed validation.
    3. The final DataFrame shape and the path to the engineered CSV.
    4. Any columns where skewness correction was applied (visible in quality_report).

    Never apply min-max normalization after z-score scaling has already run.
    If the LLM feature proposal step fails due to a missing API key, log the
    error and continue — the pipeline must still complete without interaction features.
    """,
    tools=[run_feature_engineering_on_csv],
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "storage/dataframes/cleaned_example.csv"
    target   = sys.argv[2] if len(sys.argv) > 2 else None
    final    = execute_feature_engineering(csv_path, target_variable=target)
    print(final.model_dump_json(indent=2))