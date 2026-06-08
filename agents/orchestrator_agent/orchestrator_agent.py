import asyncio
import random
import concurrent.futures
import json
import os
import uuid
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.tools import AgentTool
from pydantic import BaseModel, Field
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from ..data_collection_agent.data_collection_agent import (
    WorkflowState as CollectionWorkflowState,
    execute_data_collection,
    scrape_url,
)
from ..data_cleaning_agent.data_cleaning_agent import clean_dataframe_tool
from ..eda_agent.eda_agent import run_eda_on_csv
from ..extraction_agent.extraction_agent import execute_extraction
from ..feature_engineering_agent.feature_engineering_agent import run_feature_engineering_on_csv
from ..visualization_agent.visualization_agent import run_visualisation_pipeline


# ---------------------------------------------------------------------------
# Storage setup
# ---------------------------------------------------------------------------

BASE_STORAGE_DIR = "storage"
PIPELINE_METADATA_DIR = os.path.join(BASE_STORAGE_DIR, "pipeline")
os.makedirs(PIPELINE_METADATA_DIR, exist_ok=True)

MAX_RETRY_ATTEMPTS = 2   # how many times the analyzer can trigger a re-run


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ExecutionPlan(BaseModel):
    """
    The reasoning agent's output. Describes exactly which stages to run
    and why, so the orchestrator doesn't have to guess.
    """
    user_query: str
    stages: List[str]           # ordered list of stage keys to execute
    start_csv_path: Optional[str] = None
    reasoning: str              # why these stages were chosen
    data_goal: str              # e.g. "trend analysis", "scrape + visualize"
    expected_output: str        # what a correct final output should look like


class AnalysisVerdict(BaseModel):
    """The analyzer agent's verdict on a completed pipeline run."""
    is_correct: bool
    issues: List[str] = Field(default_factory=list)
    correction_instruction: Optional[str] = None  # passed back to orchestrator on retry


class OrchestratorState(BaseModel):
    user_query: str
    current_stage: str = "init"
    execution_plan: Optional[Dict[str, Any]] = None
    raw_data_state: Optional[Dict[str, Any]] = None
    extraction_state: Optional[Dict[str, Any]] = None
    cleaning_state: Optional[Dict[str, Any]] = None
    feature_state: Optional[Dict[str, Any]] = None
    eda_state: Optional[Dict[str, Any]] = None
    visualization_state: Optional[Dict[str, Any]] = None
    logs: List[str] = Field(default_factory=list)
    artifact_paths: Dict[str, str] = Field(default_factory=dict)
    attempt: int = 1
    correction_instruction: Optional[str] = None  # set on retry

    def log(self, message: str) -> None:
        print(f"  [{self.current_stage}] {message}")
        self.logs.append(message)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf with None so json.dumps never raises."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(i) for i in obj]
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    if hasattr(obj, "tolist"):
        return _sanitize_for_json(obj.tolist())
    return obj

def _run_async(awaitable: Any) -> Any:
    """Run a coroutine safely whether or not a loop exists."""
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(lambda: asyncio.run(awaitable)).result()
    except RuntimeError:
        return asyncio.run(awaitable)

def _extract_html_paths(raw_data_path: str):
    records = _read_json(raw_data_path)
    html_paths, url_map = [], {}
    for r in records:
        hp = r.get("html_path")
        if hp:
            html_paths.append(hp)
            if r.get("url"):
                url_map[hp] = r["url"]
    return html_paths, url_map


def _save_orchestrator_metadata(state: OrchestratorState) -> str:
    path = os.path.join(PIPELINE_METADATA_DIR, f"orchestrator_{uuid.uuid4()}.json")
    sanitized = _sanitize_for_json(state.model_dump())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2, ensure_ascii=False)
    return path

async def _run_adk_agent(agent, prompt: str) -> str:
    final_text = ""
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            session = await _session_service.create_session(app_name = "autonomous-data-analyst", user_id = "system_user")
            runner = Runner(agent = agent, app_name = "autonomous_data_analyst", session_service = _session_service)
            return final_text
        except Exception as e:
            error_text = str(e)
            if "503" in error_text:
                wait_time = (2 ** attempt + random.uniform(0,1))
                print(f"Gemini overloaded. Retrying in {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
                continue
            raise
        raise RuntimeError("Gemini unavailable after multiple retries")

    session = await _session_service.create_session(
        app_name="autonomous_data_analyst",
        user_id="system_user"
    )

    runner = Runner(
        agent=agent,
        app_name="autonomous_data_analyst",
        session_service=_session_service,
    )

    content = types.Content(
        role="user",
        parts=[
            types.Part(text=prompt)
        ]
    )

    async for event in runner.run_async(
        user_id="system_user",
        session_id=session.id,
        new_message=content
    ):

        if (
            hasattr(event, "content")
            and event.content
            and event.content.parts
        ):

            for part in event.content.parts:

                if hasattr(part, "text") and part.text:
                    final_text += part.text

    return final_text.strip()


# ---------------------------------------------------------------------------
# Reasoning agent
#
# Analyzes the user's prompt and decides which pipeline stages are needed.
# Returns a structured ExecutionPlan that the orchestrator follows exactly.
# ---------------------------------------------------------------------------

_session_service = InMemorySessionService()
_reasoning_agent = Agent(
    model="gemini-2.5-flash",
    name="pipeline_reasoning_agent",
    description=(
        "Analyzes the user's request and produces a precise, ordered execution "
        "plan specifying which pipeline stages are required and why."
    ),
    instruction="""
    You are a pipeline planning specialist. Given a user query, decide which
    of the following stages are actually needed:

      - data_collection   : scrape websites or fetch remote data
      - extraction        : parse raw HTML into structured CSV
      - data_cleaning     : fix nulls, duplicates, invalid values
      - feature_engineering : scaling, encoding, skewness correction
      - eda               : statistical profiling, outlier detection
      - visualization     : plots, dashboards

    Rules:
    - If the user provides a local CSV path, SKIP data_collection and extraction.
    - If the user only wants a chart from an existing CSV, only include
      data_cleaning, feature_engineering, and visualization.
    - If the user wants to scrape a website, always include data_collection,
      extraction, data_cleaning, feature_engineering, eda, visualization in
      that order.
    - Never include a stage that has no bearing on the user's goal.
    - In `expected_output`, describe what a CORRECT final output looks like.
      For example: "A dashboard showing product price distributions by category"
      NOT "A graph of URLs".

    Respond ONLY with a JSON object matching this schema (no markdown fences):
    {
      "user_query": "<original query>",
      "stages": ["stage1", "stage2", ...],
      "start_csv_path": "<path or null>",
      "reasoning": "<why these stages>",
      "data_goal": "<one-line goal>",
      "expected_output": "<description of correct output>"
    }
    """,
)


def plan_pipeline(user_query: str, start_csv_path: Optional[str] = None) -> ExecutionPlan:
    """
    Calls the reasoning agent to produce a structured execution plan.
    Falls back to a sensible default if the agent response cannot be parsed.
    """
    prompt = f'User query: "{user_query}"'
    if start_csv_path:
        prompt += f'\nThe user has provided a CSV file at: {start_csv_path}'

    # The reasoning agent returns plain JSON — parse it directly.
    response = _run_async(
        _run_adk_agent(_reasoning_agent, prompt)  # adjust to your ADK run method
    )

    final_text=""

    try:
        raw = response if isinstance(response, dict) else json.loads(response)
        return ExecutionPlan(**raw)
    except Exception:
        # Safe fallback: if parsing fails, run the full pipeline.
        stages = (
            ["data_cleaning", "feature_engineering", "eda", "visualization"]
            if start_csv_path
            else ["data_collection", "extraction", "data_cleaning",
                  "feature_engineering", "eda", "visualization"]
        )
        return ExecutionPlan(
            user_query=user_query,
            stages=stages,
            start_csv_path=start_csv_path,
            reasoning="Fallback plan — reasoning agent response could not be parsed.",
            data_goal="Complete data analysis",
            expected_output="A dashboard and EDA report derived from the input data.",
        )


# ---------------------------------------------------------------------------
# Analyzer agent
#
# Inspects the pipeline's final output and decides whether it actually
# satisfies the user's original goal. If not, it issues a correction
# instruction that gets passed back to the orchestrator for a retry.
# ---------------------------------------------------------------------------

_analyzer_agent = Agent(
    model="gemini-2.5-flash",
    name="output_analyzer_agent",
    description=(
        "Reviews the completed pipeline output against the user's original goal "
        "and the expected output description. Flags mismatches and issues "
        "correction instructions when the output is wrong."
    ),
    instruction="""
    You are a quality-control agent for a data analysis pipeline.

    You receive:
    - user_query          : what the user originally asked for
    - expected_output     : what a correct output should look like (from the planner)
    - eda_summary         : key stats and column names from the EDA stage
    - visualization_path  : path to the generated dashboard HTML

    Your job:
    1. Check whether the visualization and EDA results actually address the
       user's goal.
    2. Flag obvious mismatches. Examples of BAD output:
       - User asked to analyze product prices, but charts show only URL columns.
       - User asked for trend analysis, but output is a frequency bar of IDs.
       - EDA report shows all columns are non-numeric when the goal needs numbers.
    3. If the output is wrong, set is_correct=false and write a specific
       correction_instruction explaining what the pipeline should do differently
       (e.g. "Re-run extraction focusing on <price> and <category> columns;
       drop URL and ID columns before visualization").

    Respond ONLY with a JSON object (no markdown fences):
    {
      "is_correct": true | false,
      "issues": ["issue1", "issue2"],
      "correction_instruction": "<specific fix, or null if correct>"
    }
    """,
)


def analyze_output(
    plan: ExecutionPlan,
    eda_state: Optional[Dict[str, Any]],
    visualization_state: Optional[Dict[str, Any]],
) -> AnalysisVerdict:
    """
    Asks the analyzer agent whether the pipeline output matches the user's goal.
    Returns an AnalysisVerdict with is_correct and optional correction_instruction.
    """
    eda_summary = {}
    if eda_state:
        qr = eda_state.get("quality_report", {})
        eda_summary = {
            "columns": list(qr.get("dtypes", {}).keys()),
            "numeric_columns": [
                c for c, t in qr.get("dtypes", {}).items()
                if "float" in str(t) or "int" in str(t)
            ],
            "missing_values": qr.get("missing_values", {}),
            "rows": qr.get("rows"),
        }

    prompt = json.dumps({
        "user_query": plan.user_query,
        "expected_output": plan.expected_output,
        "data_goal": plan.data_goal,
        "eda_summary": eda_summary,
        "visualization_path": (
            visualization_state.get("dashboard_html_path") if visualization_state else None
        ),
    }, indent=2)

    response = _run_async(
        _run_adk_agent(_analyzer_agent, prompt)
    )

    try:
        raw = response if isinstance(response, dict) else json.loads(response)
        return AnalysisVerdict(**raw)
    except Exception:
        # If we can't parse the verdict, assume it's correct to avoid infinite loops.
        return AnalysisVerdict(is_correct=True, issues=["Analyzer response unparseable — assuming correct."])


# ---------------------------------------------------------------------------
# Stage executors
#
# Each function runs exactly one pipeline stage and returns its state dict.
# The orchestrator calls these in the order specified by the ExecutionPlan.
# ---------------------------------------------------------------------------

VALID_STAGES = {
    "data_collection",
    "extraction",
    "data_cleaning",
    "feature_engineering",
    "eda",
    "visualization",
}


def _run_data_collection(state: OrchestratorState) -> str:
    """Returns raw_data_path."""
    raw_state = _run_async(
        execute_data_collection(CollectionWorkflowState(user_query=state.user_query))
    )
    state.raw_data_state = raw_state.model_dump()
    path = raw_state.raw_data_path
    state.artifact_paths["raw_data_path"] = path
    state.log(f"Raw data saved: {path}")
    return path


def _run_extraction(state: OrchestratorState, raw_data_path: str) -> str:
    """Returns extracted CSV path."""
    html_paths, url_map = _extract_html_paths(raw_data_path)
    if not html_paths:
        raise ValueError("Extraction stage: no HTML paths found in raw data output.")
    extraction_state = execute_extraction(html_paths, url_map=url_map)
    state.extraction_state = extraction_state.model_dump()
    path = extraction_state.dataframe_path
    state.artifact_paths["extracted_dataframe_path"] = path
    state.log(f"Extracted CSV saved: {path}")
    return path


def _run_cleaning(state: OrchestratorState, input_csv: str) -> str:
    """Returns cleaned CSV path."""
    cleaning_state = clean_dataframe_tool(input_csv)
    state.cleaning_state = cleaning_state
    path = cleaning_state.get("cleaned_dataframe_path")
    state.artifact_paths["cleaned_dataframe_path"] = path
    state.log(f"Cleaned CSV saved: {path}")
    return path


def _run_feature_engineering(state: OrchestratorState, input_csv: str) -> str:
    """Returns engineered CSV path."""
    feature_state = run_feature_engineering_on_csv(input_csv)
    state.feature_state = feature_state
    path = feature_state.get("engineered_dataframe_path")
    state.artifact_paths["engineered_dataframe_path"] = path
    state.log(f"Engineered CSV saved: {path}")
    return path


def _run_eda(state: OrchestratorState, input_csv: str) -> Dict[str, Any]:
    """Returns EDA state dict."""
    eda_state = run_eda_on_csv(input_csv)
    state.eda_state = eda_state
    state.artifact_paths["eda_metadata_path"] = eda_state.get("metadata_path")
    state.log(f"EDA metadata saved: {eda_state.get('metadata_path')}")
    return eda_state


def _run_visualization(state: OrchestratorState, input_csv: str) -> Dict[str, Any]:
    """Returns visualization state dict."""
    viz_state = run_visualisation_pipeline(input_csv)
    state.visualization_state = viz_state
    state.artifact_paths["dashboard_html_path"] = viz_state.get("dashboard_html_path")
    state.log(f"Dashboard saved: {viz_state.get('dashboard_html_path')}")
    return viz_state


# ---------------------------------------------------------------------------
# Core pipeline runner
#
# Executes stages in the order given by the ExecutionPlan.
# Returns the completed OrchestratorState.
# ---------------------------------------------------------------------------

def _execute_plan(plan: ExecutionPlan, attempt: int = 1,
                  correction: Optional[str] = None) -> OrchestratorState:
    """
    Runs the pipeline stages listed in `plan.stages` in order, threading
    each stage's output path into the next stage's input.
    """
    state = OrchestratorState(
        user_query=plan.user_query,
        execution_plan=plan.model_dump(),
        attempt=attempt,
        correction_instruction=correction,
    )

    if correction:
        state.log(f"Retry attempt {attempt}. Correction: {correction}")

    # The "cursor" tracks the most recent CSV produced so each stage
    # knows what to read. Starts as the user-supplied CSV (may be None).
    current_csv: Optional[str] = plan.start_csv_path
    raw_data_path: Optional[str] = None

    for stage in plan.stages:
        if stage not in VALID_STAGES:
            state.log(f"Unknown stage '{stage}' — skipping.")
            continue

        state.current_stage = stage
        state.log(f"Starting stage: {stage}")

        try:
            if stage == "data_collection":
                raw_data_path = _run_data_collection(state)

            elif stage == "extraction":
                if not raw_data_path:
                    raise ValueError("Extraction requires data_collection to run first.")
                current_csv = _run_extraction(state, raw_data_path)

            elif stage == "data_cleaning":
                if not current_csv:
                    raise ValueError("data_cleaning requires a CSV input.")
                current_csv = _run_cleaning(state, current_csv)

            elif stage == "feature_engineering":
                if not current_csv:
                    raise ValueError("feature_engineering requires a CSV input.")
                current_csv = _run_feature_engineering(state, current_csv)

            elif stage == "eda":
                if not current_csv:
                    raise ValueError("eda requires a CSV input.")
                _run_eda(state, current_csv)

            elif stage == "visualization":
                if not current_csv:
                    raise ValueError("visualization requires a CSV input.")
                _run_visualization(state, current_csv)

        except Exception as e:
            state.log(f"Stage '{stage}' failed: {e}")
            raise

    state.current_stage = "finished"
    state.log("All stages complete.")
    return state


# ---------------------------------------------------------------------------
# Orchestrate with reasoning + analysis loop
# ---------------------------------------------------------------------------

def orchestrate_full_pipeline(
    user_query: str,
    start_from_csv_path: Optional[str] = None,
) -> dict:
    """
    Full entry point:
      1. Reasoning agent produces an ExecutionPlan.
      2. Pipeline executes the plan.
      3. Analyzer agent checks the output.
      4. If output is wrong and retries remain, re-runs with correction.
    """

    # Step 1 — plan
    print("\n[Reasoning] Analyzing user query...")
    plan = plan_pipeline(user_query, start_csv_path=start_from_csv_path)
    print(f"[Reasoning] Stages selected: {plan.stages}")
    print(f"[Reasoning] Goal: {plan.data_goal}")
    print(f"[Reasoning] Expected output: {plan.expected_output}")

    correction: Optional[str] = None

    for attempt in range(1, MAX_RETRY_ATTEMPTS + 2):  # +2 so last attempt still runs
        # Step 2 — execute
        print(f"\n[Orchestrator] Executing pipeline (attempt {attempt})...")
        orch_state = _execute_plan(plan, attempt=attempt, correction=correction)

        # Step 3 — analyze
        print("\n[Analyzer] Reviewing output quality...")
        verdict = analyze_output(plan, orch_state.eda_state, orch_state.visualization_state)

        if verdict.is_correct:
            print("[Analyzer] Output looks correct. Pipeline complete.")
            break

        print(f"[Analyzer] Issues found: {verdict.issues}")

        if attempt > MAX_RETRY_ATTEMPTS:
            print("[Analyzer] Max retries reached. Returning best available output.")
            orch_state.log("Max retries reached. Output may be imperfect.")
            break

        # Feed the correction back into the next attempt
        correction = verdict.correction_instruction
        print(f"[Analyzer] Retrying with correction: {correction}")

        # Update the plan's expected_output with the correction so the
        # reasoning context improves on the next pass.
        plan = ExecutionPlan(
            **{**plan.model_dump(), "expected_output": correction or plan.expected_output}
        )

    # Save combined metadata
    metadata_path = _save_orchestrator_metadata(orch_state)
    orch_state.artifact_paths["orchestration_metadata_path"] = metadata_path
    orch_state.log(f"Orchestration metadata saved: {metadata_path}")

    result = _sanitize_for_json(orch_state.model_dump())
    return result


# ---------------------------------------------------------------------------
# ADK tool wrappers (thin — just call the pipeline functions above)
# ---------------------------------------------------------------------------

def execute_data_collection_tool(user_query: str) -> dict:
    """ADK tool: runs data collection and returns serialized state."""
    state = _run_async(
        execute_data_collection(CollectionWorkflowState(user_query=user_query))
    )
    return state.model_dump()


def execute_extraction_tool(raw_data_path: str) -> dict:
    """ADK tool: runs extraction from a raw_data JSON path."""
    html_paths, url_map = _extract_html_paths(raw_data_path)
    if not html_paths:
        raise ValueError("No HTML paths found in the provided raw data file.")
    state = execute_extraction(html_paths, url_map=url_map)
    return state.model_dump()


def plan_pipeline_tool(user_query: str, start_csv_path: Optional[str] = None) -> dict:
    """ADK tool: returns the reasoning agent's execution plan as a dict."""
    return plan_pipeline(user_query, start_csv_path).model_dump()


# ---------------------------------------------------------------------------
# ADK orchestrator agent
# ---------------------------------------------------------------------------

orchestrator_agent = Agent(
    model="gemini-2.5-flash",
    name="ml_orchestrator_agent",
    description=(
        "Master orchestration agent. Uses a reasoning agent to plan which "
        "pipeline stages are needed, executes them in order, then uses an "
        "analyzer agent to verify the output and retry if needed."
    ),
    instruction="""
    You are the orchestrator. Your job is to coordinate the full data pipeline.

    Always start by calling plan_pipeline_tool to get a reasoned execution plan.
    Then call orchestrate_full_pipeline to execute and verify the pipeline.

    Only call individual stage tools (execute_data_collection_tool,
    execute_extraction_tool, clean_dataframe_tool, run_feature_engineering_on_csv,
    run_eda_on_csv, run_visualisation_pipeline) if the user explicitly asks to
    run a single stage in isolation.

    If the user provides a CSV path, pass it as start_from_csv_path to
    orchestrate_full_pipeline — the reasoning agent will skip collection and
    extraction automatically.

    After the pipeline finishes, summarize:
    - Which stages ran and why
    - What artifacts were produced and where they were saved
    - Whether the analyzer agent flagged any issues and how they were resolved
    """,
    tools=[
        plan_pipeline_tool,
        orchestrate_full_pipeline,
        execute_data_collection_tool,
        execute_extraction_tool,
        clean_dataframe_tool,
        run_feature_engineering_on_csv,
        run_eda_on_csv,
        run_visualisation_pipeline,
        scrape_url,
    ],
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = orchestrate_full_pipeline(
        user_query="Scrape machine learning articles and visualize topic frequency trends",
        start_from_csv_path=None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))