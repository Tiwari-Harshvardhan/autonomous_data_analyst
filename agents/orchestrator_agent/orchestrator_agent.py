import asyncio
import json
import os
import uuid
from typing import Any, Dict, List, Optional

from google.adk.agents import Agent
from pydantic import BaseModel, Field

from ..data_collection_agent.data_collection_agent import (
    WorkflowState as CollectionWorkflowState,
    execute_data_collection,
)
from ..data_collection_agent.data_collection_agent import scrape_url
from ..data_cleaning_agent.data_cleaning_agent import clean_dataframe_tool
from ..eda_agent.eda_agent import run_eda_on_csv
from ..extraction_agent.extraction_agent import execute_extraction
from ..feature_engineering_agent.feature_engineering_agent import run_feature_engineering_on_csv
from ..visualization_agent.visualization_agent import run_visualisation_pipeline

BASE_STORAGE_DIR = "storage"
PIPELINE_METADATA_DIR = os.path.join(BASE_STORAGE_DIR, "pipeline")
os.makedirs(PIPELINE_METADATA_DIR, exist_ok=True)


class OrchestratorState(BaseModel):
    user_query: str
    current_stage: str = "init"
    pipeline_plan: List[str] = Field(default_factory=list)
    raw_data_state: Optional[Dict[str, Any]] = None
    extraction_state: Optional[Dict[str, Any]] = None
    cleaning_state: Optional[Dict[str, Any]] = None
    feature_state: Optional[Dict[str, Any]] = None
    eda_state: Optional[Dict[str, Any]] = None
    visualization_state: Optional[Dict[str, Any]] = None
    logs: List[str] = Field(default_factory=list)
    artifact_paths: Dict[str, str] = Field(default_factory=dict)

    def add_log(self, message: str) -> None:
        self.logs.append(message)


def _read_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_html_paths_from_raw_data(raw_data_path: str) -> (List[str], Dict[str, str]):
    raw_records = _read_json_file(raw_data_path)
    html_paths: List[str] = []
    url_map: Dict[str, str] = {}

    for record in raw_records:
        html_path = record.get("html_path")
        url = record.get("url")
        if html_path:
            html_paths.append(html_path)
            if url:
                url_map[html_path] = url

    return html_paths, url_map


def _run_async(coro: Any) -> Any:
    return asyncio.run(coro)


def plan_data_pipeline(user_query: str, start_from_csv_path: Optional[str] = None) -> dict:
    """Returns a structured plan for the requested data workflow."""
    if start_from_csv_path:
        execution_plan = [
            "clean_data",
            "feature_engineering",
            "exploratory_data_analysis",
            "visualization",
        ]
    else:
        execution_plan = [
            "data_collection",
            "extraction",
            "clean_data",
            "feature_engineering",
            "exploratory_data_analysis",
            "visualization",
        ]

    return {
        "user_query": user_query,
        "start_from_csv_path": start_from_csv_path,
        "execution_plan": execution_plan,
        "agents": {
            "data_collection": "data_collection_agent",
            "extraction": "extraction_agent",
            "clean_data": "data_cleaning_agent",
            "feature_engineering": "feature_engineering_agent",
            "exploratory_data_analysis": "eda_agent",
            "visualization": "visualisation_agent",
        },
    }


def execute_data_collection_tool(user_query: str) -> dict:
    """Runs the data collection stage and returns serialized workflow state."""
    state = _run_async(execute_data_collection(CollectionWorkflowState(user_query=user_query)))
    return state.model_dump()


def execute_extraction_tool(raw_data_path: str) -> dict:
    """Runs the extraction stage using raw HTML paths produced by data collection."""
    html_paths, url_map = _extract_html_paths_from_raw_data(raw_data_path)
    if not html_paths:
        raise ValueError("No HTML paths were discovered in the data collection output.")

    state = execute_extraction(html_paths, url_map=url_map)
    return state.model_dump()


def orchestrate_full_pipeline(user_query: str, start_from_csv_path: Optional[str] = None) -> dict:
    """Runs the end-to-end pipeline and returns the combined orchestrator state."""
    state = OrchestratorState(user_query=user_query)
    state.pipeline_plan = plan_data_pipeline(user_query, start_from_csv_path)["execution_plan"]
    state.add_log("Created execution plan.")

    if start_from_csv_path:
        state.current_stage = "data_cleaning"
        state.add_log(f"Starting pipeline from supplied CSV: {start_from_csv_path}")
        cleaning_state = clean_dataframe_tool(start_from_csv_path)
        state.cleaning_state = cleaning_state
        cleaned_path = cleaning_state.get("cleaned_dataframe_path")
        state.artifact_paths["cleaned_dataframe_path"] = cleaned_path
        state.add_log(f"Data cleaning completed: {cleaned_path}")

        state.current_stage = "feature_engineering"
        feature_state = run_feature_engineering_on_csv(cleaned_path)
        state.feature_state = feature_state
        engineered_path = feature_state.get("engineered_dataframe_path")
        state.artifact_paths["engineered_dataframe_path"] = engineered_path
        state.add_log(f"Feature engineering completed: {engineered_path}")

    else:
        state.current_stage = "data_collection"
        state.add_log("Starting data collection stage.")
        raw_state = _run_async(execute_data_collection(CollectionWorkflowState(user_query=user_query)))
        state.raw_data_state = raw_state.model_dump()
        state.artifact_paths["raw_data_path"] = raw_state.raw_data_path
        state.add_log(f"Data collection completed: {raw_state.raw_data_path}")

        state.current_stage = "extraction"
        state.add_log("Starting extraction stage.")
        extraction_state = execute_extraction_tool(raw_state.raw_data_path)
        state.extraction_state = extraction_state
        extracted_csv_path = extraction_state.get("dataframe_path")
        state.artifact_paths["extracted_dataframe_path"] = extracted_csv_path
        state.add_log(f"Extraction completed: {extracted_csv_path}")

        state.current_stage = "data_cleaning"
        state.add_log("Starting data cleaning stage.")
        cleaning_state = clean_dataframe_tool(extracted_csv_path)
        state.cleaning_state = cleaning_state
        cleaned_path = cleaning_state.get("cleaned_dataframe_path")
        state.artifact_paths["cleaned_dataframe_path"] = cleaned_path
        state.add_log(f"Data cleaning completed: {cleaned_path}")

        state.current_stage = "feature_engineering"
        feature_state = run_feature_engineering_on_csv(cleaned_path)
        state.feature_state = feature_state
        engineered_path = feature_state.get("engineered_dataframe_path")
        state.artifact_paths["engineered_dataframe_path"] = engineered_path
        state.add_log(f"Feature engineering completed: {engineered_path}")

    state.current_stage = "exploratory_data_analysis"
    state.add_log("Starting exploratory data analysis stage.")
    eda_state = run_eda_on_csv(engineered_path)
    state.eda_state = eda_state
    state.artifact_paths["eda_metadata_path"] = eda_state.get("metadata_path")
    state.add_log(f"EDA completed: {eda_state.get('metadata_path')}")

    state.current_stage = "visualization"
    state.add_log("Starting visualization stage.")
    visualization_state = run_visualisation_pipeline(engineered_path)
    state.visualization_state = visualization_state
    state.artifact_paths["dashboard_html_path"] = visualization_state.get("dashboard_html_path")
    state.add_log(f"Visualization completed: {visualization_state.get('dashboard_html_path')}")

    state.current_stage = "finished"
    state.add_log("Full pipeline execution finished.")

    metadata_path = os.path.join(PIPELINE_METADATA_DIR, f"orchestrator_state_{os.path.basename(str(uuid.uuid4()))}.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(state.model_dump(), f, indent=2, ensure_ascii=False)
    state.add_log(f"Orchestration metadata saved: {metadata_path}")
    state.artifact_paths["orchestration_metadata_path"] = metadata_path

    return state.model_dump()


orchestrator_agent = Agent(
    model="gemini-2.5-flash",
    name="ml_orchestrator_agent",
    description="Master orchestration agent that coordinates collection, extraction, cleaning, engineering, EDA, and visualization workflows.",
    instruction="""
    You are the orchestrator agent. Your job is to evaluate user goals, create a structured execution plan, and delegate work to specialized agents.

    Use the following tools when orchestrating the pipeline:
    - execute_data_collection_tool
    - execute_extraction_tool
    - clean_dataframe_tool
    - run_feature_engineering_on_csv
    - run_eda_on_csv
    - run_visualisation_pipeline
    - orchestrate_full_pipeline
    - plan_data_pipeline

    If the user provides a CSV path, skip the collection and extraction stages.
    If not, run the full end-to-end pipeline from data collection through visualization.
    """,
    tools=[
        plan_data_pipeline,
        execute_data_collection_tool,
        execute_extraction_tool,
        clean_dataframe_tool,
        run_feature_engineering_on_csv,
        run_eda_on_csv,
        run_visualisation_pipeline,
        orchestrate_full_pipeline,
        scrape_url,
    ],
)


if __name__ == "__main__":
    result = orchestrate_full_pipeline(
        user_query="Collect, clean, analyze, and visualize web-based machine learning content",
        start_from_csv_path=None,
    )
    print(json.dumps(result, indent=2))

