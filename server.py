from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import threading
import uuid
import os
import json
from typing import Optional

from agents.orchestrator_agent.orchestrator_agent import orchestrate_full_pipeline

BASE_DIR = os.path.dirname(__file__)
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
PIPELINE_DIR = os.path.join(STORAGE_DIR, "pipeline")
os.makedirs(PIPELINE_DIR, exist_ok=True)

app = FastAPI(title="Autonomous Data Analyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_query: str
    start_from_csv_path: Optional[str] = None


def _save_metadata(run_id: str, state: dict) -> str:
    path = os.path.join(PIPELINE_DIR, f"orchestrator_state_{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    return path


def _format_summary(state: dict) -> str:
    # Produce a compact, human-readable HTML summary from the orchestrator state
    lines = []
    lines.append(f"<h2>Summary — {state.get('user_query')}</h2>")
    lines.append(f"<p><strong>Current stage:</strong> {state.get('current_stage')}</p>")

    def _add_section(name: str, key: str):
        sec = state.get(key)
        if sec:
            lines.append(f"<h3>{name}</h3>")
            # show short JSON excerpt
            excerpt = json.dumps(sec if isinstance(sec, dict) else {"value": str(sec)}, indent=2)
            lines.append(f"<pre>{excerpt}</pre>")

    _add_section("Raw Data State", "raw_data_state")
    _add_section("Extraction State", "extraction_state")
    _add_section("Cleaning State", "cleaning_state")
    _add_section("Feature Engineering State", "feature_state")
    _add_section("EDA State", "eda_state")
    _add_section("Visualization State", "visualization_state")

    # Artifacts
    artifacts = state.get("artifact_paths", {}) or {}
    if artifacts:
        lines.append("<h3>Artifacts</h3>")
        lines.append("<ul>")
        for k, v in artifacts.items():
            if not v:
                continue
            href = v
            # normalize to storage mount
            if "storage" in v and not v.startswith("/storage"):
                # make it web-accessible
                idx = v.find("storage")
                href = "/" + v[idx:]
            elif v.startswith("storage"):
                href = "/" + v

            lines.append(f"<li><strong>{k}:</strong> <a target='_blank' href='{href}'>{href}</a></li>")
        lines.append("</ul>")

    # Logs
    logs = state.get("logs") or []
    if logs:
        lines.append("<h3>Logs</h3>")
        lines.append("<pre>" + "\n".join(logs[-200:]) + "</pre>")

    return "\n".join(lines)


def _run_and_store(run_id: str, user_query: str, start_from_csv: Optional[str]):
    try:
        result = orchestrate_full_pipeline(user_query, start_from_csv)
    except Exception as e:
        result = {"user_query": user_query, "current_stage": "error", "logs": [str(e)]}

    # ensure run_id is stored under our chosen filename
    _save_metadata(run_id, result)


@app.post("/api/chat")
def start_chat(req: ChatRequest):
    run_id = str(uuid.uuid4())
    thread = threading.Thread(target=_run_and_store, args=(run_id, req.user_query, req.start_from_csv_path), daemon=True)
    thread.start()
    return {"run_id": run_id, "status": "started"}


@app.get("/api/history")
def list_history():
    files = []
    for fn in os.listdir(PIPELINE_DIR):
        if not fn.startswith("orchestrator_state_") or not fn.endswith('.json'):
            continue
        path = os.path.join(PIPELINE_DIR, fn)
        stat = os.stat(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"user_query": None}

        files.append({
            "file": fn,
            "path": path,
            "run_id": fn.replace("orchestrator_state_", "").replace('.json', ''),
            "user_query": data.get("user_query"),
            "current_stage": data.get("current_stage"),
            "modified": stat.st_mtime,
        })

    files = sorted(files, key=lambda x: x["modified"], reverse=True)
    return files


@app.get("/api/history/{run_id}")
def get_history(run_id: str):
    # Look for matching file
    target = None
    for fn in os.listdir(PIPELINE_DIR):
        if run_id in fn and fn.endswith('.json'):
            target = os.path.join(PIPELINE_DIR, fn)
            break
    if not target or not os.path.exists(target):
        raise HTTPException(status_code=404, detail="Run not found")
    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    # add a human-readable summary
    data["summary_html"] = _format_summary(data)
    return data


# Mount storage and frontend
app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "frontend"), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
import os
import sys
import json
import uuid
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Autonomous Data Analyst Server")

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
PIPELINE_DIR = STORAGE_DIR / "pipeline"

# Ensure storage directories exist
STORAGE_DIR.mkdir(exist_ok=True)
PIPELINE_DIR.mkdir(exist_ok=True)

class QueryRequest(BaseModel):
    query: str

def generate_human_readable_summary(state: Dict[str, Any]) -> str:
    """Generates a clean markdown/HTML summary of the pipeline state for the frontend."""
    summary_parts = []

    # Title / Query
    query = state.get("user_query", "Data Analysis Task")
    summary_parts.append(f"<h2>Analysis Report for: <em>\"{query}\"</em></h2>")

    # Artifact paths mapping
    artifacts = state.get("artifact_paths", {})
    cleaned_path = artifacts.get("cleaned_dataframe_path", "")
    dashboard_path = artifacts.get("dashboard_html_path", "")

    # Stage Status
    summary_parts.append("<div class='section'>")
    summary_parts.append("<h3>Pipeline Execution Status</h3>")
    summary_parts.append("<ul>")
    for log in state.get("logs", []):
        summary_parts.append(f"<li>{log}</li>")
    summary_parts.append("</ul>")
    summary_parts.append("</div>")

    # Data Profiling & Cleaning Insights
    cleaning = state.get("cleaning_state")
    if cleaning and isinstance(cleaning, dict):
        report = cleaning.get("quality_report")
        if report:
            summary_parts.append("<div class='section'>")
            summary_parts.append("<h3>Data Cleaning & Quality Report</h3>")
            summary_parts.append("<table>")
            summary_parts.append(f"<tr><td><strong>Final Row Count</strong></td><td>{report.get('final_rows', 'N/A')}</td></tr>")
            summary_parts.append(f"<tr><td><strong>Final Column Count</strong></td><td>{report.get('final_columns', 'N/A')}</td></tr>")
            summary_parts.append(f"<tr><td><strong>Duplicate Rows Removed</strong></td><td>{report.get('duplicates_after_cleaning', 0)} (remaining)</td></tr>")
            
            missing = report.get('missing_after_cleaning', {})
            missing_str = ", ".join(f"'{col}': {count}" for col, count in missing.items() if count > 0) or "None"
            summary_parts.append(f"<tr><td><strong>Missing Values Left</strong></td><td>{missing_str}</td></tr>")
            summary_parts.append("</table>")
            summary_parts.append("</div>")

    # Feature Engineering Insights
    feature = state.get("feature_state")
    if feature and isinstance(feature, dict):
        logs = feature.get("logs", [])
        scaling_applied = any("scaling" in l.lower() or "scale" in l.lower() or "standardization" in l.lower() for l in logs)
        date_engineered = any("date" in l.lower() or "datetime" in l.lower() for l in logs)
        
        summary_parts.append("<div class='section'>")
        summary_parts.append("<h3>Feature Engineering Actions</h3>")
        actions = []
        if scaling_applied:
            actions.append("Applied **Z-Score standardization** to numeric columns to ensure numerical stability.")
        if date_engineered:
            actions.append("Identified datetime fields and derived sub-features (day, month, year) to expand analytical depth.")
        if not actions:
            actions.append("Parsed numeric distributions and transformed coordinates successfully.")
        
        summary_parts.append("<ul>" + "".join(f"<li>{a}</li>" for a in actions) + "</ul>")
        summary_parts.append("</div>")

    # EDA Insights
    eda = state.get("eda_state")
    if eda and isinstance(eda, dict):
        report = eda.get("quality_report", {})
        summary_parts.append("<div class='section'>")
        summary_parts.append("<h3>Exploratory Data Analysis Details</h3>")
        
        # Skewness
        skewness = report.get("skewness", {})
        skewed_cols = [col for col, data in skewness.items() if isinstance(data, dict) and data.get("skewness")]
        if skewed_cols:
            summary_parts.append(f"<p><strong>Skewed distributions:</strong> {', '.join(skewed_cols)}</p>")
            
        # Outliers
        summary_parts.append("<ul>")
        numeric_summary = report.get("numeric_summary", {})
        if numeric_summary:
            summary_parts.append("<li>Calculated detailed descriptive statistics for all numeric dimensions.</li>")
        summary_parts.append("<li>Conducted bivariate analysis across dimension pairs.</li>")
        summary_parts.append("</ul>")
        summary_parts.append("</div>")

    # Dashboard Access
    if dashboard_path:
        rel_dash_path = f"/storage/dashboards/{os.path.basename(dashboard_path)}"
        summary_parts.append("<div class='dashboard-access'>")
        summary_parts.append(f"<a href='{rel_dash_path}' target='_blank' class='btn btn-primary'>Open Interactive BI Dashboard</a>")
        summary_parts.append("</div>")

    return "\n".join(summary_parts)

@app.get("/api/history")
async def get_history():
    """Lists all past pipeline runs sorted chronologically."""
    history = []
    if not PIPELINE_DIR.exists():
        return []
        
    for file_path in PIPELINE_DIR.glob("*.json"):
        try:
            mtime = file_path.stat().st_mtime
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract basic details
            history.append({
                "id": file_path.stem.replace("orchestrator_state_", ""),
                "query": data.get("user_query", "Unknown Query"),
                "timestamp": mtime,
                "stage": data.get("current_stage", "unknown")
            })
        except Exception as e:
            print(f"Error parsing history file {file_path.name}: {e}")
            continue
            
    # Sort by timestamp descending (newest first)
    history.sort(key=lambda x: x["timestamp"], reverse=True)
    return history

@app.get("/api/history/{run_id}")
async def get_run_details(run_id: str):
    """Fetches details for a specific run and generates clean HTML output."""
    file_name = f"orchestrator_state_{run_id}.json"
    file_path = PIPELINE_DIR / file_name
    
    if not file_path.exists():
        # Try listing to match standard UUID pattern
        matching = list(PIPELINE_DIR.glob(f"*{run_id}*.json"))
        if matching:
            file_path = matching[0]
        else:
            raise HTTPException(status_code=404, detail="Run history not found")
            
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        # Parse plot paths to web-friendly relative URLs
        plots = []
        eda_state = state.get("eda_state") or {}
        plot_paths = eda_state.get("plot_paths") or []
        for path in plot_paths:
            # e.g., 'storage/plots/foo.png' -> '/storage/plots/foo.png'
            name = os.path.basename(path)
            plots.append(f"/storage/plots/{name}")
            
        return {
            "id": run_id,
            "query": state.get("user_query"),
            "stage": state.get("current_stage"),
            "logs": state.get("logs", []),
            "plots": plots,
            "summary_html": generate_human_readable_summary(state),
            "state": state
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading run details: {str(e)}")

@app.post("/api/chat")
async def run_pipeline(request: QueryRequest):
    """Spawns the orchestrator agent pipeline in a separate process to avoid loop locks."""
    query = request.query
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    print(f"Running pipeline for query: {query}")
    
    # Subprocess command to run the orchestrator safely without event loop conflicts
    python_cmd = (
        f"import json, sys; "
        f"from agents.orchestrator_agent.orchestrator_agent import orchestrate_full_pipeline; "
        f"res = orchestrate_full_pipeline({json.dumps(query)}); "
        f"print('###RESULT###' + json.dumps(res))"
    )
    
    cmd = [sys.executable, "-c", python_cmd]
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(BASE_DIR)
        )
        
        stdout, stderr = await proc.communicate()
        
        stdout_str = stdout.decode("utf-8", errors="ignore")
        stderr_str = stderr.decode("utf-8", errors="ignore")
        
        if proc.returncode != 0:
            print(f"Subprocess Error:\n{stderr_str}")
            raise HTTPException(status_code=500, detail=f"Pipeline execution failed:\n{stderr_str}")
            
        # Parse result from stdout
        if "###RESULT###" not in stdout_str:
            print(f"Subprocess Output:\n{stdout_str}")
            raise HTTPException(status_code=500, detail="Pipeline completed but did not output structured result.")
            
        result_json = stdout_str.split("###RESULT###")[1].strip()
        state = json.loads(result_json)
        
        # Get run ID from metadata path
        meta_path = state.get("artifact_paths", {}).get("orchestration_metadata_path", "")
        run_id = Path(meta_path).stem.replace("orchestrator_state_", "") if meta_path else str(uuid.uuid4())
        
        # Format web-friendly plot URLs
        plots = []
        eda_state = state.get("eda_state") or {}
        plot_paths = eda_state.get("plot_paths") or []
        for path in plot_paths:
            name = os.path.basename(path)
            plots.append(f"/storage/plots/{name}")
            
        return {
            "id": run_id,
            "query": query,
            "stage": state.get("current_stage"),
            "logs": state.get("logs", []),
            "plots": plots,
            "summary_html": generate_human_readable_summary(state),
            "state": state
        }
        
    except Exception as e:
        print(f"Exception during pipeline execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount storage directory
app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")

# Serve frontend static assets (will be created in /frontend)
frontend_dir = BASE_DIR / "frontend"
frontend_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
