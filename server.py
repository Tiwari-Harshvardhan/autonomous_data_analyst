import os
import sys
import json
import uuid
import asyncio
import subprocess
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts NaN and Inf to null."""
    def encode(self, o):
        if isinstance(o, float):
            if np.isnan(o) or np.isinf(o):
                return 'null'
        return super().encode(o)
    
    def iterencode(self, o, _one_shot=False):
        """Override iterencode to handle NaN/Inf values."""
        for chunk in super().iterencode(o, _one_shot):
            yield chunk


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


app = FastAPI(title="Autonomous Data Analyst Server")

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
PIPELINE_DIR = STORAGE_DIR / "pipeline"

STORAGE_DIR.mkdir(exist_ok=True)
PIPELINE_DIR.mkdir(exist_ok=True)

class QueryRequest(BaseModel):
    query: str


def escape_html(value: str) -> str:
    return (
        str(value)
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&#039;')
    )


def generate_human_readable_summary(state: Dict[str, Any]) -> str:
    summary_parts = []
    query = state.get("user_query", "Data Analysis Task")
    summary_parts.append(f"<h2>Analysis Report for: <em>\"{escape_html(query)}\"</em></h2>")

    summary_parts.append("<div class='section'>")
    summary_parts.append("<h3>Key Findings</h3>")

    eda = state.get("eda_state")
    if eda and isinstance(eda, dict):
        if eda.get("analysis_summary"):
            summary_parts.append(f"<p>{escape_html(eda['analysis_summary'])}</p>")
        else:
            summary_parts.append("<p>Exploratory analysis generated a structured quality profile and measured the most important numeric and categorical patterns.</p>")

    feature = state.get("feature_state")
    if feature and isinstance(feature, dict) and feature.get("analysis_summary"):
        summary_parts.append(f"<p>{escape_html(feature['analysis_summary'])}</p>")

    visualization = state.get("visualization_state")
    if visualization and isinstance(visualization, dict) and visualization.get("visualization_summary"):
        summary_parts.append(f"<p>{escape_html(visualization['visualization_summary'])}</p>")

    summary_parts.append("</div>")

    cleaning = state.get("cleaning_state")
    if cleaning and isinstance(cleaning, dict):
        report = cleaning.get("quality_report") or {}
        summary_parts.append("<div class='section'>")
        summary_parts.append("<h3>Data Quality Summary</h3>")
        summary_parts.append("<ul>")
        summary_parts.append(f"<li>After cleansing, the dataset contains {escape_html(str(report.get('final_rows', 'N/A')))} rows and {escape_html(str(report.get('final_columns', 'N/A')))} columns.</li>")
        duplicates = report.get('duplicates_after_cleaning', 0)
        if duplicates:
            summary_parts.append(f"<li>{escape_html(str(duplicates))} duplicate rows were removed to improve data reliability.</li>")
        missing = report.get('missing_after_cleaning', {})
        if any(count for count in missing.values()):
            missing_str = ", ".join(f"{escape_html(col)}: {count}" for col, count in missing.items() if count > 0)
            summary_parts.append(f"<li>Remaining missing values are concentrated in: {missing_str}.</li>")
        else:
            summary_parts.append("<li>Missing values were addressed successfully during cleaning.</li>")
        summary_parts.append("</ul>")
        summary_parts.append("</div>")

    if visualization and visualization.get('chart_explanations'):
        summary_parts.append("<div class='section'>")
        summary_parts.append("<h3>Visualization Highlights</h3>")
        chart_explanations = visualization.get('chart_explanations') or []
        if chart_explanations:
            summary_parts.append("<ul>")
            for chart in chart_explanations[:3]:
                summary_parts.append(f"<li><strong>{escape_html(chart.get('chart_title', 'Chart'))}:</strong> {escape_html(chart.get('insight', 'No insight available.'))}</li>")
            summary_parts.append("</ul>")
        summary_parts.append("</div>")

    if visualization and visualization.get('dashboard_html_path'):
        rel_dash_path = f"/storage/dashboards/{os.path.basename(visualization['dashboard_html_path'])}"
        summary_parts.append("<div class='dashboard-access'>")
        summary_parts.append(f"<a href='{rel_dash_path}' target='_blank' class='btn btn-primary'>Open Interactive BI Dashboard</a>")
        summary_parts.append("</div>")

    summary_parts.append("<div class='section'>")
    summary_parts.append("<h3>So what did we learn?</h3>")
    if eda and isinstance(eda, dict) and eda.get("analysis_summary"):
        summary_parts.append(f"<p>{escape_html(eda['analysis_summary'])}</p>")
    elif visualization and visualization.get('visualization_summary'):
        summary_parts.append(f"<p>{escape_html(visualization['visualization_summary'])}</p>")
    else:
        summary_parts.append("<p>The analysis has produced a clear, actionable view of the dataset and surfaced the strongest patterns for review.</p>")
    summary_parts.append("</div>")

    return "\n".join(summary_parts)

@app.get('/api/history')
async def get_history():
    history = []
    if not PIPELINE_DIR.exists():
        return []
    for file_path in PIPELINE_DIR.glob('*.json'):
        try:
            mtime = file_path.stat().st_mtime
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            history.append({
                'id': file_path.stem.replace('orchestrator_state_', ''),
                'query': data.get('user_query', 'Unknown Query'),
                'timestamp': mtime,
                'stage': data.get('current_stage', 'unknown')
            })
        except Exception as e:
            print(f'Error parsing history file {file_path.name}: {e}')
            continue
    history.sort(key=lambda x: x['timestamp'], reverse=True)
    return history

@app.get('/api/history/{run_id}')
async def get_run_details(run_id: str):
    file_name = f'orchestrator_state_{run_id}.json'
    file_path = PIPELINE_DIR / file_name
    if not file_path.exists():
        matching = list(PIPELINE_DIR.glob(f'*{run_id}*.json'))
        if matching:
            file_path = matching[0]
        else:
            raise HTTPException(status_code=404, detail='Run history not found')
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        plots = []
        eda_state = state.get('eda_state') or {}
        plot_paths = eda_state.get('plot_paths') or []
        for path in plot_paths:
            name = os.path.basename(path)
            plots.append(f'/storage/plots/{name}')
        return {
            'id': run_id,
            'query': state.get('user_query'),
            'stage': state.get('current_stage'),
            'plots': plots,
            'summary_html': generate_human_readable_summary(state),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error reading run details: {str(e)}')

@app.post('/api/chat')
async def run_pipeline(request: QueryRequest):
    query = request.query
    if not query.strip():
        raise HTTPException(status_code=400, detail='Query cannot be empty')
    print(f'Running pipeline for query: {query}')
    python_cmd = (
        f"import json, sys; "
        f"from agents.orchestrator_agent.orchestrator_agent import orchestrate_full_pipeline; "
        f"res = orchestrate_full_pipeline({json.dumps(query)}); "
        f"sys.stdout.buffer.write(b'###RESULT_START###'); "
        f"sys.stdout.buffer.write(json.dumps(res, ensure_ascii=False, allow_nan=False).encode('utf-8')); "
        f"sys.stdout.buffer.write(b'###RESULT_END###')"
    )
    cmd = [sys.executable, '-c', python_cmd]
    env = {**os.environ, "PYTHONUTF8": "1"}
    
    def run_subprocess():
        """Run subprocess in thread pool to avoid asyncio limitations on Windows."""
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(BASE_DIR),
            env=env,
        )
        stdout, stderr = proc.communicate(timeout=300)
        return proc.returncode, stdout, stderr
    
    try:
        returncode, stdout, stderr = await asyncio.to_thread(run_subprocess)
        stdout_str = stdout.decode('utf-8', errors='replace')
        stderr_str = stderr.decode('utf-8', errors='replace')
        if returncode != 0:
            print(f'Subprocess Error:\n{stderr_str}')
            raise RuntimeError(f'Pipeline execution failed:\n{stderr_str}')
        start_marker = '###RESULT_START###'
        end_marker = '###RESULT_END###'
        if start_marker not in stdout_str or end_marker not in stdout_str:
            print(f'Subprocess Output:\n{stdout_str}')
            raise RuntimeError('Pipeline completed but did not output structured result.')
        result_section = stdout_str.rsplit(start_marker, 1)[1]
        result_json = result_section.split(end_marker, 1)[0].strip()
        if not result_json:
            print(f'Subprocess Output:\n{stdout_str}')
            raise RuntimeError('Pipeline completed but returned an empty result payload.')
        state = json.loads(result_json)
        # Sanitize the state to remove NaN/Inf values
        state = _sanitize_for_json(state)
        meta_path = state.get('artifact_paths', {}).get('orchestration_metadata_path', '')
        run_id = Path(meta_path).stem.replace('orchestrator_state_', '') if meta_path else str(uuid.uuid4())
        plots = []
        eda_state = state.get('eda_state') or {}
        plot_paths = eda_state.get('plot_paths') or []
        for path in plot_paths:
            name = os.path.basename(path)
            plots.append(f'/storage/plots/{name}')
        response_payload = {
            'id': run_id,
            'query': query,
            'stage': state.get('current_stage'),
            'plots': plots,
            'summary_html': generate_human_readable_summary(state),
        }
        dashboard_path = state.get('artifact_paths', {}).get('dashboard_html_path')
        if dashboard_path:
            response_payload['dashboard_url'] = f"/storage/dashboards/{os.path.basename(dashboard_path)}"
        return response_payload
    except HTTPException:
        raise
    except Exception as e:
        print(f'Exception during pipeline execution: {e}')
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e) or 'Unknown pipeline execution error')

app.mount('/storage', StaticFiles(directory=str(STORAGE_DIR)), name='storage')
frontend_dir = BASE_DIR / 'frontend'
frontend_dir.mkdir(exist_ok=True)
app.mount('/', StaticFiles(directory=str(frontend_dir), html=True), name='frontend')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='127.0.0.1', port=8000, reload=True)
