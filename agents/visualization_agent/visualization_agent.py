import os
import json
import uuid
import pandas as pd
import plotly.express as px
import plotly.io as pio

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from google.adk.agents import Agent

# Extending your existing storage directory tree
BASE_STORAGE_DIR = "storage"
DASHBOARD_DIR = os.path.join(BASE_STORAGE_DIR, "dashboards")
VISUAL_METADATA_DIR = os.path.join(BASE_STORAGE_DIR, "visual_metadata")

for d in (DASHBOARD_DIR, VISUAL_METADATA_DIR):
    os.makedirs(d, exist_ok=True)

class VisualisationState(BaseModel):
    input_dataframe_path: str
    dashboard_html_path: Optional[str] = None
    generated_plots: List[str] = Field(default_factory=list)
    metadata_path: Optional[str] = None
    logs: List[str] = Field(default_factory=list)

def _unique_path(directory: str, prefix: str, extension: str) -> str:
    return os.path.join(directory, f"{prefix}_{uuid.uuid4()}.{extension}")

def save_visual_metadata(state: VisualisationState) -> str:
    path = _unique_path(VISUAL_METADATA_DIR, "visualisation_metadata", "json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(state.model_dump()), f, indent=2, ensure_ascii=False)
    return path


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if obj != obj or obj == float('inf') or obj == float('-inf'):
            return None
        return obj
    elif isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    elif hasattr(obj, 'tolist'):
        return _sanitize_for_json(obj.tolist())
    return str(obj)


def generate_interactive_dashboard(csv_path: str) -> VisualisationState:
    state = VisualisationState(input_dataframe_path=csv_path)
    state.logs.append(f"[INIT] Started visualization generation for: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        state.logs.append(f"[ERROR] Failed to read file: {str(e)}")
        return state

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    html_components = []
    
    # 1. Tableau-Style Summary Cards (KPIs)
    kpi_html = "<div style='display: flex; gap: 20px; margin-bottom: 30px;'>"
    kpi_html += f"<div style='flex: 1; background: #f8f9fa; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'><strong>Total Rows</strong><h2 style='margin: 10px 0 0 0; color: #2c3e50;'>{df.shape[0]}</h2></div>"
    kpi_html += f"<div style='flex: 1; background: #f8f9fa; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'><strong>Total Features</strong><h2 style='margin: 10px 0 0 0; color: #2c3e50;'>{df.shape[1]}</h2></div>"
    kpi_html += f"<div style='flex: 1; background: #f8f9fa; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'><strong>Numeric Columns</strong><h2 style='margin: 10px 0 0 0; color: #2c3e50;'>{len(numeric_cols)}</h2></div>"
    kpi_html += "</div>"
    html_components.append(kpi_html)

    # 2. Distribution Plots (Histograms for numeric data)
    if numeric_cols:
        for col in numeric_cols[:3]: # Limit to top 3 to keep dashboard performant
            fig = px.histogram(df, x=col, title=f"Distribution of {col}", 
                               template="plotly_white", color_discrete_sequence=['#4682B4'])
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            html_components.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn' if not html_components else False))
            state.generated_plots.append(f"Histogram: {col}")
            state.logs.append(f"[PLOT] Generated distribution plot for {col}")

    # 3. Relationship Plots (Scatter Matrix / Correlation heatmap placeholder)
    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                         title=f"Relationship: {numeric_cols[0]} vs {numeric_cols[1]}",
                         template="plotly_white", color_discrete_sequence=['#E67E22'])
        fig.update_layout(height=400)
        html_components.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))
        state.generated_plots.append(f"Scatter: {numeric_cols[0]} vs {numeric_cols[1]}")
        state.logs.append(f"[PLOT] Generated scatter plot matrix sample")

    # 4. Categorical Breakdown (Bar Charts)
    if categorical_cols:
        for col in categorical_cols[:2]:
            counts = df[col].value_counts().reset_index().head(10)
            fig = px.bar(counts, x=col, y='count', title=f"Top Categorical Breakdowns: {col}",
                         template="plotly_white", color_discrete_sequence=['#2ECC71'])
            fig.update_layout(height=350)
            html_components.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))
            state.generated_plots.append(f"Bar Chart: {col}")
            state.logs.append(f"[PLOT] Generated categorical distribution for {col}")

    # Construct the Final Tableau-style HTML Canvas
    dashboard_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Automated Insights Dashboard</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 30px; background-color: #f3f4f6; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
            header {{ border-bottom: 2px solid #eaeded; padding-bottom: 15px; margin-bottom: 30px; }}
            h1 {{ color: #111827; margin: 0; font-size: 28px; }}
            p.subtitle {{ color: #6b7280; margin: 5px 0 0 0; font-size: 14px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 25px; margin-top: 20px; }}
            .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>BI Dashboard Engine</h1>
                <p class="subtitle">Automated visual discovery profile for dataset: <code>{os.path.basename(csv_path)}</code></p>
            </header>
            
            {html_components[0]} <div class="grid">
                {"".join([f'<div class="card">{chart}</div>' for chart in html_components[1:]])}
            </div>
        </div>
    </body>
    </html>
    """
    
    output_html_path = _unique_path(DASHBOARD_DIR, "tableau_dashboard", "html")
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(dashboard_html)
        
    state.dashboard_html_path = output_html_path
    state.logs.append(f"[PERSIST] Interactive Tableau-like dashboard compiled and saved to: {output_html_path}")
    
    state.metadata_path = save_visual_metadata(state)
    return state

def run_visualisation_pipeline(csv_path: str) -> dict:
    state = generate_interactive_dashboard(csv_path)
    return _sanitize_for_json(state.model_dump())

# Define the Visualisation Agent
visualisation_agent = Agent(
    name="visualisation_agent",
    model="gemini-2.0-flash",
    description="Automated business intelligence system that evaluates transformed datasets to compile responsive interactive dashboards.",
    instruction="""
    You accept tabular datasets (.csv paths) processed by the Feature Engineering pipeline to generate rich visual representations.
    Always execute the tool function `run_visualisation_pipeline` to scan numerical distribution scales, trends, and categorical metrics.
    Review runtime log tracking info to verify how many distributions or relationship patterns were parsed into the layout map.
    Communicate final analytical products clearly, supplying the user with the file paths where the interactive HTML Tableau dashboards can be reviewed.
    """,
    tools=[run_visualisation_pipeline]
)

if __name__ == "__main__":
    # Example standalone execution test
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "storage/eda_data/engineered_data_example.csv"
    # Create mock dataset if testing directly
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        pd.DataFrame({
            'Age': [23, 45, 12, 36, 52, 40, 22],
            'Salary': [50000, 84000, 0, 62000, 110000, 90000, 52000],
            'Department': ['HR', 'Eng', 'Student', 'Eng', 'Exec', 'Eng', 'HR']
        }).to_csv(csv_path, index=False)
        
    final_visual_state = generate_interactive_dashboard(csv_path)
    print(final_visual_state.model_dump_json(indent=2))