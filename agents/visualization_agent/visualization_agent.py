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
    chart_explanations: List[Dict[str, Any]] = Field(default_factory=list)
    visualization_summary: Optional[str] = None
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
    state.logs.append("[INIT] Started visualization generation.")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        state.logs.append(f"[ERROR] Failed to read dataset: {str(e)}")
        state.visualization_summary = "No visualizations were generated because the dataset could not be loaded."
        return state

    if df.empty:
        state.logs.append("[INFO] Dataset contains no rows. Skipping visualization generation.")
        state.visualization_summary = "The dataset was empty, so no analytical visualizations could be produced."
        return state

    def meaningful_numeric(col: str) -> bool:
        values = df[col].dropna()
        return len(values) >= 10 and values.nunique() > 4

    def meaningful_categorical(col: str) -> bool:
        values = df[col].dropna()
        return len(values) >= 10 and 1 < values.nunique() <= 40

    def meaningful_datetime(col: str) -> bool:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return df[col].dropna().nunique() > 1 and len(df[col].dropna()) >= 10
        converted = pd.to_datetime(df[col], errors='coerce')
        return converted.notna().sum() >= 10 and converted.nunique() > 1

    numeric_cols = [col for col in df.select_dtypes(include=['number']).columns if meaningful_numeric(col)]
    categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if meaningful_categorical(col)]
    datetime_cols = [col for col in df.columns if meaningful_datetime(col)]

    plot_html = []
    html_components: List[str] = []
    chart_candidates: List[Dict[str, Any]] = []

    if datetime_cols and numeric_cols:
        date_col = datetime_cols[0]
        value_col = numeric_cols[0]
        chart_candidates.append({
            'type': 'trend',
            'columns': [date_col, value_col],
            'title': f'Trend over time: {value_col} by {date_col}',
            'reason': 'Time series data exists alongside a numeric measure, making a trend line meaningful.',
            'insight': f'This chart shows how {value_col} evolves over time and highlights any directionally consistent patterns or seasonality.'
        })

    for col in numeric_cols:
        chart_candidates.append({
            'type': 'histogram',
            'columns': [col],
            'title': f'Distribution of {col}',
            'reason': 'Numeric feature with sufficient samples and variance.',
            'insight': f'This histogram reveals the shape of {col}, including skew, central tendency, and whether values are concentrated in one range.'
        })
        if df[col].dropna().nunique() > 10:
            chart_candidates.append({
                'type': 'boxplot',
                'columns': [col],
                'title': f'Boxplot for {col}',
                'reason': 'Numeric feature with enough data for outlier and spread evaluation.',
                'insight': f'This boxplot exposes outliers, variability, and whether the central values are tightly clustered.'
            })

    if len(numeric_cols) >= 2:
        chart_candidates.append({
            'type': 'scatter',
            'columns': [numeric_cols[0], numeric_cols[1]],
            'title': f'Relationship between {numeric_cols[0]} and {numeric_cols[1]}',
            'reason': 'Two meaningful numeric variables exist, allowing relational analysis.',
            'insight': f'This scatter plot helps identify correlation or groupings between {numeric_cols[0]} and {numeric_cols[1]}. '
        })

    if len(numeric_cols) >= 3:
        chart_candidates.append({
            'type': 'heatmap',
            'columns': numeric_cols[:4],
            'title': 'Correlation heatmap for numeric features',
            'reason': 'Multiple numeric variables are available, making correlation analysis valuable.',
            'insight': 'This heatmap summarizes the strength of relationships among the strongest numeric features.'
        })

    for col in categorical_cols:
        if df[col].dropna().nunique() <= 20:
            chart_candidates.append({
                'type': 'bar',
                'columns': [col],
                'title': f'Frequency of {col}',
                'reason': 'Categorical column with a manageable number of levels and sufficient observations.',
                'insight': f'This bar chart highlights the most common values in {col} and reveals potential dominance or imbalance.'
            })

    if not chart_candidates:
        state.visualization_summary = 'No meaningful visualizations could be generated because the dataset lacks numeric or categorical features with sufficient samples.'
        state.logs.append('[INFO] No visualizations generated due to insufficient analytical signal.')
    else:
        for chart in chart_candidates:
            try:
                if chart['type'] == 'trend':
                    series = pd.to_datetime(df[chart['columns'][0]], errors='coerce')
                    plot_df = pd.DataFrame({
                        chart['columns'][0]: series,
                        chart['columns'][1]: df[chart['columns'][1]]
                    }).dropna()
                    if len(plot_df) < 10:
                        continue
                    fig = px.line(plot_df.sort_values(chart['columns'][0]), x=chart['columns'][0], y=chart['columns'][1], title=chart['title'], template='plotly_white')
                elif chart['type'] == 'histogram':
                    fig = px.histogram(df, x=chart['columns'][0], title=chart['title'], template='plotly_white', color_discrete_sequence=['#4682B4'])
                elif chart['type'] == 'boxplot':
                    fig = px.box(df, y=chart['columns'][0], title=chart['title'], template='plotly_white', color_discrete_sequence=['#5B84B1'])
                elif chart['type'] == 'scatter':
                    fig = px.scatter(df, x=chart['columns'][0], y=chart['columns'][1], title=chart['title'], template='plotly_white', color_discrete_sequence=['#E67E22'])
                elif chart['type'] == 'heatmap':
                    corr = df[chart['columns']].corr()
                    fig = px.imshow(corr, text_auto=True, title=chart['title'], color_continuous_scale='RdBu_r')
                else:
                    continue

                fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
                if chart['type'] == 'heatmap':
                    fig.update_xaxes(tickangle=45)
                plot_html.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn' if not plot_html else False))
                state.generated_plots.append(chart['title'])
                state.chart_explanations.append({
                    'chart_title': chart['title'],
                    'reason': chart['reason'],
                    'insight': chart['insight']
                })
                state.logs.append(f"[PLOT] Generated {chart['type']} chart for {', '.join(chart['columns'])}.")
            except Exception as exc:
                state.logs.append(f"[WARN] Skipped chart {chart['title']} due to: {exc}")

        if plot_html:
            state.visualization_summary = f"Created {len(plot_html)} meaningful visualization(s) based on dataset structure and analytical value."
        else:
            state.visualization_summary = 'No meaningful visualizations were created because the dataset did not meet the insight thresholds.'
            state.logs.append('[INFO] No charts retained after relevance filtering.')

    if plot_html:
        html_components.extend(plot_html)

    state.visualization_summary = state.visualization_summary or 'Visualization generation completed with a neutral insight profile.'

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
                <p>{state.visualization_summary}</p>
            </header>
            <div class="grid">
                {"".join([f'<div class="card">{chart}</div>' for chart in html_components])}
            </div>
        </div>
    </body>
    </html>
    """

    output_html_path = _unique_path(DASHBOARD_DIR, "tableau_dashboard", "html")
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(dashboard_html)

    state.dashboard_html_path = output_html_path
    state.logs.append("Interactive dashboard compiled and persisted.")
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