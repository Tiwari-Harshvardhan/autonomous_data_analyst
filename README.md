# Autonomous Data Analyst

> A local autonomous data analytics assistant for scraping, extracting, cleaning, profiling, engineering, and visualizing datasets through a web-based UI.

![Architecture](docs/images/architecture.svg)

## 🚀 Project Overview

`Autonomous Data Analyst` is a modular, multi-agent pipeline built in Python. It accepts a user prompt or URL, orchestrates end-to-end data analysis using specialized sub-agents, and serves an interactive frontend at `http://localhost:8000`.

This repository includes:

- A FastAPI backend server (`server.py`)
- A browser-based frontend interface (`frontend/`)
- Agent components for each stage of the data workflow
- Local artifact persistence under `storage/`

## 🔧 What It Does

The system can:

- Scrape web content from URLs using static requests or Playwright for dynamic pages
- Parse HTML into structured records
- Convert extracted content into tabular CSV datasets
- Clean and standardize the data
- Apply feature engineering and statistical transformations
- Run exploratory data analysis and visualizations
- Generate interactive dashboard HTML reports
- Keep a search-able run history with logs and artifacts

## 🧠 Workflow

![UI Workflow](docs/images/ui-workflow.svg)

### Pipeline stages

1. **Data Collection**
   - Extracts URLs from the user query
   - Scrapes pages with either `requests` + BeautifulSoup or Playwright
   - Saves raw HTML, screenshots, and JSON metadata

2. **Extraction**
   - Parses saved HTML files
   - Extracts headings, paragraphs, links, tables, and page titles
   - Builds a tidy CSV dataset for downstream processing

3. **Data Cleaning**
   - Profiles the dataset
   - Handles missing values
   - Removes duplicates
   - Standardizes text columns
   - Validates numeric ranges

4. **Feature Engineering**
   - Applies scaling and normalization
   - Corrects skewed numeric features
   - Extracts date parts from datetime columns
   - Saves transformed CSV artifacts

5. **Exploratory Data Analysis (EDA)**
   - Profiles numeric and categorical fields
   - Detects outliers using the IQR method
   - Computes skewness and bivariate relationships
   - Generates PNG plots for distributions and counts

6. **Visualization**
   - Builds a Plotly-powered HTML dashboard
   - Produces charts for numerical distributions, relationships, and categorical breakdowns
   - Saves the dashboard under `storage/dashboards/`

## 📁 Repository Structure

```text
autonomous_data_analyst/
├── README.md
├── installation_and_setup.py
├── local_test_runner.py
├── requirements.txt
├── server.py
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── style.css
├── agents/
│   ├── __init__.py
│   ├── orchestrator_agent/
│   │   └── orchestrator_agent.py
│   ├── data_collection_agent/
│   │   └── data_collection_agent.py
│   ├── extraction_agent/
│   │   └── extraction_agent.py
│   ├── data_cleaning_agent/
│   │   └── data_cleaning_agent.py
│   ├── feature_engineering_agent/
│   │   └── feature_engineering_agent.py
│   ├── eda_agent/
│   │   └── eda_agent.py
│   └── visualization_agent/
│       └── visualization_agent.py
├── storage/
│   ├── cleaned_data/
│   ├── dashboards/
│   ├── dataframes/
│   ├── eda_data/
│   ├── metadata/
│   ├── plots/
│   ├── profiles/
│   ├── raw/
│   ├── raw_html/
│   ├── screenshots/
│   └── visual_metadata/
└── docs/
    └── images/
        ├── architecture.svg
        └── ui-workflow.svg
```

## ⚙️ Setup & Installation

1. Create or activate your Python environment.

```powershell
python -m venv my-adk-env
my-adk-env\Scripts\activate
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. If you want project scaffolding or environment bootstrap, run:

```powershell
python installation_and_setup.py
```

> `installation_and_setup.py` will create a `.env` file with the expected `GOOGLE_API_KEY` placeholder if one does not already exist.

## ▶️ Run the App

### Start the backend server

```powershell
python server.py
```

or

```powershell
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### Open the frontend

Visit `http://localhost:8000` in your browser.

### Example prompt

In the UI, enter a prompt such as:

```text
Scrape data from https://example.com and analyze it
```

The app will begin a run and you can watch live logs, view summaries, and open generated dashboards.

## 🧪 Local Testing

Run the built-in pipeline test:

```powershell
python local_test_runner.py
```

This verifies that the end-to-end workflow completes and that expected artifacts are written to disk.

## 📡 API Endpoints

- `POST /api/chat`
  - Body: `{ "user_query": "..." }`
  - Starts a new pipeline run and returns a `run_id`

- `GET /api/history`
  - Returns a list of saved pipeline run metadata files

- `GET /api/history/{run_id}`
  - Returns the run state, logs, and embedded summary HTML


## 🧩 How the App Works

The system is built around an orchestration agent in `agents/orchestrator_agent/orchestrator_agent.py`.

- `server.py` receives a query and starts a background thread.
- The orchestrator plans the pipeline and coordinates sub-agents.
- Each agent is responsible for one stage and saves artifacts to `storage/`.
- The frontend fetches history and renders summaries for completed runs.

## 🗂️ Artifact Locations

- Raw scrape JSON: `storage/raw/`
- Extracted CSV tables: `storage/dataframes/`
- Cleaned CSVs: `storage/cleaned_data/`
- Engineered datasets: `storage/eda_data/`
- EDA plots: `storage/plots/`
- Dashboards: `storage/dashboards/`
- Metadata & logs: `storage/metadata/`

## 💡 Tips

- Use a direct URL in your query for faster and more predictable collection.
- If you already have a CSV dataset, you can pass a local path to the orchestrator in code and skip scraping.
- The dashboard HTML files are static and can be opened directly if the app is not running.

## 📌 Notes

- The app is designed to run locally in a self-contained environment.
- It uses Google ADK agent abstractions, but the core data processing is implemented in Python.
- For best results, keep dependencies up to date and ensure Playwright can launch Chromium in your environment.

## 🛠️ Future Enhancements

- Add file upload support for direct CSV ingestion from the frontend
- Add real-time progress steps and richer run metadata in the UI
- Support additional extraction types such as PDF, APIs, and structured feeds
- Add a guided run builder for advanced analysis templates

---

Made for local autonomous data analytics with an emphasis on meaningful artifacts, transparency, and reusable pipeline stages.
