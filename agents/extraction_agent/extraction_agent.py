import os
import json
import uuid
import asyncio
import pandas as pd

from bs4 import BeautifulSoup
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from google.adk.agents import Agent


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


# ---------------------------------------------------------------------------
# Storage setup
# ---------------------------------------------------------------------------

BASE_STORAGE_DIR = "storage"
PARSED_DATA_DIR  = os.path.join(BASE_STORAGE_DIR, "parsed")
DATAFRAME_DIR    = os.path.join(BASE_STORAGE_DIR, "dataframes")
METADATA_DIR     = os.path.join(BASE_STORAGE_DIR, "metadata")

for d in (PARSED_DATA_DIR, DATAFRAME_DIR, METADATA_DIR):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ParsedPage(BaseModel):
    """Structured data extracted from a single HTML file."""
    source_html_path: str
    url: Optional[str] = None
    title: Optional[str] = None
    headings: List[str] = Field(default_factory=list)
    paragraphs: List[str] = Field(default_factory=list)
    links: List[Dict[str, str]] = Field(default_factory=list)   # [{text, href}]
    tables: List[List[List[str]]] = Field(default_factory=list) # [table][row][cell]
    success: bool = True
    error: Optional[str] = None


class ExtractionState(BaseModel):
    """Tracks the full extraction pipeline for a batch of HTML files."""
    html_paths: List[str]
    parsed_pages: List[Dict[str, Any]] = Field(default_factory=list)
    dataframe_path: Optional[str] = None
    metadata_path: Optional[str] = None
    logs: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------

class HTMLExtractor:
    """
    Reads a saved HTML file from disk and uses BeautifulSoup to pull out
    structured content: title, headings, paragraphs, links, and tables.
    """

    def extract(self, html_path: str, url: Optional[str] = None) -> ParsedPage:
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()

            soup = BeautifulSoup(html, "html.parser")

            title = soup.title.string.strip() if soup.title and soup.title.string else None

            headings = [
                tag.get_text(strip=True)
                for tag in soup.find_all(["h1", "h2", "h3"])
                if tag.get_text(strip=True)
            ]

            paragraphs = [
                p.get_text(strip=True)
                for p in soup.find_all("p")
                if p.get_text(strip=True)
            ]

            links = [
                {"text": a.get_text(strip=True), "href": a.get("href", "")}
                for a in soup.find_all("a", href=True)
                if a.get_text(strip=True)
            ]

            tables = self._extract_tables(soup)

            return ParsedPage(
                source_html_path=html_path,
                url=url,
                title=title,
                headings=headings,
                paragraphs=paragraphs,
                links=links,
                tables=tables,
            )

        except Exception as e:
            return ParsedPage(source_html_path=html_path, success=False, error=str(e))

    def _extract_tables(self, soup: BeautifulSoup) -> List[List[List[str]]]:
        """
        Returns all HTML tables as a list of 2D cell grids.
        Each table is a list of rows; each row is a list of cell strings.
        """
        tables = []
        for table_tag in soup.find_all("table"):
            rows = []
            for tr in table_tag.find_all("tr"):
                cells = [
                    td.get_text(strip=True)
                    for td in tr.find_all(["td", "th"])
                ]
                if cells:
                    rows.append(cells)
            if rows:
                tables.append(rows)
        return tables


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def build_dataframe(parsed_pages: List[ParsedPage]) -> pd.DataFrame:
    """
    Flattens a list of ParsedPage objects into a tidy DataFrame.

    Each row represents one paragraph from one page, with the page-level
    fields (title, url, heading count, link count) repeated on every row.
    This makes the data easy to filter, search, and export.
    """
    rows = []
    for page in parsed_pages:
        if not page.success:
            continue

        # If the page has no paragraphs, still emit one row so the page
        # appears in the output with its metadata intact.
        texts = page.paragraphs if page.paragraphs else [""]

        for para in texts:
            rows.append({
                "url":            page.url,
                "title":          page.title,
                "paragraph":      para,
                "heading_count":  len(page.headings),
                "link_count":     len(page.links),
                "table_count":    len(page.tables),
                "html_source":    page.source_html_path,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_dataframe(df: pd.DataFrame) -> str:
    """Writes the DataFrame to CSV and returns the file path."""
    path = os.path.join(DATAFRAME_DIR, f"extracted_{uuid.uuid4()}.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def save_metadata(state: ExtractionState) -> str:
    path = os.path.join(METADATA_DIR, f"extraction_metadata_{uuid.uuid4()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.model_dump(), f, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Extraction pipeline
# ---------------------------------------------------------------------------

def execute_extraction(
    html_paths: List[str],
    url_map: Optional[Dict[str, str]] = None,
) -> ExtractionState:
    """
    Reads each HTML file, extracts structured content, builds a DataFrame,
    and saves everything to disk.

    Args:
        html_paths: Paths to the raw HTML files saved by the scraper.
        url_map:    Optional {html_path: original_url} so the DataFrame
                    records where each page came from.
    """
    url_map = url_map or {}
    state = ExtractionState(html_paths=html_paths)
    extractor = HTMLExtractor()

    state.logs.append(f"Starting extraction for {len(html_paths)} HTML files")

    # Step 1: parse every HTML file
    parsed_pages: List[ParsedPage] = []
    for path in html_paths:
        print(f"Parsing: {path}")
        page = extractor.extract(path, url=url_map.get(path))
        parsed_pages.append(page)
        status = "ok" if page.success else f"failed ({page.error})"
        state.logs.append(f"{path} → {status}")

    state.parsed_pages = [p.model_dump() for p in parsed_pages]
    state.logs.append("Extraction complete")

    # Step 2: build the DataFrame
    df = build_dataframe(parsed_pages)
    state.logs.append(f"DataFrame built: {len(df)} rows, {len(df.columns)} columns")
    print(f"\nDataFrame shape: {df.shape}")
    print(df.head())

    # Step 3: save to disk
    state.dataframe_path = save_dataframe(df)
    state.logs.append(f"DataFrame saved: {state.dataframe_path}")

    state.metadata_path = save_metadata(state)
    state.logs.append(f"Metadata saved: {state.metadata_path}")

    print(f"\nDataFrame saved to: {state.dataframe_path}")
    return state


# ---------------------------------------------------------------------------
# ADK tool wrapper
# ---------------------------------------------------------------------------

def extract_from_html_file(html_path: str, url: Optional[str] = None) -> dict:
    """
    ADK-compatible tool. Parses a single HTML file and returns
    the structured result as a dict.
    """
    extractor = HTMLExtractor()
    result = extractor.extract(html_path, url=url)
    return _sanitize_for_json(result.model_dump())


# ---------------------------------------------------------------------------
# ADK agent
# ---------------------------------------------------------------------------

extraction_agent = Agent(
    model="gemini-2.0-flash",
    name="extraction_agent",
    description=(
        "Reads saved HTML files, extracts structured content using BeautifulSoup, "
        "and stores the results as a CSV-backed pandas DataFrame."
    ),
    instruction="""
    You are a data extraction agent. Your job is:
    1. Accept a list of HTML file paths (produced by the scraper agent).
    2. Call extract_from_html_file on each path to parse its contents.
    3. Report what was extracted: titles, paragraph counts, link counts, tables found.
    4. Confirm where the final DataFrame CSV was saved.

    Rules:
    - Never fabricate extracted content.
    - If a file fails to parse, log the error and continue with the rest.
    - Always report the dataframe_path from the ExtractionState at the end.
    """,
    tools=[extract_from_html_file],
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # In production these paths come from the scraper agent's WorkflowState.
    # Here we use example paths to demonstrate the pipeline.
    sample_html_paths = [
        "storage/raw_html/example1.html",
        "storage/raw_html/example2.html",
    ]
    sample_url_map = {
        "storage/raw_html/example1.html": "https://en.wikipedia.org/wiki/Machine_learning",
        "storage/raw_html/example2.html": "https://www.ibm.com/topics/machine-learning",
    }

    final_state = execute_extraction(sample_html_paths, url_map=sample_url_map)
    print(final_state.model_dump_json(indent=2))