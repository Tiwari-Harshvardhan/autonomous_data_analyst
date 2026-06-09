import os
import json
import uuid
import re
import pandas as pd

from bs4 import BeautifulSoup
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from google.adk.agents import Agent


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
    """Structured data extracted from a single raw data record."""
    source_html_path: str
    url: Optional[str] = None
    title: Optional[str] = None
    extraction_method: str = "html_parse"   # "structured_data" | "product_cards" | "html_parse"
    records: List[Dict[str, Any]] = Field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class ExtractionState(BaseModel):
    """Tracks the full extraction pipeline for a batch of raw scraper records."""
    html_paths: List[str]
    parsed_pages: List[Dict[str, Any]] = Field(default_factory=list)
    dataframe_path: Optional[str] = None
    metadata_path: Optional[str] = None
    logs: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Encoding fix
# ---------------------------------------------------------------------------

def _fix_encoding(text: str) -> str:
    """
    Repairs mojibake produced when UTF-8 bytes are decoded as latin-1.
    e.g. 'â£51.77' → '£51.77'
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def _clean_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Applies encoding fix to every string value in a record dict."""
    return {
        k: (_fix_encoding(v) if isinstance(v, str) else v)
        for k, v in record.items()
    }


# ---------------------------------------------------------------------------
# Extraction strategies
# ---------------------------------------------------------------------------

class HTMLExtractor:
    """
    Three-tier extractor. For each raw scraper record it tries strategies
    in priority order and stops at the first one that yields useful rows:

      1. structured_data  — scraper already extracted product cards / table rows
      2. product_cards    — re-parse HTML looking for repeated card elements
      3. html_parse       — generic fallback: headings + paragraphs + tables
    """

    def extract(
        self,
        html_path: str,
        url: Optional[str] = None,
        structured_data: Optional[List[Dict[str, Any]]] = None,
        page_type: Optional[str] = None,
        title: Optional[str] = None,
    ) -> ParsedPage:

        # --- Strategy 1: use pre-extracted structured_data from the scraper ---
        if structured_data:
            records = [_clean_record(r) for r in structured_data if r]
            if records:
                return ParsedPage(
                    source_html_path=html_path,
                    url=url,
                    title=title,
                    extraction_method="structured_data",
                    records=records,
                )

        # --- Read HTML from disk for strategies 2 & 3 ---
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()
        except Exception as e:
            return ParsedPage(
                source_html_path=html_path, success=False,
                error=f"Could not read HTML file: {e}"
            )

        soup = BeautifulSoup(html, "html.parser")
        page_title = (
            title
            or (soup.title.string.strip() if soup.title and soup.title.string else None)
        )

        # --- Strategy 2: product card re-parse (for product_listing pages) ---
        if page_type == "product_listing" or self._looks_like_product_page(soup):
            records = self._extract_product_cards(soup)
            if records:
                return ParsedPage(
                    source_html_path=html_path,
                    url=url,
                    title=page_title,
                    extraction_method="product_cards",
                    records=records,
                )

        # --- Strategy 3: generic HTML parse ---
        records = self._extract_generic(soup, url, page_title, html_path)
        return ParsedPage(
            source_html_path=html_path,
            url=url,
            title=page_title,
            extraction_method="html_parse",
            records=records,
        )

    # ------------------------------------------------------------------
    # Strategy 2 helpers
    # ------------------------------------------------------------------

    def _looks_like_product_page(self, soup: BeautifulSoup) -> bool:
        """Quick heuristic check without re-classifying the full page."""
        return bool(
            soup.find_all("article")
            or soup.find_all(class_=re.compile(r"product|item|card", re.I))
        )

    def _extract_product_cards(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extracts product records from repeated article/card elements.
        Targets Books to Scrape's <article class="product_pod"> pattern
        as well as generic product grids on other e-commerce sites.
        """
        records: List[Dict[str, Any]] = []

        candidate_selectors = [
            {"tag": "article", "attrs": {}},
            {"tag": "li",  "attrs": {"class": re.compile(r"product|item|card", re.I)}},
            {"tag": "div", "attrs": {"class": re.compile(r"product|item|card|grid", re.I)}},
        ]

        for sel in candidate_selectors:
            cards = soup.find_all(sel["tag"], attrs=sel["attrs"])
            if len(cards) < 3:
                continue

            for card in cards:
                record: Dict[str, Any] = {}

                # Title / name
                name_tag = (
                    card.find("h3") or card.find("h2") or card.find("h4")
                    or card.find("strong")
                    or card.find(class_=re.compile(r"name|title", re.I))
                )
                if name_tag:
                    # Books to Scrape puts the real title in the <a> title attr
                    a = name_tag.find("a")
                    record["name"] = (
                        a["title"] if a and a.get("title") else name_tag.get_text(strip=True)
                    )

                # Price — class-based first, then currency symbol fallback
                price_tag = card.find(class_=re.compile(r"price", re.I))
                if price_tag:
                    record["price"] = _fix_encoding(price_tag.get_text(strip=True))

                # Rating — Books to Scrape uses <p class="star-rating One/Two/...">
                rating_tag = card.find(class_=re.compile(r"star-rating|rating", re.I))
                if rating_tag:
                    # CSS class holds the word rating e.g. "star-rating Three"
                    classes = " ".join(rating_tag.get("class", []))
                    word_rating = re.search(
                        r"\b(One|Two|Three|Four|Five)\b", classes, re.I
                    )
                    record["rating"] = (
                        word_rating.group(1) if word_rating
                        else rating_tag.get_text(strip=True)
                    )

                # Availability
                avail_tag = card.find(class_=re.compile(r"availab|stock", re.I))
                if avail_tag:
                    record["availability"] = avail_tag.get_text(strip=True)

                # Product URL
                link = card.find("a", href=True)
                if link:
                    href = link["href"]
                    record["product_url"] = href

                if record:
                    records.append(record)

            if records:
                break

        return records

    # ------------------------------------------------------------------
    # Strategy 3 helper
    # ------------------------------------------------------------------

    def _extract_generic(
        self,
        soup: BeautifulSoup,
        url: Optional[str],
        title: Optional[str],
        html_path: str,
    ) -> List[Dict[str, Any]]:
        """
        Fallback for non-product pages (articles, Wikipedia, etc.).
        Returns one record per paragraph with page metadata attached.
        Also extracts table rows as individual records when tables are present.
        """
        records: List[Dict[str, Any]] = []

        # Tables first — they contain the most structured data
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if not rows:
                continue
            headers = [th.get_text(strip=True) for th in rows[0].find_all("th")]
            for row in (rows[1:] if headers else rows):
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                if not cells:
                    continue
                if headers and len(cells) == len(headers):
                    records.append(dict(zip(headers, cells)))
                else:
                    records.append({f"col_{i}": v for i, v in enumerate(cells)})

        # Paragraphs + headings if no tables produced records
        if not records:
            for tag in soup.find_all(["h1", "h2", "h3", "p"]):
                text = _fix_encoding(tag.get_text(strip=True))
                if text:
                    records.append({
                        "tag":   tag.name,
                        "text":  text,
                        "url":   url,
                        "title": title,
                    })

        return records


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def build_dataframe(parsed_pages: List[ParsedPage]) -> pd.DataFrame:
    """
    Builds a single tidy DataFrame from all parsed pages.

    - For structured_data / product_cards pages: each record is one row
      (one product), with url and title added as metadata columns.
    - For html_parse pages: each paragraph/table-row is one row.

    The method column records which extraction strategy was used, which
    is useful for debugging and for the analyzer agent's quality check.
    """
    all_rows: List[Dict[str, Any]] = []

    for page in parsed_pages:
        if not page.success or not page.records:
            continue

        for record in page.records:
            row = dict(record)
            # Always attach page-level metadata so the analyzer can verify
            row.setdefault("source_url",   page.url)
            row.setdefault("page_title",   page.title)
            row["extraction_method"] = page.extraction_method
            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Drop columns that are 100 % identical across all rows — they add no
    # analytical value and are exactly what caused the bad charts (e.g. a
    # 'url' column where every row is 'https://books.toscrape.com').
    cols_to_drop = [
        col for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_dataframe(df: pd.DataFrame) -> str:
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
#
# The key change: execute_extraction now accepts an optional raw_data_path
# so it can read the scraper's structured_data directly, rather than always
# re-parsing from HTML.
# ---------------------------------------------------------------------------

def execute_extraction(
    html_paths: List[str],
    url_map: Optional[Dict[str, str]] = None,
    raw_data_path: Optional[str] = None,     # NEW: path to scraper's raw JSON
) -> ExtractionState:
    """
    Reads each raw scraper record and extracts structured content.

    Priority:
      1. If raw_data_path is provided, read structured_data and page_type
         from the scraper's JSON output for each URL.
      2. Fall back to HTML re-parsing when structured_data is absent.

    Args:
        html_paths:     Paths to saved HTML files.
        url_map:        {html_path: original_url}
        raw_data_path:  Path to the raw_data JSON written by save_raw_data()
                        in data_collection_agent.py. When provided, the extractor
                        uses pre-extracted product records instead of re-parsing.
    """
    url_map = url_map or {}
    state = ExtractionState(html_paths=html_paths)
    extractor = HTMLExtractor()

    # Build a lookup from html_path → scraper record so we can pass
    # structured_data and page_type into the extractor.
    scraper_record_map: Dict[str, Dict[str, Any]] = {}
    if raw_data_path:
        try:
            with open(raw_data_path, "r", encoding="utf-8") as f:
                raw_records: List[Dict[str, Any]] = json.load(f)
            for rec in raw_records:
                hp = rec.get("html_path")
                if hp:
                    scraper_record_map[hp] = rec
            state.logs.append(
                f"Loaded {len(scraper_record_map)} scraper record(s) from {raw_data_path}"
            )
        except Exception as e:
            state.logs.append(f"Could not load raw_data_path ({e}) — falling back to HTML parse")

    state.logs.append(f"Starting extraction for {len(html_paths)} HTML file(s)")

    parsed_pages: List[ParsedPage] = []
    for path in html_paths:
        print(f"Extracting: {path}")
        scraper_rec = scraper_record_map.get(path, {})

        page = extractor.extract(
            html_path=path,
            url=url_map.get(path) or scraper_rec.get("url"),
            structured_data=scraper_rec.get("structured_data"),
            page_type=scraper_rec.get("page_type"),
            title=scraper_rec.get("title"),
        )
        parsed_pages.append(page)
        state.logs.append(
            f"{path} → method={page.extraction_method}, "
            f"records={len(page.records)}, success={page.success}"
        )

    state.parsed_pages = [p.model_dump() for p in parsed_pages]
    state.logs.append("Extraction complete")

    df = build_dataframe(parsed_pages)
    state.logs.append(f"DataFrame built: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"\nDataFrame shape: {df.shape}")
    print(df.head())

    state.dataframe_path = save_dataframe(df)
    state.logs.append(f"DataFrame saved: {state.dataframe_path}")

    state.metadata_path = save_metadata(state)
    state.logs.append(f"Metadata saved: {state.metadata_path}")

    return state


# ---------------------------------------------------------------------------
# ADK tool wrapper
# ---------------------------------------------------------------------------

def extract_from_html_file(html_path: str, url: Optional[str] = None) -> dict:
    """ADK-compatible tool. Parses a single HTML file and returns structured records."""
    extractor = HTMLExtractor()
    result = extractor.extract(html_path, url=url)
    return result.model_dump()


# ---------------------------------------------------------------------------
# ADK agent
# ---------------------------------------------------------------------------

extraction_agent = Agent(
    model="gemini-2.0-flash",
    name="extraction_agent",
    description=(
        "Reads scraper output records, extracts structured content using a "
        "three-tier strategy (pre-extracted structured_data → product card "
        "re-parse → generic HTML parse), and stores results as a tidy CSV."
    ),
    instruction="""
    You are a data extraction agent. Your job:
    1. Accept a list of HTML file paths and optionally a raw_data_path JSON.
    2. For each file, use the highest-quality extraction strategy available:
       - structured_data from the scraper (best)
       - product card re-parse from HTML
       - generic paragraph/table extraction (fallback)
    3. Report which strategy was used per file and how many records were extracted.
    4. Confirm the final DataFrame path and its shape.

    Rules:
    - Never fabricate data.
    - If a file fails, log the error and continue with the rest.
    - Always prefer structured_data over raw HTML re-parsing.
    """,
    tools=[extract_from_html_file],
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_html_paths = [
        "storage/raw_html/example1.html",
    ]
    sample_url_map = {
        "storage/raw_html/example1.html": "https://books.toscrape.com",
    }
    final_state = execute_extraction(
        sample_html_paths,
        url_map=sample_url_map,
        raw_data_path="storage/raw/raw_data_example.json",
    )
    print(final_state.model_dump_json(indent=2))