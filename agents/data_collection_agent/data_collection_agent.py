import os
import json
import re
import uuid
import asyncio
import requests
import numpy as np

from bs4 import BeautifulSoup, Tag
from playwright.async_api import async_playwright
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field

from google.adk.agents import Agent
from google.adk.tools import google_search


# ---------------------------------------------------------------------------
# JSON sanitizer
# ---------------------------------------------------------------------------

def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    elif isinstance(obj, np.number):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    elif isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Storage setup
# ---------------------------------------------------------------------------

BASE_STORAGE_DIR = "storage"
RAW_DATA_DIR   = os.path.join(BASE_STORAGE_DIR, "raw")
LOG_DIR        = os.path.join(BASE_STORAGE_DIR, "logs")
METADATA_DIR   = os.path.join(BASE_STORAGE_DIR, "metadata")
RAW_HTML_DIR   = os.path.join(BASE_STORAGE_DIR, "raw_html")
SCREENSHOT_DIR = os.path.join(BASE_STORAGE_DIR, "screenshots")

for d in (RAW_DATA_DIR, LOG_DIR, METADATA_DIR, RAW_HTML_DIR, SCREENSHOT_DIR):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ScraperResult(BaseModel):
    url: str
    scraping_mode: str
    title: Optional[str] = None
    extracted_text: Optional[str] = None
    structured_data: Optional[List[Dict[str, Any]]] = None
    page_type: Optional[str] = None
    html_path: Optional[str] = None
    screenshot_path: Optional[str] = None
    metadata_path: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class WorkflowState(BaseModel):
    user_query: str
    source_urls: List[str] = Field(default_factory=list)
    raw_data_path: Optional[str] = None
    metadata_path: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    current_stage: str = "data_collection"
    raw_data_preview: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Page classification helpers (used by the scraper to tag each page)
# ---------------------------------------------------------------------------

_PRODUCT_SIGNALS = [
    "product", "price", "rating", "add-to-cart", "buy", "shop",
    "item", "catalogue", "catalog", "listing", "card",
]
_TABLE_SIGNALS = ["<table", "thead", "tbody", "<tr", "<td"]


def _classify_page(html: str, soup: BeautifulSoup) -> str:
    """
    Heuristically classifies a page into one of four types so the scraper
    can choose the right extraction strategy.

      product_listing — repeated card/grid elements with names and prices
      table_heavy     — page dominated by HTML tables
      article         — structured prose with headings and paragraphs
      general         — everything else
    """
    lower_html = html.lower()

    product_score = sum(1 for s in _PRODUCT_SIGNALS if s in lower_html)
    if product_score >= 3 or soup.find_all(class_=re.compile(r"product|item|card", re.I)):
        return "product_listing"

    table_score = sum(1 for s in _TABLE_SIGNALS if s in lower_html)
    if table_score >= 2 or len(soup.find_all("table")) >= 2:
        return "table_heavy"

    if soup.find_all(["h1", "h2", "h3"]) and soup.find_all("p"):
        return "article"

    return "general"


# ---------------------------------------------------------------------------
# Layout-aware extraction helpers
# ---------------------------------------------------------------------------

def _fix_encoding(text: str) -> str:
    """Repairs mojibake: 'â£51.77' → '£51.77'"""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def _clean_price(raw: str) -> str:
    """Strips concatenated availability text from a price string."""
    match = re.search(r"[\$\£\€\₹\¥]?\s*\d[\d,]*\.?\d*", raw)
    return match.group(0).strip() if match else raw


def extract_layout_aware(soup: BeautifulSoup) -> str:
    """
    Walks the document in order and captures headings, paragraphs,
    list items, and table rows — preserving semantic structure as plain text.
    """
    lines: List[str] = []
    seen: set = set()

    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr"]):
        if tag in seen:
            continue
        seen.add(tag)
        text = tag.get_text(separator=" ", strip=True)
        if not text:
            continue
        if tag.name.startswith("h"):
            lines.append(f"{'#' * int(tag.name[1])} {text}")
        elif tag.name == "tr":
            cells = [td.get_text(strip=True) for td in tag.find_all(["td", "th"])]
            if cells:
                lines.append(" | ".join(cells))
        else:
            lines.append(text)

    return "\n".join(lines)


def extract_product_cards(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """
    Extracts product records from repeated card/grid elements.
    Targets Books to Scrape's <article class="product_pod"> and generic grids.
    """
    products: List[Dict[str, Any]] = []

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

            # Name — Books to Scrape stores real title in the <a> title attr
            name_tag = (
                card.find("h3") or card.find("h2") or card.find("h4")
                or card.find("strong")
                or card.find(class_=re.compile(r"name|title", re.I))
            )
            if name_tag:
                a = name_tag.find("a")
                record["name"] = (
                    a["title"] if a and a.get("title")
                    else name_tag.get_text(strip=True)
                )

            # Price — target the specific price element, not the container div
            price_tag = (
                card.find("p", class_=re.compile(r"price_color", re.I))
                or card.find(class_=re.compile(
                    r"^price$|price[_-](?!container|box|wrap|product)", re.I
                ))
            )
            if price_tag:
                raw_price = _fix_encoding(price_tag.get_text(separator="", strip=True))
                record["price"] = _clean_price(raw_price)

            # Rating — Books to Scrape uses <p class="star-rating Three">
            rating_tag = card.find(class_=re.compile(r"star-rating|rating", re.I))
            if rating_tag:
                classes     = " ".join(rating_tag.get("class", []))
                word_rating = re.search(r"\b(One|Two|Three|Four|Five)\b", classes, re.I)
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
                record["product_url"] = link["href"]

            if record:
                products.append(record)

        if products:
            break

    return products


def extract_tables_as_records(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Converts HTML tables into a list of row dicts keyed by header names."""
    records: List[Dict[str, Any]] = []
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
    return records


# ---------------------------------------------------------------------------
# LLM fallback parser
# ---------------------------------------------------------------------------

_LLM_EXTRACTION_PROMPT = """
You are a data extraction specialist inside a web scraping pipeline.

You will receive a compact structural map of an HTML page — tag names, CSS
class names, and a sample of the visible text in each node.

Your task:
  Identify which fields contain meaningful structured data (product name,
  price, rating, category, description, date, author, or any other repeated
  field) and return a JSON array of extracted records.

Rules:
  - Return ONLY a JSON array. No markdown fences, no text outside the array.
  - Each element is one record (one product, one table row, one article).
  - Use clean snake_case key names (e.g. "product_name", "price", "rating").
  - Only include values that appear verbatim in the structural map.
  - If no structured data is present, return an empty array: []

Structural map:
{structural_map}
"""


def _build_structural_map(soup: BeautifulSoup, max_nodes: int = 60) -> str:
    lines: List[str] = []
    for i, tag in enumerate(soup.find_all(True)):
        if i >= max_nodes:
            break
        if not isinstance(tag, Tag):
            continue
        classes = " ".join(tag.get("class") or [])
        text    = tag.get_text(separator=" ", strip=True)[:80]
        if text:
            lines.append(f"<{tag.name} class='{classes}'> {text}")
    return "\n".join(lines)


def _call_llm_for_extraction(structural_map: str) -> List[Dict[str, Any]]:
    """Sends structural map to Gemini and returns parsed JSON records."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("[LLM extractor] GEMINI_API_KEY not set — skipping LLM fallback.")
        return []

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    prompt = _LLM_EXTRACTION_PROMPT.format(structural_map=structural_map)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
    }

    try:
        resp = requests.post(url, json=body, timeout=30)
        resp.raise_for_status()
        raw = (
            resp.json()
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )
        raw    = re.sub(r"^```json\s*", "", raw)
        raw    = re.sub(r"\s*```$",     "", raw)
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception as e:
        print(f"[LLM extractor] Failed: {e}")

    return []


# ===========================================================================
# Universal Scraper
# ===========================================================================

class UniversalScraper:
    """
    Scrapes a URL using requests+BeautifulSoup (static) or Playwright (dynamic).

    Extraction is page-type aware:
      article / general  → extract_layout_aware (structured plain text)
      table_heavy        → extract_tables_as_records (list of row dicts)
      product_listing    → extract_product_cards, with LLM fallback if
                           fewer than 3 cards are found programmatically
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36"
        )
    }

    DYNAMIC_MARKERS = [
        "react", "next.js", "__next", "vue", "angular",
        "hydration", "webpack", "application/json",
    ]

    async def scrape(self, url: str) -> ScraperResult:
        is_dynamic = await self._is_dynamic(url)
        mode = "dynamic" if is_dynamic else "static"
        print(f"[{mode}] {url}")
        return await (self._scrape_dynamic if is_dynamic else self._scrape_static)(url)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    async def _is_dynamic(self, url: str) -> bool:
        try:
            resp  = requests.get(url, headers=self.HEADERS, timeout=10)
            html  = resp.text.lower()
            score = sum(1 for m in self.DYNAMIC_MARKERS if m in html)
            if len(html) < 5000:
                score += 2
            return score >= 2
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Extraction dispatcher (shared by static and dynamic paths)
    # ------------------------------------------------------------------

    def _extract(
        self, html: str, soup: BeautifulSoup
    ) -> Tuple[str, Optional[List[Dict[str, Any]]], str]:
        """
        Classifies the page and runs the appropriate extractor.

        Returns:
          extracted_text  : plain-text (always populated for pipeline compat)
          structured_data : list of records for product/table pages, else None
          page_type       : classification label
        """
        page_type       = _classify_page(html, soup)
        structured_data: Optional[List[Dict[str, Any]]] = None

        if page_type == "product_listing":
            records = extract_product_cards(soup)
            if len(records) < 3:
                print(f"[extractor] Only {len(records)} card(s) — trying LLM fallback")
                records = _call_llm_for_extraction(_build_structural_map(soup))
            structured_data = records if records else None
            extracted_text  = "\n".join(
                " | ".join(f"{k}: {v}" for k, v in r.items())
                for r in (records or [])
            )[:10_000]

        elif page_type == "table_heavy":
            records         = extract_tables_as_records(soup)
            structured_data = records if records else None
            extracted_text  = "\n".join(
                " | ".join(f"{k}: {v}" for k, v in r.items())
                for r in (records or [])
            )[:10_000]

        else:
            extracted_text = extract_layout_aware(soup)[:10_000]

        return extracted_text, structured_data, page_type

    # ------------------------------------------------------------------
    # Static path
    # ------------------------------------------------------------------

    async def _scrape_static(self, url: str) -> ScraperResult:
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            resp.raise_for_status()

            soup  = BeautifulSoup(resp.text, "html.parser")
            title = (
                soup.title.string.strip()
                if soup.title and soup.title.string else None
            )
            extracted_text, structured_data, page_type = self._extract(resp.text, soup)

            return ScraperResult(
                url=url,
                scraping_mode="static",
                title=title,
                extracted_text=extracted_text,
                structured_data=structured_data,
                page_type=page_type,
                html_path=self._save_html(resp.text),
                metadata_path=self._save_metadata(url, "static", title),
            )
        except Exception as e:
            return ScraperResult(
                url=url, scraping_mode="static", success=False, error=str(e)
            )

    # ------------------------------------------------------------------
    # Dynamic path
    # ------------------------------------------------------------------

    async def _scrape_dynamic(self, url: str) -> ScraperResult:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page    = await browser.new_page(user_agent=self.HEADERS["User-Agent"])

                await page.goto(url, timeout=60_000, wait_until="networkidle")
                await self._scroll_to_bottom(page)

                html  = await page.content()
                title = await page.title()
                soup  = BeautifulSoup(html, "html.parser")
                extracted_text, structured_data, page_type = self._extract(html, soup)

                result = ScraperResult(
                    url=url,
                    scraping_mode="dynamic",
                    title=title,
                    extracted_text=extracted_text,
                    structured_data=structured_data,
                    page_type=page_type,
                    html_path=self._save_html(html),
                    screenshot_path=await self._save_screenshot(page),
                    metadata_path=self._save_metadata(url, "dynamic", title),
                )
                await browser.close()
                return result

        except Exception as e:
            return ScraperResult(
                url=url, scraping_mode="dynamic", success=False, error=str(e)
            )

    # ------------------------------------------------------------------
    # Infinite scroll
    # ------------------------------------------------------------------

    async def _scroll_to_bottom(self, page, max_rounds: int = 20) -> None:
        prev_height = await page.evaluate("document.body.scrollHeight")
        for _ in range(max_rounds):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)
            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == prev_height:
                break
            prev_height = new_height

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _unique_path(self, directory: str, extension: str) -> str:
        return os.path.join(directory, f"{uuid.uuid4()}.{extension}")

    def _save_html(self, html: str) -> str:
        path = self._unique_path(RAW_HTML_DIR, "html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

    async def _save_screenshot(self, page) -> str:
        path = self._unique_path(SCREENSHOT_DIR, "png")
        await page.screenshot(path=path, full_page=True)
        return path

    def _save_metadata(self, url: str, mode: str, title: Optional[str]) -> str:
        path = self._unique_path(METADATA_DIR, "json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"url": url, "mode": mode, "title": title},
                f, indent=2, ensure_ascii=False,
            )
        return path


# ---------------------------------------------------------------------------
# Workflow-level persistence
# ---------------------------------------------------------------------------

def save_raw_data(scraped_results: List[ScraperResult]) -> str:
    """Serializes ScraperResults to JSON. Path is read by extraction agent."""
    path    = os.path.join(RAW_DATA_DIR, f"raw_data_{uuid.uuid4()}.json")
    records = [r.model_dump() for r in scraped_results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(records), f, indent=2, ensure_ascii=False)
    return path


def save_workflow_metadata(state: WorkflowState) -> str:
    path = os.path.join(METADATA_DIR, f"metadata_{uuid.uuid4()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(state.model_dump()), f, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# ADK tool wrapper
# ---------------------------------------------------------------------------

async def scrape_url(url: str) -> dict:
    """ADK tool: scrapes a single URL and returns a JSON-safe ScraperResult dict."""
    scraper = UniversalScraper()
    result  = await scraper.scrape(url)
    return _sanitize_for_json(result.model_dump())


# ---------------------------------------------------------------------------
# Data collection pipeline
# ---------------------------------------------------------------------------

async def execute_data_collection(state: WorkflowState) -> WorkflowState:
    state.logs.append("Started data collection workflow")

    urls_in_query = re.findall(r"https?://[^\s]+", state.user_query)
    if urls_in_query:
        discovered_urls = urls_in_query
        state.logs.append(f"Extracted URL(s) from query: {discovered_urls}")
    else:
        discovered_urls = [
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://www.ibm.com/topics/machine-learning",
        ]
        state.logs.append("No URL found in query — using default sources")

    state.source_urls = discovered_urls
    state.logs.append(f"Discovered {len(discovered_urls)} URLs")

    scraper = UniversalScraper()
    scraped_results: List[ScraperResult] = list(
        await asyncio.gather(*(scraper.scrape(url) for url in discovered_urls))
    )
    state.logs.append("Completed web scraping")

    for r in scraped_results:
        state.logs.append(
            f"  {r.url} → page_type={r.page_type}, "
            f"structured_records={len(r.structured_data or [])}, "
            f"success={r.success}"
        )

    state.raw_data_path = save_raw_data(scraped_results)
    state.logs.append(f"Raw data saved: {state.raw_data_path}")

    state.raw_data_preview = [
        _sanitize_for_json(r.model_dump()) for r in scraped_results[:2]
    ]
    state.metadata_path = save_workflow_metadata(state)
    state.logs.append(f"Metadata saved: {state.metadata_path}")

    print("Data collection complete.")
    return state


# ---------------------------------------------------------------------------
# ADK agent
# ---------------------------------------------------------------------------

data_collection_agent = Agent(
    model="gemini-2.0-flash",
    name="data_collection_agent",
    description=(
        "Discovers and scrapes public web pages. Automatically detects page type "
        "(product listing, table-heavy, article, general) and applies the right "
        "extraction strategy, with an LLM fallback for complex product grids."
    ),
    instruction="""
    Your responsibilities:
    1. If the user provides a URL, scrape it directly using scrape_url.
    2. If no URL is given, use google_search to find relevant pages, then
       scrape the top results with scrape_url.
    3. After scraping, report for each URL:
       - The page_type detected
       - How many structured_data records were extracted
       - The raw_data_path where results were saved
    4. If structured_data is present, show the first 3 records so the user
       can verify the extraction looks correct.

    Constraints:
    - Never fabricate data.
    - Respect robots.txt.
    - Never attempt to bypass CAPTCHAs.
    - Always preserve source attribution.
    """,
    tools=[google_search, scrape_url],
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Scrape product listings from https://books.toscrape.com"
    )

    async def _run():
        response = await data_collection_agent.run_async(query)
        print(response)

    asyncio.run(_run())