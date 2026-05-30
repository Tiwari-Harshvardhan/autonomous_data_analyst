import os
import json
import uuid
import asyncio
import requests
import numpy as np

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from google.adk.agents import Agent
from google.adk.tools import google_search


def _sanitize_for_json(obj: Any) -> Any:
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
# Universal scraper
# ---------------------------------------------------------------------------

class UniversalScraper:
    """
    Scrapes a URL using either requests+BeautifulSoup (static) or Playwright
    (dynamic). The mode is chosen automatically via heuristics on the raw HTML.
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36"
        )
    }

    # Strings in the raw HTML that suggest the page requires JS to render.
    DYNAMIC_MARKERS = [
        "react", "next.js", "__next", "vue", "angular",
        "hydration", "webpack", "application/json",
    ]

    async def scrape(self, url: str) -> ScraperResult:
        is_dynamic = await self._is_dynamic(url)
        mode = "dynamic" if is_dynamic else "static"
        print(f"[{mode}] {url}")
        scrape_fn = self._scrape_dynamic if is_dynamic else self._scrape_static
        return await scrape_fn(url)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    async def _is_dynamic(self, url: str) -> bool:
        """
        Score the raw HTML against known JS-framework markers.
        A score >= 2 means we use Playwright. On any network error we
        default to dynamic, since a broken static fetch usually means JS.
        """
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=10)
            html = resp.text.lower()
            score = sum(1 for m in self.DYNAMIC_MARKERS if m in html)
            if len(html) < 5000:
                score += 2  # tiny body → almost certainly a JS shell
            return score >= 2
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Static path (requests + BeautifulSoup)
    # ------------------------------------------------------------------

    async def _scrape_static(self, url: str) -> ScraperResult:
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string.strip() if soup.title and soup.title.string else None
            text = "\n".join(p.get_text(strip=True) for p in soup.find_all("p"))

            return ScraperResult(
                url=url,
                scraping_mode="static",
                title=title,
                extracted_text=text[:10_000],
                html_path=self._save_html(resp.text),
                metadata_path=self._save_metadata(url, "static", title),
            )
        except Exception as e:
            return ScraperResult(url=url, scraping_mode="static", success=False, error=str(e))

    # ------------------------------------------------------------------
    # Dynamic path (Playwright)
    # ------------------------------------------------------------------

    async def _scrape_dynamic(self, url: str) -> ScraperResult:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(user_agent=self.HEADERS["User-Agent"])

                await page.goto(url, timeout=60_000, wait_until="networkidle")
                await self._scroll_to_bottom(page)

                html = await page.content()
                title = await page.title()
                soup = BeautifulSoup(html, "html.parser")
                text = "\n".join(p.get_text(strip=True) for p in soup.find_all("p"))

                result = ScraperResult(
                    url=url,
                    scraping_mode="dynamic",
                    title=title,
                    extracted_text=text[:10_000],
                    html_path=self._save_html(html),
                    screenshot_path=await self._save_screenshot(page),
                    metadata_path=self._save_metadata(url, "dynamic", title),
                )
                await browser.close()
                return result

        except Exception as e:
            return ScraperResult(url=url, scraping_mode="dynamic", success=False, error=str(e))

    # ------------------------------------------------------------------
    # Infinite scroll
    # ------------------------------------------------------------------

    async def _scroll_to_bottom(self, page, max_rounds: int = 20) -> None:
        """
        Scroll until the document height stops growing or we hit max_rounds.
        """
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
            json.dump({"url": url, "mode": mode, "title": title}, f, indent=2, ensure_ascii=False)
        return path


# ---------------------------------------------------------------------------
# Persistence helpers (workflow-level)
# ---------------------------------------------------------------------------

def save_raw_data(scraped_results: List[ScraperResult]) -> str:
    """Serializes a list of ScraperResults to JSON and writes to disk."""
    path = os.path.join(RAW_DATA_DIR, f"raw_data_{uuid.uuid4()}.json")
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
#
# ADK tools must be plain async functions, not classes. This thin wrapper
# exposes the scraper in a form the agent can call.
# ---------------------------------------------------------------------------

async def scrape_url(url: str) -> dict:
    """
    ADK-compatible tool. Scrapes a single URL and returns the result as a dict.
    """
    scraper = UniversalScraper()
    result = await scraper.scrape(url)
    return _sanitize_for_json(result.model_dump())


# ---------------------------------------------------------------------------
# Data collection pipeline
# ---------------------------------------------------------------------------

async def execute_data_collection(state: WorkflowState) -> WorkflowState:
    state.logs.append("Started data collection workflow")

    # Step 1: discover URLs (extract from query if present, otherwise use defaults)
    import re
    urls = re.findall(r'https?://[^\s]+', state.user_query)
    if urls:
        discovered_urls = urls
        state.logs.append(f"Extracted URL(s) from query: {discovered_urls}")
    else:
        discovered_urls = [
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://www.ibm.com/topics/machine-learning",
        ]
        state.logs.append("No URL found in query; using default machine learning URLs")
    state.source_urls = discovered_urls
    state.logs.append(f"Discovered {len(discovered_urls)} URLs")

    # Step 2: scrape all URLs concurrently
    scraper = UniversalScraper()
    scraped_results: List[ScraperResult] = await asyncio.gather(
        *(scraper.scrape(url) for url in discovered_urls)
    )
    state.logs.append("Completed web scraping")

    # Step 3: persist raw data
    state.raw_data_path = save_raw_data(scraped_results)
    state.logs.append(f"Raw data saved: {state.raw_data_path}")

    # Step 4: store a small preview in the state for quick inspection
    state.raw_data_preview = [r.model_dump() for r in scraped_results[:2]]

    # Step 5: persist workflow metadata
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
        "Autonomous agent that discovers, scrapes, structures, "
        "and stores public web data."
    ),
    instruction="""
    Your responsibilities:
    1. Discover relevant public datasets and web sources via google_search.
    2. Scrape each discovered URL using the scrape_url tool.
    3. Structure and summarize the collected data.
    4. Report what was saved and where.

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
    state = WorkflowState(
        user_query="Collect machine learning related data from public web sources"
    )
    final_state = asyncio.run(execute_data_collection(state))
    print(final_state.model_dump_json(indent=2))


