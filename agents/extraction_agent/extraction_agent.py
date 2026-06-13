import os
import json
import uuid
import re
import requests
import pandas as pd

from bs4 import BeautifulSoup, Tag
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field

from google.adk.agents import Agent


# Storage setup
BASE_STORAGE_DIR = "storage"
PARSED_DATA_DIR  = os.path.join(BASE_STORAGE_DIR, "parsed")
DATAFRAME_DIR    = os.path.join(BASE_STORAGE_DIR, "dataframes")
METADATA_DIR     = os.path.join(BASE_STORAGE_DIR, "metadata")

for d in (PARSED_DATA_DIR, DATAFRAME_DIR, METADATA_DIR):
    os.makedirs(d, exist_ok=True)


# Data models
class ExtractionResult(BaseModel):
    """
    Represents the outcome of one extraction attempt on a single page.
    Carries both the records and a confidence score so the orchestrator
    can decide whether to trigger a more expensive fallback.
    """
    source_html_path: str
    url: Optional[str] = None
    title: Optional[str] = None
    page_type: Optional[str] = None           # classified page archetype
    extraction_method: str = "html_parse"     # which strategy succeeded
    confidence: float = 0.0                   # 0.0 – 1.0
    records: List[Dict[str, Any]] = Field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class ExtractionState(BaseModel):
    """Tracks the full extraction pipeline for a batch of raw scraper records."""
    html_paths: List[str]
    parsed_pages: List[Dict[str, Any]] = Field(default_factory=list)
    dataframe_path: Optional[str] = None
    metadata_path: Optional[str] = None
    schema_inference: Optional[Dict[str, str]] = None  # column → semantic type
    logs: List[str] = Field(default_factory=list)


# Encoding fix
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
    return {
        k: (_fix_encoding(v) if isinstance(v, str) else v)
        for k, v in record.items()
    }


def _clean_price(raw: str) -> str:
    """
    Extracts just the numeric price from a string that may contain
    concatenated availability text, e.g. '£51.77In stockAdd to basket'.
    """
    match = re.search(r"[\$\£\€\₹\¥]?\s*\d[\d,]*\.?\d*", raw)
    return match.group(0).strip() if match else raw


# CHANGE 1 — Page-Type Classification
# Schema.org types mapped to our internal page archetypes
_SCHEMA_TYPE_MAP = {
    "Product":        "ecommerce",
    "ItemList":       "ecommerce",
    "Offer":          "ecommerce",
    "NewsArticle":    "article",
    "Article":        "article",
    "BlogPosting":    "article",
    "JobPosting":     "listing",
    "RealEstateListing": "listing",
    "Recipe":         "listing",
    "FAQPage":        "documentation",
    "TechArticle":    "documentation",
    "Dataset":        "table_data",
    "Table":          "table_data",
}


def classify_page_type(soup: BeautifulSoup, html: str) -> str:
    """
    Classifies the page into one of six archetypes using three signals
    in priority order: JSON-LD schema, microdata itemtype, then DOM heuristics.

    Returns one of: ecommerce | table_data | article | documentation |
                    listing | unknown
    """
    # --- Signal 1: JSON-LD schema.org type ---
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            schema_type = (
                data.get("@type", "")
                if isinstance(data, dict)
                else data[0].get("@type", "") if isinstance(data, list) else ""
            )
            archetype = _SCHEMA_TYPE_MAP.get(schema_type)
            if archetype:
                return archetype
        except (json.JSONDecodeError, IndexError, KeyError):
            continue

    # --- Signal 2: microdata itemtype ---
    for tag in soup.find_all(attrs={"itemtype": True}):
        itemtype = tag.get("itemtype", "")
        for schema_name, archetype in _SCHEMA_TYPE_MAP.items():
            if schema_name.lower() in itemtype.lower():
                return archetype

    # --- Signal 3: DOM heuristics ---
    table_count   = len(soup.find_all("table"))
    article_count = len(soup.find_all("article"))
    product_count = len(soup.select(".product, .product-card, .product_pod, [class*='product']"))
    listing_count = len(soup.select(".listing, .job, .property, [class*='listing']"))
    code_count    = len(soup.find_all(["code", "pre"]))

    if product_count >= 3:
        return "ecommerce"
    if table_count >= 2:
        return "table_data"
    if listing_count >= 3:
        return "listing"
    if code_count >= 3:
        return "documentation"
    if article_count >= 1 and soup.find_all("p"):
        return "article"

    return "unknown"


# CHANGE 2 — JSON-LD Extraction
def extract_jsonld(soup: BeautifulSoup) -> Tuple[List[Dict[str, Any]], float]:
    """
    Extracts structured data from <script type="application/ld+json"> blocks.
    Returns (records, confidence). Confidence is 0.95 when records are found
    since JSON-LD is explicitly machine-readable data.
    """
    records: List[Dict[str, Any]] = []

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, dict):
                items = [data]
            elif isinstance(data, list):
                items = data
            else:
                continue

            for item in items:
                if not isinstance(item, dict):
                    continue
                # Flatten one level — skip context/type metadata keys
                record = {
                    k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                    for k, v in item.items()
                    if not k.startswith("@") and v is not None
                }
                if record:
                    records.append(_clean_record(record))

        except (json.JSONDecodeError, AttributeError):
            continue

    confidence = 0.95 if records else 0.0
    return records, confidence


# CHANGE 4 — Microdata Extraction
def extract_microdata(soup: BeautifulSoup) -> Tuple[List[Dict[str, Any]], float]:
    """
    Extracts data from HTML microdata attributes (itemprop / itemscope).
    Works on sites that use schema.org microdata instead of JSON-LD.
    """
    records: List[Dict[str, Any]] = []

    # Each itemscope element is one record
    for scope in soup.find_all(attrs={"itemscope": True}):
        record: Dict[str, Any] = {}
        for prop in scope.find_all(attrs={"itemprop": True}):
            name  = prop.get("itemprop", "").strip()
            value = (
                prop.get("content")
                or prop.get("datetime")
                or prop.get("href")
                or prop.get_text(strip=True)
            )
            if name and value:
                record[name] = _fix_encoding(str(value))
        if record:
            records.append(record)

    confidence = 0.90 if records else 0.0
    return records, confidence


# Repeating Block Detection (real & fake tables)
def detect_repeating_blocks(soup: BeautifulSoup) -> Tuple[List[Tag], float]:
    """
    Finds the dominant repeating DOM structure — the pattern that appears
    most frequently under a single parent. This handles both real product
    grids (<article>, <li class="product">) and fake tables (repeated
    <div class="row"> structures that visually look like tables).

    Returns (list_of_matching_tags, confidence).
    """
    # Count how many times each (parent_tag, child_tag, frozenset_of_classes)
    # pattern appears. The most frequent pattern is almost certainly the
    # repeating data block.
    pattern_counts: Dict[Tuple, List[Tag]] = {}

    for tag in soup.find_all(True):
        if not isinstance(tag, Tag):
            continue
        parent = tag.parent
        if not parent or not isinstance(parent, Tag):
            continue

        siblings = parent.find_all(tag.name, recursive=False)
        if len(siblings) < 5:
            continue

        classes = frozenset(tag.get("class") or [])
        key = (parent.name, tag.name, classes)
        if key not in pattern_counts:
            pattern_counts[key] = []
        if tag not in pattern_counts[key]:
            pattern_counts[key].append(tag)

    if not pattern_counts:
        return [], 0.0

    # Pick the most frequent pattern
    best_key  = max(pattern_counts, key=lambda k: len(pattern_counts[k]))
    best_tags = pattern_counts[best_key]

    # Confidence scales with repetition — 20+ occurrences → near certain
    confidence = min(0.85, 0.4 + len(best_tags) * 0.02)
    return best_tags, confidence


def extract_from_repeating_blocks(
    blocks: List[Tag],
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Dynamically extracts fields from each repeating block without assuming
    any specific field names. Captures: text nodes, links, numbers, dates,
    currency values, and images — labelled by their position/class.
    """
    records: List[Dict[str, Any]] = []
    currency_re = re.compile(r"[\$\£\€\₹\¥]\s*[\d,]+\.?\d*")
    date_re     = re.compile(r"\b\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}\b")

    for block in blocks:
        record: Dict[str, Any] = {}

        for child in block.find_all(True):
            if not isinstance(child, Tag):
                continue

            text = child.get_text(separator=" ", strip=True)
            if not text or len(text) > 300:
                continue

            # Build a stable field label from tag + class
            classes = "_".join(child.get("class") or [])
            label   = f"{child.name}_{classes}" if classes else child.name
            label   = re.sub(r"[^\w]", "_", label)[:40]

            # Classify the value type
            if currency_re.search(text):
                match = currency_re.search(text)
                record[f"price_{label}"] = match.group(0).strip()
            elif date_re.search(text):
                record[f"date_{label}"] = text
            elif child.name in ("a",) and child.get("href"):
                record[f"link_{label}"] = child["href"]
                if text:
                    record[f"text_{label}"] = _fix_encoding(text)
            elif child.name in ("img",):
                record[f"image_{label}"] = child.get("src") or child.get("data-src", "")
            elif text:
                record[f"text_{label}"] = _fix_encoding(text)

        # Deduplicate: if a child value also appears in a parent key, drop parent
        clean: Dict[str, Any] = {}
        values_seen = set()
        for k, v in record.items():
            if v not in values_seen:
                clean[k] = v
                values_seen.add(v)

        if clean:
            records.append(clean)

    confidence = min(0.82, 0.4 + len(records) * 0.02) if records else 0.0
    return records, confidence


# LLM Schema Inference (sends 3 examples, not full HTML)
_SCHEMA_INFERENCE_PROMPT = """
You are a data schema specialist inside a web scraping pipeline.

Below are 3 sample records extracted from repeated blocks on a webpage.
Each record is a flat dict of raw field labels and values.

Your task:
1. Infer the entity type (e.g. "book", "job_posting", "product", "article").
2. Map the raw field labels to clean, meaningful snake_case names.
3. Return ONLY a JSON object — no markdown fences, no explanation:

{
  "entity_type": "<type>",
  "field_mapping": {
    "<raw_label>": "<clean_name>",
    ...
  }
}

Only include fields that have clear meaning. Drop internal IDs, CSS artifacts,
and duplicated values. If a field's purpose is unclear, omit it.

Sample records:
SAMPLES
"""


def infer_schema_with_llm(
    sample_records: List[Dict[str, Any]],
) -> Optional[Dict[str, str]]:
    """
    Sends 3 sample records to Gemini and returns a field_mapping dict
    that renames raw extracted labels to clean semantic names.
    Returns None if the API call fails — extraction continues without renaming.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or not sample_records:
        return None

    samples = json.dumps(sample_records[:3], indent=2, ensure_ascii=False)
    prompt  = _SCHEMA_INFERENCE_PROMPT.replace("SAMPLES", samples)

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 512},
    }

    try:
        resp = requests.post(url, json=body, timeout=20)
        resp.raise_for_status()
        raw = (
            resp.json()
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$",     "", raw)
        parsed = json.loads(raw)
        return parsed.get("field_mapping", {})
    except Exception:
        return None


def apply_schema_mapping(
    records: List[Dict[str, Any]],
    mapping: Optional[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Renames raw field labels using the LLM-inferred schema mapping."""
    if not mapping:
        return records
    return [
        {mapping.get(k, k): v for k, v in r.items()}
        for r in records
    ]


# DOM Label Normalizer — strips HTML tag/class prefixes that leak into names
# Matches the artifact prefixes generated by extract_from_repeating_blocks():
#   text_span_country_population  →  country_population
#   text_h3_country_name          →  country_name
#   link_a_product_url            →  product_url   (link cols kept for now)
_DOM_PREFIX_RE = re.compile(
    r"^(?:text|link|price|date|image)_"         # value-type prefix
    r"(?:span|div|p|h\d|li|td|th|a|img|ul|ol|"  # html tag
    r"strong|em|small|b|i|label|header|footer|"   # more tags
    r"section|article|nav|main|aside)_?",          # structural tags
    re.I,
)


def _normalize_dom_labels(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Strips HTML DOM artifact prefixes from column names that were not
    (or only partially) renamed by the LLM schema inference step.

    Always runs as a safety fallback even when LLM renaming succeeded,
    so that any remaining raw DOM labels are cleaned up.

    Examples:
      'text_span_country_population' → 'country_population'
      'text_h3_country_name'         → 'country_name'
      'price_p_price_color'          → 'price_color'
      'image_img_product_img'        → 'product_img'
    """
    cleaned_records: List[Dict[str, Any]] = []
    for record in records:
        new_record: Dict[str, Any] = {}
        for key, val in record.items():
            new_key = _DOM_PREFIX_RE.sub("", key)
            # Normalise separators and collapse repeated underscores
            new_key = re.sub(r"[-]+", "_", new_key)
            new_key = re.sub(r"_+", "_", new_key).strip("_")
            new_key = new_key or key  # never produce an empty key
            new_record[new_key] = val
        cleaned_records.append(new_record)
    return cleaned_records


# Extraction Confidence Score tracker
def _make_result(
    html_path: str,
    url: Optional[str],
    title: Optional[str],
    page_type: str,
    method: str,
    records: List[Dict[str, Any]],
    confidence: float,
) -> ExtractionResult:
    return ExtractionResult(
        source_html_path=html_path,
        url=url,
        title=title,
        page_type=page_type,
        extraction_method=method,
        confidence=round(confidence, 3),
        records=records,
    )


# Main HTML Extractor — orchestrates all strategies in priority order
class HTMLExtractor:
    """
    Universal extractor. Tries extraction strategies in this priority order,
    stopping at the first one that exceeds the confidence threshold:

      1. Pre-extracted structured_data from the scraper  (confidence: 0.95)
      2. JSON-LD                                          (confidence: 0.95)
      3. Microdata (itemprop/itemscope)                   (confidence: 0.90)
      4. Real HTML tables                                 (confidence: 0.85)
      5. Repeating DOM block detection                    (confidence: 0.40–0.85)
         └─ LLM schema inference to rename raw fields
      6. LLM full-page fallback                           (confidence: 0.50)
      7. Generic layout-aware text extraction             (confidence: 0.20)
    """

    # Minimum confidence to accept a strategy's result without trying the next
    CONFIDENCE_THRESHOLD = 0.60

    def extract(
        self,
        html_path: str,
        url: Optional[str] = None,
        structured_data: Optional[List[Dict[str, Any]]] = None,
        page_type: Optional[str] = None,
        title: Optional[str] = None,
    ) -> ExtractionResult:

        # --- Strategy 1: pre-extracted structured_data from the scraper ---
        if structured_data:
            records = [_clean_record(r) for r in structured_data if r]
            # Clean price fields that may contain concatenated text
            for r in records:
                if "price" in r:
                    r["price"] = _clean_price(r["price"])
            if records:
                return _make_result(
                    html_path, url, title,
                    page_type or "unknown", "structured_data", records, 0.95
                )

        # --- Read HTML ---
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()
        except Exception as e:
            return ExtractionResult(
                source_html_path=html_path, success=False,
                error=f"Could not read HTML file: {e}"
            )

        soup       = BeautifulSoup(html, "html.parser")
        page_title = title or (
            soup.title.string.strip() if soup.title and soup.title.string else None
        )

        # Classify page type if not provided by the scraper
        detected_type = page_type or classify_page_type(soup, html)

        # --- Strategy 2: JSON-LD ---
        records, confidence = extract_jsonld(soup)
        if records and confidence >= self.CONFIDENCE_THRESHOLD:
            return _make_result(
                html_path, url, page_title,
                detected_type, "jsonld", records, confidence
            )

        # --- Strategy 3: Microdata ---
        records, confidence = extract_microdata(soup)
        if records and confidence >= self.CONFIDENCE_THRESHOLD:
            return _make_result(
                html_path, url, page_title,
                detected_type, "microdata", records, confidence
            )

        # --- Strategy 4: Real HTML tables ---
        records, confidence = self._extract_tables(soup)
        if records and confidence >= self.CONFIDENCE_THRESHOLD:
            return _make_result(
                html_path, url, page_title,
                detected_type, "html_tables", records, confidence
            )

        # --- Strategy 5: Repeating block detection + LLM schema inference ---
        blocks, block_confidence = detect_repeating_blocks(soup)
        if blocks and block_confidence >= 0.40:
            raw_records, confidence = extract_from_repeating_blocks(blocks)
            if raw_records:
                # Ask LLM to rename the raw dynamically-labelled fields
                mapping = infer_schema_with_llm(raw_records)
                records = apply_schema_mapping(raw_records, mapping)
                # Always run DOM normalizer as a safety fallback —
                # strips any remaining 'text_span_*' / 'text_h3_*' prefixes
                records = _normalize_dom_labels(records)
                if confidence >= self.CONFIDENCE_THRESHOLD:
                    return _make_result(
                        html_path, url, page_title,
                        detected_type, "repeating_blocks", records, confidence
                    )

        # --- Strategy 6: LLM full-page fallback ---
        records, confidence = self._llm_fallback(soup)
        if records:
            return _make_result(
                html_path, url, page_title,
                detected_type, "llm_fallback", records, confidence
            )

        # --- Strategy 7: Generic layout-aware text (last resort) ---
        records = self._extract_generic(soup, url, page_title)
        return _make_result(
            html_path, url, page_title,
            detected_type, "generic_text", records, 0.20
        )

    # Strategy 4 — HTML tables
    def _extract_tables(
        self, soup: BeautifulSoup
    ) -> Tuple[List[Dict[str, Any]], float]:
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

        confidence = min(0.85, 0.50 + len(records) * 0.01) if records else 0.0
        return records, confidence

    # Strategy 6 — LLM full-page fallback
    _LLM_EXTRACT_PROMPT = """
You are a data extraction specialist. Below is a compact structural map of a webpage.

Extract all meaningful structured data from it and return a JSON array of records.
Each record represents one entity (product, article, job, etc.).
Use clean snake_case field names. Return ONLY the JSON array — no markdown, no explanation.
If there is nothing structured to extract, return [].

Structural map:
{map}
"""

    def _build_structural_map(self, soup: BeautifulSoup, max_nodes: int = 60) -> str:
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

    def _llm_fallback(
        self, soup: BeautifulSoup
    ) -> Tuple[List[Dict[str, Any]], float]:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return [], 0.0

        structural_map = self._build_structural_map(soup)
        prompt = self._LLM_EXTRACT_PROMPT.replace("{map}", structural_map)
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={api_key}"
        )
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
            raw     = re.sub(r"^```json\s*", "", raw)
            raw     = re.sub(r"\s*```$",     "", raw)
            parsed  = json.loads(raw)
            records = parsed if isinstance(parsed, list) else []
            confidence = 0.55 if records else 0.0
            return [_clean_record(r) for r in records], confidence
        except Exception:
            return [], 0.0

    # Strategy 7 — Generic layout-aware text
    def _extract_generic(
        self,
        soup: BeautifulSoup,
        url: Optional[str],
        title: Optional[str],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        seen: set = set()

        for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
            if tag in seen:
                continue
            seen.add(tag)
            text = _fix_encoding(tag.get_text(separator=" ", strip=True))
            if text:
                records.append({
                    "tag":   tag.name,
                    "text":  text,
                    "url":   url,
                    "title": title,
                })

        return records


# DataFrame builder
# Columns that are never analytically useful regardless of content
_ALWAYS_DROP = {"extraction_method", "source_html_path", "page_title"}

# Column types that signal low analytical value even when populated
_URL_PATTERN = re.compile(r"https?://|/catalogue/|\.html$", re.I)


def build_dataframe(results: List[ExtractionResult]) -> pd.DataFrame:
    """
    Builds a tidy DataFrame from all extraction results.

    Drops:
    - Columns where all values are identical (zero variance)
    - Columns where every value is a URL path (not analytically useful)
    - Internal bookkeeping columns (extraction_method, source_html_path)
    - Columns with > 95% missing values
    """
    all_rows: List[Dict[str, Any]] = []

    for result in results:
        if not result.success or not result.records:
            continue
        for record in result.records:
            row = dict(record)
            row["_page_type"]   = result.page_type
            row["_confidence"]  = result.confidence
            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Drop always-useless columns
    df = df.drop(columns=[c for c in _ALWAYS_DROP if c in df.columns])

    # Drop zero-variance columns (all values identical)
    df = df.drop(columns=[
        c for c in df.columns
        if df[c].nunique(dropna=False) <= 1
    ])

    # Drop columns where > 95% values are URL paths
    url_cols = [
        c for c in df.columns
        if df[c].dtype == object
        and df[c].dropna().apply(lambda v: bool(_URL_PATTERN.search(str(v)))).mean() > 0.95
    ]
    df = df.drop(columns=url_cols)

    # Drop columns with > 95% missing values
    df = df.dropna(axis=1, thresh=max(1, int(len(df) * 0.05)))

    return df


# EDA hint: which columns are worth visualizing
def get_visualization_hints(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns metadata that tells the EDA and visualization agents which
    columns are worth plotting and how.

    Rules:
    - Numeric columns with > 1 unique value → histogram / KDE
    - Categorical columns with 2–20 unique values → bar chart
    - Categorical columns with > 20 unique values → skip (too many categories)
    - Columns with all-unique values (IDs, URLs) → skip
    """
    hints: Dict[str, Any] = {
        "numeric_plot":      [],
        "categorical_plot":  [],
        "skip":              [],
    }

    for col in df.columns:
        if col.startswith("_"):
            hints["skip"].append(col)
            continue

        n_unique = df[col].nunique(dropna=True)
        n_rows   = len(df)

        if pd.api.types.is_numeric_dtype(df[col]):
            if n_unique > 1:
                hints["numeric_plot"].append(col)
            else:
                hints["skip"].append(col)
        else:
            if 2 <= n_unique <= 20:
                hints["categorical_plot"].append(col)
            else:
                # Too many unique values to be useful as a bar chart
                hints["skip"].append(col)

    return hints


# Semantic Schema Inference
_SCHEMA_URL_RE      = re.compile(r"https?://|www\.", re.I)
_SCHEMA_EMAIL_RE    = re.compile(r"@\w+\.\w+")
_SCHEMA_CURRENCY_RE = re.compile(r"^[\$\£\€\₹\¥]\s*[\d,]+")
_SCHEMA_PCT_RE      = re.compile(r"[\d.]+\s*%$")
_SCHEMA_BOOL_VALS   = frozenset({"true", "false", "yes", "no", "1", "0", "t", "f"})


def _try_parse_datetime(series: pd.Series) -> bool:
    """
    Attempts to parse `series` as datetime values and returns True when
    at least 70% of values parse successfully.

    Suppresses the pandas UserWarning about format inference by trying
    ``format='mixed'`` first (pandas >= 2.0) and falling back gracefully.
    The warning itself is also caught with a warnings filter so it never
    surfaces in the pipeline logs.
    """
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        # format='mixed' not supported (pandas < 2.0) — fall back silently
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(series, errors="coerce")
    return parsed.notna().mean() > 0.7


def infer_semantic_schema(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infers a semantic type for every non-internal column using a combination
    of pandas dtype checks and value-level pattern matching.

    Returns a mapping  {column_name: semantic_type}  where semantic_type is
    one of:
      numeric | categorical | datetime | boolean | url | email |
      currency | percentage | identifier | text | unknown

    This schema is saved into the extraction metadata JSON under the key
    'schema_inference' so that downstream agents (cleaning, feature
    engineering, visualization) can use it for type-aware processing.
    """
    schema: Dict[str, str] = {}

    for col in df.columns:
        if col.startswith("_"):  # skip internal bookkeeping columns
            continue

        series = df[col].dropna()
        if series.empty:
            schema[col] = "unknown"
            continue

        # --- dtype-based checks (fast path) ---
        if pd.api.types.is_bool_dtype(df[col]):
            schema[col] = "boolean"
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            schema[col] = "numeric"
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            schema[col] = "datetime"
            continue

        # --- value-pattern checks on string columns ---
        sample = series.astype(str).str.strip()
        n = len(sample)

        if sample.str.lower().isin(_SCHEMA_BOOL_VALS).sum() / n > 0.8:
            schema[col] = "boolean"
        elif sample.apply(lambda v: bool(_SCHEMA_URL_RE.search(v))).sum() / n > 0.5:
            schema[col] = "url"
        elif sample.apply(lambda v: bool(_SCHEMA_EMAIL_RE.search(v))).sum() / n > 0.5:
            schema[col] = "email"
        elif sample.apply(lambda v: bool(_SCHEMA_CURRENCY_RE.match(v))).sum() / n > 0.5:
            schema[col] = "currency"
        elif sample.apply(lambda v: bool(_SCHEMA_PCT_RE.search(v))).sum() / n > 0.5:
            schema[col] = "percentage"
        elif _try_parse_datetime(series):
            schema[col] = "datetime"
        elif series.nunique() == len(series) and sample.str.len().mean() < 20:
            # every value is unique + short → likely an ID
            schema[col] = "identifier"
        elif series.nunique() <= 50:
            schema[col] = "categorical"
        else:
            schema[col] = "text"

    return schema


# Persistence helpers
def save_dataframe(df: pd.DataFrame) -> str:
    path = os.path.join(DATAFRAME_DIR, f"extracted_{uuid.uuid4()}.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def save_metadata(state: ExtractionState) -> str:
    path = os.path.join(METADATA_DIR, f"extraction_metadata_{uuid.uuid4()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.model_dump(), f, indent=2, ensure_ascii=False)
    return path


# Extraction pipeline — public interface unchanged for orchestrator compat
def execute_extraction(
    html_paths: List[str],
    url_map: Optional[Dict[str, str]] = None,
    raw_data_path: Optional[str] = None,
) -> ExtractionState:
    """
    Orchestrates extraction for a batch of HTML files.

    Args:
        html_paths:    Paths to saved HTML files.
        url_map:       {html_path: original_url}
        raw_data_path: Path to the scraper's raw JSON. When provided, uses
                       pre-extracted structured_data as the first-priority source.
    """
    url_map = url_map or {}
    state   = ExtractionState(html_paths=html_paths)
    extractor = HTMLExtractor()

    scraper_record_map: Dict[str, Dict[str, Any]] = {}
    if raw_data_path:
        try:
            with open(raw_data_path, "r", encoding="utf-8") as f:
                raw_records = json.load(f)
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

    results: List[ExtractionResult] = []
    for path in html_paths:
        print(f"Extracting: {path}")
        rec = scraper_record_map.get(path, {})

        result = extractor.extract(
            html_path=path,
            url=url_map.get(path) or rec.get("url"),
            structured_data=rec.get("structured_data"),
            page_type=rec.get("page_type"),
            title=rec.get("title"),
        )
        results.append(result)
        state.logs.append(
            f"{path} → type={result.page_type}, method={result.extraction_method}, "
            f"records={len(result.records)}, confidence={result.confidence:.2f}"
        )

    state.parsed_pages = [r.model_dump() for r in results]
    state.logs.append("Extraction complete")

    df = build_dataframe(results)
    hints = get_visualization_hints(df)
    state.logs.append(f"DataFrame: {df.shape[0]} rows × {df.shape[1]} cols")
    state.logs.append(f"Visualization hints: {hints}")
    print(f"\nDataFrame shape: {df.shape}")
    print(df.head())

    state.dataframe_path = save_dataframe(df)
    state.logs.append(f"DataFrame saved: {state.dataframe_path}")

    # Infer semantic schema and attach to state
    schema = infer_semantic_schema(df)
    state.schema_inference = schema
    state.logs.append(f"Schema inference: {schema}")

    state.metadata_path = save_metadata(state)
    state.logs.append(f"Metadata saved: {state.metadata_path}")

    return state


# ADK tool wrapper
def extract_from_html_file(html_path: str, url: Optional[str] = None) -> dict:
    """ADK-compatible tool. Extracts structured records from a single HTML file."""
    result = HTMLExtractor().extract(html_path, url=url)
    return result.model_dump()


# ADK agent
extraction_agent = Agent(
    model="gemini-2.0-flash",
    name="extraction_agent",
    description=(
        "Universal extraction agent. Classifies page type, then tries "
        "JSON-LD → microdata → HTML tables → repeating blocks (with LLM "
        "schema inference) → LLM fallback → generic text, stopping at the "
        "first strategy that exceeds a confidence threshold."
    ),
    instruction="""
    You are a data extraction agent. Your job:
    1. Accept HTML file paths and optionally a raw_data_path JSON.
    2. For each file, run the full extraction pipeline. It will automatically
       choose the best strategy.
    3. Report: page_type, extraction_method, confidence, and record count
       for each file.
    4. Report the final DataFrame shape and path.
    5. If confidence < 0.6 for any file, flag it and suggest the user
       verify that the extracted records look correct.

    Rules:
    - Never fabricate data.
    - If a file fails, log the error and continue.
    - Always prefer high-confidence structured sources (JSON-LD, microdata)
      over heuristic or LLM-based extraction.
    """,
    tools=[extract_from_html_file],
)


# Entry point
if __name__ == "__main__":
    sample_html_paths = ["storage/raw_html/example1.html"]
    sample_url_map    = {"storage/raw_html/example1.html": "https://books.toscrape.com"}
    final_state = execute_extraction(
        sample_html_paths,
        url_map=sample_url_map,
        raw_data_path="storage/raw/raw_data_example.json",
    )
    print(final_state.model_dump_json(indent=2))