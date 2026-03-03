"""
Stage 1: Data Preprocessing — Extract item_1a from 10-K JSON files,
clean text, split into paragraphs, and export master table.

Splitting strategy:
  1. Extract bold+italic titles from raw HTML (font tag or span with font-weight:700).
  2. Split cleaned text by detected title boundaries (title-based splitting).
  3. Fallback: if fewer than 3 titles detected, use double-newline splitting;
     sub-split paragraphs > 1500 chars on sentence boundaries.
  4. Merge bullet-point fragments that follow a title (short lines < 80 chars)
     with their preceding intro sentence.

Usage:
    python -m src.preprocess [--min-chars 100] [--data-dir data/10k_cleaned] [--raw-dir data/10k_raw]
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import Counter, defaultdict

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

from src.logger import setup_logger

logger = setup_logger(__name__)

# Character threshold for secondary sentence-boundary split
SECONDARY_SPLIT_CHARS = 1500
# Min consecutive bullets to merge back into intro
BULLET_MERGE_MIN = 2

# ---------------------------------------------------------------------------
# Step 1: Scan files & parse metadata
# ---------------------------------------------------------------------------

def scan_files(data_dir: str) -> list[dict]:
    """
    Scan all primary-document.html files under data/10k_raw.

    Directory structure:
        {data_dir}/{TICKER}/10-K/{accession}/primary-document.html

    Year is inferred from the accession number (first two digits after dash).

    Returns list of dicts: {'path': ..., 'ticker': ..., 'year': ..., 'accession': ...}
    """
    pattern = os.path.join(data_dir, "*", "10-K", "*", "primary-document.html")
    files = sorted(glob.glob(pattern))
    results = []
    for fp in files:
        parts = fp.replace("\\", "/").split("/")
        # parts: [..., data_dir_leaf, TICKER, '10-K', accession, 'primary-document.html']
        try:
            ticker = parts[-4]
            accession = parts[-2]
        except IndexError:
            logger.warning("Unexpected path structure, skipped: %s", fp)
            continue
        year_match = re.search(r"-(\d{2})-", accession)
        if year_match:
            yy = int(year_match.group(1))
            year = 2000 + yy if yy < 80 else 1900 + yy
        else:
            logger.warning("Cannot parse year from accession %s", accession)
            year = 0
        results.append({"path": fp, "ticker": ticker, "year": year, "accession": accession})
    logger.info("Scanned %d HTML files from %s", len(results), data_dir)
    return results


# ---------------------------------------------------------------------------
# Step 2: Extract item_1a + HTML titles
# ---------------------------------------------------------------------------


def _extract_item1a_text_from_html(html: str) -> str:
    """
    Extract the plain text of the Item 1A (Risk Factors) section from raw HTML.

    Finds the HTML substring between the largest Item 1A and Item 1B boundary,
    then converts to plain text using BeautifulSoup.
    """
    if not HAS_BS4:
        return ""

    # Find all Item 1A positions in HTML
    item1a_matches = list(re.finditer(r"Item\s*(?:&#xA0;|&nbsp;|\\xa0)?\s*1A", html, re.I))
    item1b_matches = list(re.finditer(r"Item\s*(?:&#xA0;|&nbsp;|\\xa0)?\s*1B", html, re.I))

    if not item1a_matches:
        item1a_matches = list(re.finditer(r"Risk\s+Factors?", html, re.I))
    if not item1a_matches:
        return ""

    best_start = item1a_matches[0].start()
    best_end = len(html)
    best_gap = 0

    for m1a in item1a_matches:
        start = m1a.start()
        for m1b in item1b_matches:
            if m1b.start() > start:
                gap = m1b.start() - start
                if gap > best_gap:
                    best_gap = gap
                    best_start = start
                    best_end = m1b.start()
                break

    html_chunk = html[best_start:best_end]
    chunk_soup = BeautifulSoup(html_chunk, "html.parser")
    text = chunk_soup.get_text(separator="\n")
    return text.strip()


def extract_item_1a(file_info: dict, raw_dir: str = "data/10k_raw") -> tuple[dict, str, list[str]]:
    """
    Read a primary-document.html file and return (metadata, item_1a_text, html_titles).
    The HTML path is taken directly from file_info['path'] (set by scan_files).
    Returns empty string / empty list if extraction fails.
    """
    html_path = file_info["path"]
    try:
        html = open(html_path, "r", encoding="utf-8", errors="ignore").read()
    except OSError as e:
        logger.error("Failed to read %s: %s", html_path, e)
        return file_info, "", []

    if not HAS_BS4:
        logger.error("beautifulsoup4 required but not installed.")
        return file_info, "", []

    # Extract item_1a plain text from HTML
    text = _extract_item1a_text_from_html(html)

    # Extract bold titles from the same HTML, then whitelist-filter against item_1a text
    titles: list[str] = []
    if text:
        raw_titles = extract_titles_from_html(html_path)
        titles = [t for t in raw_titles if t in text]
        if len(raw_titles) > len(titles):
            logger.debug("Filtered %d / %d HTML titles against item_1a text",
                         len(raw_titles) - len(titles), len(raw_titles))

    return file_info, text, titles


def extract_titles_from_html(html_path: str) -> list[str]:
    """
    Extract risk-factor section titles from raw HTML.

    Strategy:
    1. Find all occurrences of 'Item 1A' in the raw HTML string.
    2. Pick the occurrence followed by the largest gap before 'Item 1B'
       (this is the actual body section, not the Table of Contents line).
    3. Parse only the HTML substring in that range.
    4. Within it, find bold+italic elements in two formats:
       - Old: <font style="...font-style:italic;font-weight:bold...">
       - Modern: <span style="...font-weight:700/bold...">

    Returns a de-duplicated list of title strings (>= 10 chars, contains letters).
    """
    if not HAS_BS4:
        return []
    try:
        html = open(html_path, "r", encoding="utf-8", errors="ignore").read()
    except OSError:
        return []

    # ---- Step 1: locate Item 1A section in raw HTML ----------------------------
    # Find all 'Item 1A' in HTML (both TOC and body)
    item1a_matches = list(re.finditer(r"Item\s*(?:&#xA0;|&nbsp;|\\xa0)?\s*1A", html, re.I))
    item1b_matches = list(re.finditer(r"Item\s*(?:&#xA0;|&nbsp;|\\xa0)?\s*1B", html, re.I))

    if not item1a_matches:
        # Try looser pattern
        item1a_matches = list(re.finditer(r"Risk\s+Factors?", html, re.I))
    if not item1a_matches:
        return []

    # Pick the Item 1A occurrence followed by the largest gap to Item 1B
    best_start = item1a_matches[0].start()
    best_end = len(html)
    best_gap = 0

    for m1a in item1a_matches:
        start = m1a.start()
        # Find next Item 1B after this position
        for m1b in item1b_matches:
            if m1b.start() > start:
                gap = m1b.start() - start
                if gap > best_gap:
                    best_gap = gap
                    best_start = start
                    best_end = m1b.start()
                break

    # Extract the HTML substring for item 1A
    html_chunk = html[best_start:best_end]

    # ---- Step 2: parse and extract bold titles ---------------------------------
    chunk_soup = BeautifulSoup(html_chunk, "html.parser")

    BOLD_ITALIC_RE = re.compile(
        r"font-style\s*:\s*italic.*font-weight\s*:\s*bold"
        r"|font-weight\s*:\s*bold.*font-style\s*:\s*italic"
        r"|font-weight\s*:\s*(700|bold)",
        re.I,
    )

    titles = []
    seen = set()

    for el in chunk_soup.find_all(True):
        style = el.get("style", "")
        if not BOLD_ITALIC_RE.search(style):
            continue
        if el.name not in ("font", "span", "p", "div", "td"):
            continue
        el_text = el.get_text(" ", strip=True)
        if not el_text:
            continue
        # Must be short (titles are not paragraphs)
        if len(el_text) > 200:
            continue
        # Must contain ≥3 consecutive letters (filter out '1', '.', page numbers)
        if not re.search(r'[A-Za-z]{3,}', el_text):
            continue
        # Minimum title length
        if len(el_text.strip()) < 10:
            continue
        key = re.sub(r"\s+", " ", el_text).strip()
        if key not in seen:
            seen.add(key)
            titles.append(key)

    logger.debug("Extracted %d HTML titles from %s", len(titles), html_path)
    return titles




# ---------------------------------------------------------------------------
# Step 3: Clean text
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean raw item_1a text:
    - Remove HTML tags
    - Remove "Table of Contents" markers
    - Remove page-number-like standalone lines
    - Normalize whitespace
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove "Table of Contents" (case-insensitive)
    text = re.sub(r"(?i)table\s+of\s+contents", "", text)
    # Remove standalone page numbers (lines that are just digits)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    # Remove "Item 1A." header itself (often repeated)
    text = re.sub(r"(?i)item\s+1a\.?\s*risk\s+factors?\s*", "", text, count=1)
    # Collapse multiple spaces into one
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Step 4: Split into paragraphs (improved 4-step logic)
# ---------------------------------------------------------------------------

def _sentence_split(text: str, max_chars: int = SECONDARY_SPLIT_CHARS) -> list[str]:
    """
    Secondary split: break a long paragraph at sentence boundaries.
    Targets paragraphs > max_chars. Splits on sentence-ending punctuation
    followed by whitespace + uppercase letter (or end of string).
    """
    if len(text) <= max_chars:
        return [text]
    # Split on . ! ? followed by whitespace+capital or end-of-string
    pieces = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text)
    # Re-merge until each piece is < max_chars
    result = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > max_chars:
            result.append(current.strip())
            current = piece
        else:
            current = (current + " " + piece).strip() if current else piece
    if current:
        result.append(current.strip())
    return result


def _is_bullet(text: str) -> bool:
    """Heuristic: is this line a bullet-point fragment?"""
    stripped = text.strip()
    # Starts with bullet chars, dash, number+dot, or is very short
    return bool(re.match(r'^[•\-\*\u2013\u2022\u25cf]|^\d+[.)\s]', stripped)) or len(stripped) < 80


def _merge_bullets(paragraphs: list[str]) -> list[str]:
    """
    Step 4: Merge consecutive short bullet-point lines with their preceding
    intro sentence if there are BULLET_MERGE_MIN or more consecutive bullets.
    """
    if not paragraphs:
        return paragraphs
    result: list[str] = []
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        # Look ahead: count consecutive bullets after this paragraph
        j = i + 1
        while j < len(paragraphs) and _is_bullet(paragraphs[j]):
            j += 1
        bullet_count = j - (i + 1)
        if bullet_count >= BULLET_MERGE_MIN:
            # Merge intro + all bullets into one paragraph
            merged = "\n".join(paragraphs[i:j])
            result.append(merged.strip())
            i = j
        else:
            result.append(para)
            i += 1
    return result


def split_paragraphs(text: str, titles: list[str] | None = None) -> list[str]:
    """
    Improved 4-step paragraph splitting:

    Step 1: Title-based splitting — use extracted HTML titles as section
            boundaries, if 3 or more titles are available.
    Step 2: Double-newline fallback — used when titles < 3.
    Step 3: Secondary sentence-boundary split for paragraphs > 1500 chars.
    Step 4: Merge bullet-point fragments back into their intro sentence.

    Returns ALL paragraphs (no length filtering — done in caller).
    """
    paragraphs: list[str] = []
    use_title_split = bool(titles and len(titles) >= 3)

    if use_title_split:
        # Step 1: title-based splitting
        # Build a regex that matches any of the known titles
        escaped = [re.escape(t) for t in titles]
        title_pattern = re.compile(
            r"(?m)^\s*(" + "|".join(escaped) + r")\s*$",
            re.MULTILINE,
        )
        # Split while keeping the delimiter (title) as start of next section
        parts = title_pattern.split(text)
        # parts alternates: [pre-title-text, title, body, title, body, ...]
        # parts[0] = text before first title (intro); then pairs of (title, body)
        # Reconstruct: each section = "Title\n\nBody"
        if len(parts) <= 1:
            # Pattern didn't match in plain text — fall back
            use_title_split = False
        else:
            # First element is pre-title intro (may be empty)
            if parts[0].strip():
                paragraphs.extend(_sentence_split(parts[0].strip()))
            # Remaining: groups of (title, body)
            it = iter(parts[1:])
            for title_text, body in zip(it, it):
                section = (title_text.strip() + "\n\n" + body.strip()).strip()
                if section:
                    paragraphs.extend(_sentence_split(section))

    if not use_title_split:
        # Step 2: double-newline fallback
        raw = re.split(r"\n{2,}", text)
        for chunk in raw:
            chunk = chunk.strip()
            if not chunk:
                continue
            # Step 3: secondary split for oversized paragraphs
            paragraphs.extend(_sentence_split(chunk))

    # Step 4: merge bullet-point fragments
    paragraphs = [p for p in paragraphs if p.strip()]
    paragraphs = _merge_bullets(paragraphs)

    return paragraphs


# ---------------------------------------------------------------------------
# Step 5: Build paragraph records with unique IDs
# ---------------------------------------------------------------------------

def build_records(
    meta: dict, paragraphs: list[str]
) -> list[dict]:
    """
    Build paragraph records with unique IDs.
    ID format: {TICKER}_{YEAR}_{SEQ:04d}
    """
    records = []
    for i, text in enumerate(paragraphs):
        pid = f"{meta['ticker']}_{meta['year']}_{i:04d}"
        records.append(
            {
                "paragraph_id": pid,
                "ticker": meta["ticker"],
                "year": meta["year"],
                "text": text,
                "char_count": len(text),
            }
        )
    return records


# ---------------------------------------------------------------------------
# Step 6 & 7: Export + Statistics
# ---------------------------------------------------------------------------

def export_data(records: list[dict], filtered: list[dict], output_dir: str):
    """Export main table and filtered table to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Main paragraphs (above threshold) ---
    csv_path = os.path.join(output_dir, "paragraphs.csv")
    json_path = os.path.join(output_dir, "paragraphs.json")

    fieldnames = ["paragraph_id", "ticker", "year", "text", "char_count"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info("Exported %d paragraphs -> %s, %s", len(records), csv_path, json_path)

    # --- Filtered paragraphs (below threshold) ---
    if filtered:
        filtered_csv = os.path.join(output_dir, "filtered_paragraphs.csv")
        with open(filtered_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered)
        logger.info("Exported %d filtered (short) paragraphs -> %s", len(filtered), filtered_csv)


def export_short_sample(all_records: list[dict], output_dir: str, threshold: int = 100):
    """
    Export a sample of short paragraphs for observation (Two-pass step 1).
    """
    short = [r for r in all_records if r["char_count"] < threshold]
    sample_path = os.path.join(output_dir, "short_paragraphs_sample.csv")
    fieldnames = ["paragraph_id", "ticker", "year", "text", "char_count"]

    # Save all short paragraphs (or a sample if too many)
    with open(sample_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(short)

    # Length distribution report
    length_bins = Counter()
    for r in all_records:
        bucket = (r["char_count"] // 50) * 50  # 0-49, 50-99, 100-149, ...
        length_bins[bucket] += 1

    logger.info("=== Short paragraphs observation ===")
    logger.info("Total paragraphs (no filter): %d", len(all_records))
    logger.info("Paragraphs < %d chars: %d (%.1f%%)",
                threshold, len(short),
                100 * len(short) / len(all_records) if all_records else 0)
    logger.info("Length distribution (bucket -> count):")
    for bucket in sorted(length_bins.keys()):
        label = f"{bucket}-{bucket + 49}"
        logger.info("  %s chars: %d", label, length_bins[bucket])

    return short


def compute_stats(records: list[dict], skipped_files: int, error_files: int, output_dir: str):
    """Compute and export summary statistics."""
    stats: dict = {}
    stats["total_paragraphs"] = len(records)
    stats["skipped_files"] = skipped_files
    stats["error_files"] = error_files

    if records:
        lengths = [r["char_count"] for r in records]
        stats["char_count_mean"] = round(statistics.mean(lengths), 1)
        stats["char_count_median"] = round(statistics.median(lengths), 1)
        stats["char_count_min"] = min(lengths)
        stats["char_count_max"] = max(lengths)
        stats["char_count_stdev"] = round(statistics.stdev(lengths), 1) if len(lengths) > 1 else 0

    # Per-company counts
    company_counts: dict[str, int] = Counter()
    year_counts: dict[int, int] = Counter()
    company_year: dict[str, dict[int, int]] = defaultdict(Counter)
    for r in records:
        company_counts[r["ticker"]] += 1
        year_counts[r["year"]] += 1
        company_year[r["ticker"]][r["year"]] += 1

    stats["unique_tickers"] = len(company_counts)
    stats["year_distribution"] = dict(sorted(year_counts.items()))
    stats["top_10_companies"] = dict(company_counts.most_common(10))

    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("=== Preprocessing Statistics ===")
    logger.info("Total paragraphs: %d", stats["total_paragraphs"])
    logger.info("Unique tickers: %d", stats["unique_tickers"])
    logger.info("Skipped files (empty item_1a): %d", skipped_files)
    logger.info("Error files (parse failure): %d", error_files)
    if records:
        logger.info("Char count — mean: %.1f, median: %.1f, min: %d, max: %d",
                     stats["char_count_mean"], stats["char_count_median"],
                     stats["char_count_min"], stats["char_count_max"])
    logger.info("Year distribution: %s", stats["year_distribution"])
    logger.info("Stats saved to %s", stats_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_preprocess(data_dir: str = "data/10k_raw",
                   output_dir: str = "data/10k_1A",
                   raw_dir: str = "data/10k_raw",
                   min_chars: int = 100):
    """
    Run the full preprocessing pipeline (Steps 1-7).
    """
    logger.info("=" * 60)
    logger.info("Starting preprocessing: data_dir=%s, raw_dir=%s, min_chars=%d",
                data_dir, raw_dir or "(none)", min_chars)
    logger.info("=" * 60)

    if raw_dir and not HAS_BS4:
        logger.warning("beautifulsoup4 not installed — HTML title extraction disabled. "
                       "Install with: uv pip install beautifulsoup4")
        raw_dir = None

    # Step 1: Scan
    file_list = scan_files(data_dir)

    # Steps 2-4: Extract, clean, split
    all_records: list[dict] = []
    skipped = 0
    errors = 0
    title_split_count = 0
    fallback_count = 0

    for i, finfo in enumerate(file_list):
        meta, raw_text, html_titles = extract_item_1a(finfo, raw_dir=raw_dir)

        if not raw_text:
            skipped += 1
            logger.debug("Skipped (empty item_1a): %s", finfo["path"])
            continue

        cleaned = clean_text(raw_text)
        paragraphs = split_paragraphs(cleaned, titles=html_titles)
        records = build_records(meta, paragraphs)
        all_records.extend(records)

        if len(html_titles) >= 3:
            title_split_count += 1
        else:
            fallback_count += 1

        if (i + 1) % 500 == 0:
            logger.info("Processed %d / %d files (%d paragraphs so far)",
                        i + 1, len(file_list), len(all_records))

    logger.info("Extraction complete: %d paragraphs from %d files "
                "(%d title-split, %d fallback, %d skipped, %d errors)",
                len(all_records), len(file_list),
                title_split_count, fallback_count, skipped, errors)

    # Step 4 observation: short paragraphs
    os.makedirs(output_dir, exist_ok=True)
    export_short_sample(all_records, output_dir, threshold=min_chars)

    # Step 4 second pass: filter
    kept = [r for r in all_records if r["char_count"] >= min_chars]
    filtered = [r for r in all_records if r["char_count"] < min_chars]
    logger.info("After filtering (>= %d chars): %d kept, %d filtered out",
                min_chars, len(kept), len(filtered))

    # Step 6: Export
    export_data(kept, filtered, output_dir)

    # Step 7: Statistics
    compute_stats(kept, skipped, errors, output_dir)

    return kept


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Preprocess 10-K Item 1A")
    parser.add_argument("--data-dir", default="data/10k_raw",
                        help="Directory with raw HTML 10-K files (default: data/10k_raw)")
    parser.add_argument("--output-dir", default="data/10k_1A",
                        help="Output directory for processed data")
    parser.add_argument("--min-chars", type=int, default=100,
                        help="Minimum character count for paragraph filtering (default: 100)")
    args = parser.parse_args()
    run_preprocess(args.data_dir, args.output_dir, min_chars=args.min_chars)


if __name__ == "__main__":
    main()
