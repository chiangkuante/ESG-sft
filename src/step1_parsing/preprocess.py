"""
Step 1: Heading-Aware Semantic Parsing & Paragraph Segmentation

Parses 10-K HTML filings using sec-parser, extracts Item 1A (Risk Factors),
and produces structured paragraphs with heading context.

Usage:
  uv run src/step1_parsing/preprocess.py             # Process all filings in data/raw/
  uv run src/step1_parsing/preprocess.py --tickers AAPL MSFT   # Specific companies
  uv run src/step1_parsing/preprocess.py --dry-run        # Preview without writing output
  uv run src/step1_parsing/preprocess.py --output data/processed/step1_parsing/paragraphs.json
"""

import argparse
import json
import os
import re
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from collections import Counter

import sec_parser as sp

# Suppress sec-parser warnings about 10-K sections in 10-Q parser
warnings.filterwarnings("ignore", message="Invalid section type")

# ============================================================
# CONSTANTS
# ============================================================

RAW_DIR = Path("data/raw")
DEFAULT_OUTPUT = Path("data/processed/step1_parsing/paragraphs.json")

# Paragraph length thresholds
MIN_CHAR_LENGTH = 80    # Minimum chars for a valid paragraph
MAX_CHAR_LENGTH = 2000   # Maximum chars before splitting (≈ 500 tokens)
MERGE_THRESHOLD = 100   # Merge if shorter than this

# Item 1A boundary patterns
ITEM_1A_PATTERN = re.compile(
  r"^item\s*1a\.?\s*(risk\s*factors)?",
  re.IGNORECASE,
)
ITEM_END_PATTERNS = [
  re.compile(r"^item\s*1b\.?\s", re.IGNORECASE),
  re.compile(r"^item\s*2\.?\s", re.IGNORECASE),
  re.compile(r"^part\s*(ii|2)\b", re.IGNORECASE),
]

# Noise patterns for filtering
NOISE_PATTERNS = [
  re.compile(r"^(table\s+of\s+contents|index)$", re.IGNORECASE),
  re.compile(r"^page\s*\d+$", re.IGNORECASE),
  re.compile(r"^\d+$"), # just a number (page number)
  re.compile(r"^(continued|cont['']?d)\.?$", re.IGNORECASE),
  re.compile(r"^see\s+(item|note|part)\s", re.IGNORECASE),
  re.compile(r"^refer\s+to\s+", re.IGNORECASE),
  re.compile(r"^\.{3,}$"), # dots (TOC leader)
  re.compile(r"^item\s+\d", re.IGNORECASE), # cross-references like "Item 7"
]

# Section intro / safe harbor patterns
SAFE_HARBOR_PATTERNS = [
  re.compile(r"forward[\-\s]looking\s+statement", re.IGNORECASE),
  re.compile(r"safe\s+harbor", re.IGNORECASE),
  re.compile(r"cautionary\s+(note|statement)", re.IGNORECASE),
]

# Generic section-level heading keywords
# These are top-level category headings, NOT individual risk factor titles
SECTION_LEVEL_KEYWORDS = [
  re.compile(r"^(business|operational|financial|strategic|legal|regulatory|general)\s+risks?$", re.IGNORECASE),
  re.compile(r"^risks?\s+(related|relating)\s+to\s+(our\s+)?(business|operations|financ)", re.IGNORECASE),
  re.compile(r"^(macro[\-\s]?economic|industry|market|competition)\s+risks?$", re.IGNORECASE),
  re.compile(r"^risks?\s+(associated|related)\s+with", re.IGNORECASE),
  re.compile(r"^(technology|cyber|information)\s+risks?$", re.IGNORECASE),
  re.compile(r"^(human\s+capital|environmental|governance|compliance)\s+risks?$", re.IGNORECASE),
  re.compile(r"^(additional|other|certain)\s+risks?", re.IGNORECASE),
]


def _is_section_level_heading(text: str) -> bool:
  """Determine if a title is a generic section-level heading vs a specific risk factor title.

  Section-level headings are short, generic category labels like:
    "Business Risks", "OPERATIONAL RISKS", "Financial Risks"

  Risk-factor titles are longer, specific descriptions like:
    "Our business will suffer if we are not able to retain key personnel."
    "Cybersecurity incidents could disrupt our operations and expose us to liability."

  Rules:
  1. Match known section-level keyword patterns → section-level
  2. ALL CAPS and short (< 50 chars) → section-level
  3. Ends with period and > 40 chars → likely risk-specific
  4. Contains risk-related action phrases → likely risk-specific
  """
  stripped = text.strip()

  # Rule 1: Known section-level patterns
  for pattern in SECTION_LEVEL_KEYWORDS:
    if pattern.match(stripped):
      return True

  # Rule 2: ALL CAPS and short → section-level
  if stripped.isupper() and len(stripped) < 50:
    return True

  # Rule 3: Ends with period and reasonably long → risk-specific
  if stripped.endswith(".") and len(stripped) > 40:
    return False

  # Rule 4: Contains action phrases typical of risk factor titles → risk-specific
  risk_phrases = [
    "could adversely", "may adversely", "could harm", "may harm",
    "could materially", "may materially", "could negatively",
    "may negatively", "could impair", "could disrupt",
    "could result in", "may result in", "failure to",
    "inability to", "loss of", "decline in",
    "if we fail", "if we are unable", "we may not be able",
    "we may be unable", "we depend on", "we rely on",
  ]
  text_lower = stripped.lower()
  if any(phrase in text_lower for phrase in risk_phrases):
    return False

  # Rule 5: Very short (< 40 chars) and no period → likely section-level
  if len(stripped) < 40 and not stripped.endswith("."):
    return True

  # Default: treat as risk-specific (prefer keeping more info)
  return False


def fix_sentence_gluing(text: str) -> str:
  """Fix missing space between sentences where period is followed directly by uppercase.

  Example: "our business.Following the" → "our business. Following the"
  """
  # Fix period directly followed by uppercase letter (no space)
  text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', text)
  # Also fix after closing parenthesis or quote
  text = re.sub(r'([)\"\'])\.([A-Z])', r'\1. \2', text)
  # Fix after numbers (e.g., "2023.The")
  text = re.sub(r'(\d)\.([A-Z])', r'\1. \2', text)
  return text


# ============================================================
# CORE PARSING FUNCTIONS
# ============================================================

def parse_filing(html: str) -> list:
  """Parse HTML into sec-parser semantic elements."""
  return sp.Edgar10QParser().parse(html)


def find_item_1a_range(elements: list) -> tuple[int, int] | None:
  """Find the start and end index of Item 1A (Risk Factors) section.

  Returns (start_idx, end_idx) where start_idx is the element AFTER
  the Item 1A title, and end_idx is exclusive.
  """
  start_idx = None

  for i, elem in enumerate(elements):
    text = (elem.text or "").strip()
    elem_type = type(elem).__name__

    # Only look for Item 1A in title-like elements
    if elem_type in ("TopSectionTitle", "TitleElement"):
      if start_idx is None:
        if ITEM_1A_PATTERN.match(text):
          start_idx = i + 1 # skip the "Item 1A" title itself
      else:
        # Check if this is the end boundary
        for pattern in ITEM_END_PATTERNS:
          if pattern.match(text):
            return (start_idx, i)

  # If we found start but no explicit end, use end of doc
  if start_idx is not None:
    return (start_idx, len(elements))

  return None


def is_noise(text: str) -> bool:
  """Check if text matches common noise patterns."""
  stripped = text.strip()
  if not stripped:
    return True
  for pattern in NOISE_PATTERNS:
    if pattern.match(stripped):
      return True
  return False


def is_number_dense(text: str) -> bool:
  """Check if text is number/symbol dense (table residual)."""
  if len(text) < 30:
    return False
  # Count digits, currency symbols, percent signs
  num_chars = sum(1 for c in text if c.isdigit() or c in "$%.,()–-")
  ratio = num_chars / len(text)
  return ratio > 0.5


def is_safe_harbor(text: str) -> bool:
  """Check if text is a safe harbor/forward-looking statement."""
  for pattern in SAFE_HARBOR_PATTERNS:
    if pattern.search(text):
      return True
  return False


def split_at_sentence_boundary(text: str, max_len: int = MAX_CHAR_LENGTH) -> list[str]:
  """Split long text at sentence boundaries."""
  if len(text) <= max_len:
    return [text]

  sentences = re.split(r"(?<=[.!?])\s+", text)
  chunks = []
  current = ""

  for sent in sentences:
    if current and len(current) + len(sent) + 1 > max_len:
      chunks.append(current.strip())
      current = sent
    else:
      current = (current + " " + sent).strip() if current else sent

  if current.strip():
    chunks.append(current.strip())

  return chunks


def extract_paragraphs(
  elements: list,
  start_idx: int,
  end_idx: int,
  ticker: str,
  metadata: dict,
) -> list[dict]:
  """Extract structured paragraphs from Item 1A elements.

  Implements:
  - Heading tracking: TitleElement updates current_heading
  - Text collection: TextElement and SupplementaryText form paragraphs
  - Table removal: TableElement skipped
  - Noise filtering: short/number-dense/cross-reference text removed
  - Safe harbor separation: forward-looking statements flagged
  - Short fragment merging
  - Long text splitting at sentence boundaries
  """
  paragraphs = []
  current_heading = ""
  current_section = "" # top-level section (e.g., "Business Risks")
  first_heading_seen = False
  section_intros = []

  # Phase 1: Collect raw paragraphs
  raw_paragraphs = []

  for i in range(start_idx, end_idx):
    elem = elements[i]
    elem_type = type(elem).__name__
    text = (elem.text or "").strip()

    if not text:
      continue

    # Skip tables entirely
    if elem_type == "TableElement":
      continue

    # Skip images
    if elem_type == "ImageElement":
      continue

    # Skip noise
    if is_noise(text):
      continue

    # Handle titles - distinguish section-level vs risk-level headings
    if elem_type == "TitleElement":
      if not text:
        continue

      # Determine if this is a section-level heading (generic) or
      # a risk-specific sub-heading (what we actually want)
      is_section_level = _is_section_level_heading(text)

      if is_section_level:
        # Top-level section heading: "Business Risks", "OPERATIONAL RISKS"
        current_section = text
        # Reset current_heading to section level
        # (will be overridden if a specific sub-heading follows)
        current_heading = text
        first_heading_seen = True
      else:
        # Risk-specific sub-heading (the kind we want)
        current_heading = text
        first_heading_seen = True

      continue

    # Text content (TextElement or SupplementaryText)
    if elem_type in ("TextElement", "SupplementaryText"):
      # Fix sentence gluing (missing space after period)
      text = fix_sentence_gluing(text)

      # Skip number-dense table residuals
      if is_number_dense(text):
        continue

      # Before first heading: section intro
      if not first_heading_seen:
        if is_safe_harbor(text):
          section_intros.append({
            "type": "safe_harbor",
            "text": text,
          })
        else:
          section_intros.append({
            "type": "section_intro",
            "text": text,
          })
        continue

      raw_paragraphs.append({
        "heading": current_heading,
        "section": current_section,
        "text": text,
        "char_count": len(text),
        "elem_type": elem_type,
      })

  # Phase 2: Merge short fragments under the same heading
  merged = []
  for para in raw_paragraphs:
    if (
      merged
      and merged[-1]["heading"] == para["heading"]
      and merged[-1]["char_count"] < MERGE_THRESHOLD
    ):
      # Merge with previous
      merged[-1]["text"] += " " + para["text"]
      merged[-1]["char_count"] = len(merged[-1]["text"])
      merged[-1]["cleaning_flags"].append("merged_short_fragment")
    else:
      para["cleaning_flags"] = []
      merged.append(para)

  # Phase 3: Split long texts and build final output
  counter = 0
  filing_date = metadata.get("filing_date", "")
  accession = metadata.get("accession_number", "")
  cik = metadata.get("cik", "")
  report_year = filing_date[:4] if filing_date else ""

  for para in merged:
    text = para["text"]
    heading = para["heading"]

    # Skip if too short even after merging
    if len(text) < MIN_CHAR_LENGTH:
      continue

    # Build combined_text: heading + newline + text
    combined = f"{heading}\n{text}" if heading and heading != text else text

    # Split if too long
    text_chunks = split_at_sentence_boundary(text, MAX_CHAR_LENGTH)

    for chunk in text_chunks:
      counter += 1
      chunk_combined = f"{heading}\n{chunk}" if heading and heading != chunk else chunk

      flags = list(para["cleaning_flags"])
      if len(text_chunks) > 1:
        flags.append("split_long_paragraph")

      paragraphs.append({
        "paragraph_id": f"{ticker}_{report_year}_RISK_{counter:03d}",
        "ticker": ticker,
        "cik": cik,
        "filing_date": filing_date,
        "accession_number": accession,
        "risk_section": para.get("section", ""),
        "risk_heading": heading,
        "paragraph_text": chunk,
        "combined_text": chunk_combined,
        "char_count": len(chunk_combined),
        "cleaning_flags": flags,
      })

  return paragraphs


# ============================================================
# FILE PROCESSING
# ============================================================

def process_filing(filing_dir: Path) -> tuple[list[dict], str] | tuple[None, str]:
  """Process a single filing directory.

  Returns (paragraphs, status) where status is 'ok', 'not_found', or 'error'.
  """
  html_path = filing_dir / "filing.html"
  meta_path = filing_dir / "metadata.json"

  if not html_path.exists():
    return None, "error"

  # Load metadata
  metadata = {}
  if meta_path.exists():
    with open(meta_path) as f:
      metadata = json.load(f)

  ticker = metadata.get("ticker", filing_dir.parent.parent.name)

  # Parse HTML
  try:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    elements = parse_filing(html)
  except Exception as e:
    return None, "error"

  # Find Item 1A
  item_range = find_item_1a_range(elements)
  if item_range is None:
    return None, "not_found"

  start_idx, end_idx = item_range

  # Extract paragraphs
  paragraphs = extract_paragraphs(elements, start_idx, end_idx, ticker, metadata)

  return paragraphs, "ok"


def _process_ticker(ticker_dir: Path) -> tuple[str, list[dict], dict]:
  """Process all filings for a single ticker. Used by multiprocessing."""
  ticker = ticker_dir.name
  ticker_paragraphs = []
  ticker_stats = {
    "total_filings": 0,
    "successful": 0,
    "item1a_not_found": 0,
    "parse_errors": 0,
    "total_paragraphs": 0,
  }

  for year_dir in sorted(ticker_dir.iterdir()):
    if not year_dir.is_dir():
      continue
    for filing_dir in sorted(year_dir.iterdir()):
      if not filing_dir.is_dir():
        continue

      ticker_stats["total_filings"] += 1
      result, status = process_filing(filing_dir)

      if result is None or len(result) == 0:
        if status == "not_found" or (result is not None and len(result) == 0):
          ticker_stats["item1a_not_found"] += 1
        else:
          ticker_stats["parse_errors"] += 1
      else:
        ticker_stats["successful"] += 1
        ticker_stats["total_paragraphs"] += len(result)
        ticker_paragraphs.extend(result)

  return ticker, ticker_paragraphs, ticker_stats


def process_all(
  raw_dir: Path,
  output_path: Path,
  tickers: list[str] | None = None,
  dry_run: bool = False,
  workers: int | None = None,
):
  """Process all filings and produce output using multiprocessing."""
  # Discover filing directories
  ticker_dirs = sorted(
    d for d in raw_dir.iterdir()
    if d.is_dir() and not d.name.startswith("_")
  ) if raw_dir.exists() else []
  if tickers:
    tickers_set = set(t.upper() for t in tickers)
    ticker_dirs = [d for d in ticker_dirs if d.name in tickers_set]

  total_tickers = len(ticker_dirs)
  n_workers = workers or min(cpu_count(), total_tickers or 1)

  print(f"{'='*60}")
  print(f" 10-K Item 1A Paragraph Extractor")
  print(f"{'='*60}")
  print(f" Input:  {raw_dir}")
  print(f" Output:  {output_path}")
  print(f" Tickers: {total_tickers}")
  print(f" Workers: {n_workers}")
  if dry_run:
    print(f" Mode:   DRY RUN")
  print(f"{'='*60}\n")

  # Process tickers in parallel
  all_paragraphs = []
  stats = {
    "total_filings": 0,
    "successful": 0,
    "item1a_not_found": 0,
    "parse_errors": 0,
    "total_paragraphs": 0,
  }
  per_ticker_stats = {}

  with Pool(processes=n_workers) as pool:
    results = pool.imap_unordered(_process_ticker, ticker_dirs)
    completed = 0
    for ticker, paras, t_stats in results:
      completed += 1
      per_ticker_stats[ticker] = t_stats["total_paragraphs"]
      all_paragraphs.extend(paras)
      for key in ("total_filings", "successful", "item1a_not_found", "parse_errors", "total_paragraphs"):
        stats[key] += t_stats[key]
      print(f" [{completed}/{total_tickers}] {ticker}: {t_stats['total_paragraphs']} paragraphs")

  # Sort by paragraph_id for deterministic output
  all_paragraphs.sort(key=lambda p: p["paragraph_id"])

  # Save output
  if not dry_run and all_paragraphs:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
      json.dump(all_paragraphs, f, ensure_ascii=False, indent=2)

  # Print summary
  print(f"\n{'='*60}")
  print(f" Processing Complete")
  print(f"{'='*60}")
  print(f" Total filings processed:  {stats['total_filings']}")
  print(f" Successful:        {stats['successful']}")
  print(f" Item 1A not found:     {stats['item1a_not_found']}")
  print(f" Parse errors:       {stats['parse_errors']}")
  print(f" Total paragraphs:     {stats['total_paragraphs']}")
  if all_paragraphs:
    lengths = [p["char_count"] for p in all_paragraphs]
    print(f" Avg paragraph length:   {sum(lengths)/len(lengths):.0f} chars")
    print(f" Min/Max length:      {min(lengths)}/{max(lengths)} chars")

  if not dry_run and all_paragraphs:
    print(f"\n Output saved to: {output_path}")

    # Save stats
    stats_path = output_path.parent / "parsing_stats.json"
    stats["per_ticker"] = per_ticker_stats
    with open(stats_path, "w") as f:
      json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f" Stats saved to: {stats_path}")

  # Warn about anomalies
  anomalies = {t: c for t, c in per_ticker_stats.items() if 0 < c < 5}
  if anomalies:
    print(f"\n ⚠ Tickers with very few paragraphs:")
    for t, c in sorted(anomalies.items()):
      print(f"  {t}: {c}")

  return all_paragraphs, stats


# ============================================================
# MAIN
# ============================================================

def main():
  parser = argparse.ArgumentParser(
    description="Extract Item 1A paragraphs from 10-K filings"
  )
  parser.add_argument(
    "--raw-dir", type=str, default=str(RAW_DIR),
    help="Directory containing raw HTML filings (default: data/raw/)",
  )
  parser.add_argument(
    "--output", type=str, default=str(DEFAULT_OUTPUT),
    help="Output JSON file path (default: data/processed/step1_parsing/paragraphs.json)",
  )
  parser.add_argument(
    "--tickers", nargs="+",
    help="Specific tickers to process (default: all)",
  )
  parser.add_argument(
    "--dry-run", action="store_true",
    help="Preview without writing output",
  )
  parser.add_argument(
    "--workers", type=int, default=None,
    help="Number of parallel workers (default: CPU count)",
  )
  args = parser.parse_args()

  process_all(
    raw_dir=Path(args.raw_dir),
    output_path=Path(args.output),
    tickers=args.tickers,
    dry_run=args.dry_run,
    workers=args.workers,
  )


if __name__ == "__main__":
  main()