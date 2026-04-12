"""
SEC 10-K Filing Downloader (Step 0)

Downloads raw 10-K HTML filings from SEC EDGAR for S&P 500 companies.
Uses sec-downloader for metadata retrieval and HTML download.

Usage:
 uv run src/step0_download/sec-download.py       # Download all S&P 500 (2021-2025)
 uv run src/step0_download/sec-download.py --start-year 2023  # Only 2023-2025
 uv run src/step0_download/sec-download.py --tickers AAPL MSFT GOOG # Specific companies
 uv run src/step0_download/sec-download.py --dry-run    # Preview without downloading
 uv run src/step0_download/sec-download.py --resume     # Resume interrupted download
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from sec_downloader import Downloader
from sec_downloader.types import RequestedFilings

# SEC EDGAR fair access policy: max 10 req/sec, include contact info
COMPANY_NAME = "NPUST-ESG-Research"
CONTACT_EMAIL = "chiangkuante@gmail.com"
REQUEST_DELAY = 0.15 # seconds between requests (conservative)

# S&P 500 tickers (as of 2025/1/2) 503筆
SP500_TICKERS = [
 'A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APO', 'APTV', 'ARE', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBY', 'BDX', 'BEN', 'BF.B', 'BG', 'BIIB', 'BK', 'BKNG', 'BKR', 'BLDR', 'BLK', 'BMY', 'BR', 'BRK.B', 'BRO', 'BSX', 'BWA', 'BX', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COR', 'COST', 'CPAY', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DAY', 'DD', 'DE', 'DECK', 'DELL', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ERIE', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FITB', 'FMC', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GDDY', 'GE', 'GEHC', 'GEN', 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KKR', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LII', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PARA', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PLTR', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR', 'PYPL', 'QCOM', 'RCL', 'REG', 'REGN', 'RF', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SMCI', 'SNA', 'SNPS', 'SO', 'SOLV', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SW', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPL', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UBER', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VICI', 'VLO', 'VLTO', 'VMC', 'VRSK', 'VRSN', 'VRTX', 'VST', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 'WDAY', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZTS',
]

# Deduplicate (PEAK appears twice in the list above)
SP500_TICKERS = sorted(set(SP500_TICKERS))

DATA_DIR = Path("data/raw")
PROGRESS_FILE = DATA_DIR / "_download_progress.json"


def load_progress() -> dict:
 """Load download progress tracker."""
 if PROGRESS_FILE.exists():
  with open(PROGRESS_FILE) as f:
   return json.load(f)
 return {"completed": {}, "failed": {}, "metadata_errors": []}


def save_progress(progress: dict):
 """Save download progress tracker."""
 PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
 with open(PROGRESS_FILE, "w") as f:
  json.dump(progress, f, ensure_ascii=False, indent=2)


def get_filing_year(metadata) -> int:
 """Extract fiscal year from report_date or filing_date."""
 date_str = metadata.report_date or metadata.filing_date
 return int(date_str[:4])


def download_filings(
 tickers: list[str],
 start_year: int = 2021,
 end_year: int = 2025,
 dry_run: bool = False,
 resume: bool = False,
):
 """Download 10-K filings for given tickers and year range."""
 dl = Downloader(COMPANY_NAME, CONTACT_EMAIL)

 # Load or init progress
 progress = load_progress() if resume else {"completed": {}, "failed": {}, "metadata_errors": []}
 completed = progress["completed"] # {ticker: [accession_numbers]}
 failed = progress["failed"]  # {ticker: [error messages]}

 total_tickers = len(tickers)
 total_downloaded = sum(len(v) for v in completed.values())
 total_failed = sum(len(v) for v in failed.values())

 print(f"{'='*60}")
 print(f" SEC 10-K Filing Downloader")
 print(f"{'='*60}")
 print(f" Tickers: {total_tickers}")
 print(f" Years:  {start_year}-{end_year}")
 print(f" Output:  {DATA_DIR}")
 if resume:
  print(f" Resumed: {total_downloaded} completed, {total_failed} failed")
 if dry_run:
  print(f" Mode:  DRY RUN (no downloads)")
 print(f"{'='*60}\n")

 # Limit: request enough filings to cover the year range
 max_filings = end_year - start_year + 2 # +2 for safety margin

 for i, ticker in enumerate(tickers):
  print(f"[{i+1}/{total_tickers}] {ticker}...", end=" ", flush=True)

  # Skip if already completed for all years
  if resume and ticker in completed and len(completed[ticker]) >= (end_year - start_year + 1):
   print(f"SKIP (already {len(completed[ticker])} filings)")
   continue

  # Step 1: Get metadata
  try:
   metadatas = dl.get_filing_metadatas(
    RequestedFilings(
     ticker_or_cik=ticker,
     form_type="10-K",
     limit=max_filings,
    )
   )
   time.sleep(REQUEST_DELAY)
  except Exception as e:
   error_msg = f"{ticker}: metadata error - {e}"
   print(f"ERROR (metadata: {e})")
   progress["metadata_errors"].append(error_msg)
   save_progress(progress)
   continue

  # Filter by year range
  filtered = []
  for m in metadatas:
   try:
    year = get_filing_year(m)
    if start_year <= year <= end_year:
     filtered.append((year, m))
   except (ValueError, TypeError):
    continue

  if not filtered:
   print(f"no filings in {start_year}-{end_year}")
   continue

  print(f"found {len(filtered)} filings", end="")

  if dry_run:
   years = sorted(set(y for y, _ in filtered))
   print(f" (years: {years})")
   continue

  # Step 2: Download HTML for each filing
  downloaded_count = 0
  for year, metadata in filtered:
   accession = metadata.accession_number

   # Skip if already downloaded in a previous run
   done_accessions = completed.get(ticker, [])
   if accession in done_accessions:
    continue

   # Create directory structure: data/raw/{TICKER}/{YEAR}/{ACCESSION}/
   filing_dir = DATA_DIR / ticker / str(year) / accession
   filing_dir.mkdir(parents=True, exist_ok=True)

   html_path = filing_dir / "filing.html"
   meta_path = filing_dir / "metadata.json"

   # Skip if files already exist on disk
   if html_path.exists() and meta_path.exists():
    completed.setdefault(ticker, []).append(accession)
    continue

   try:
    html_bytes = dl.download_filing(url=metadata.primary_doc_url)
    time.sleep(REQUEST_DELAY)

    # Save HTML
    with open(html_path, "wb") as f:
     f.write(html_bytes)

    # Save metadata
    meta_dict = {
     "ticker": ticker,
     "cik": metadata.cik,
     "company_name": metadata.company_name,
     "accession_number": accession,
     "form_type": metadata.form_type,
     "filing_date": metadata.filing_date,
     "report_date": metadata.report_date,
     "primary_doc_url": metadata.primary_doc_url,
     "tickers": [
      {"symbol": t.symbol, "exchange": t.exchange}
      for t in (metadata.tickers or [])
     ],
    }
    with open(meta_path, "w") as f:
     json.dump(meta_dict, f, ensure_ascii=False, indent=2)

    completed.setdefault(ticker, []).append(accession)
    downloaded_count += 1

   except Exception as e:
    error_msg = f"{ticker}/{year}/{accession}: {e}"
    failed.setdefault(ticker, []).append(error_msg)

  print(f" -> downloaded {downloaded_count} new")

  # Save progress after each ticker
  save_progress(progress)

 # Final summary
 total_downloaded = sum(len(v) for v in completed.values())
 total_failed = sum(len(v) for v in failed.values())

 print(f"\n{'='*60}")
 print(f" Download Complete")
 print(f"{'='*60}")
 print(f" Total downloaded: {total_downloaded}")
 print(f" Total failed:  {total_failed}")
 print(f" Metadata errors:  {len(progress['metadata_errors'])}")
 print(f" Progress saved to: {PROGRESS_FILE}")

 if failed:
  print(f"\n Failed tickers:")
  for ticker, errors in failed.items():
   for err in errors:
    print(f" - {err}")


def main():
 parser = argparse.ArgumentParser(description="SEC 10-K Filing Downloader")
 parser.add_argument(
  "--tickers", nargs="+",
  help="Specific tickers to download (default: all S&P 500)",
 )
 parser.add_argument(
  "--start-year", type=int, default=2021,
  help="Start year (default: 2021)",
 )
 parser.add_argument(
  "--end-year", type=int, default=2025,
  help="End year (default: 2025)",
 )
 parser.add_argument(
  "--dry-run", action="store_true",
  help="Preview metadata without downloading HTML",
 )
 parser.add_argument(
  "--resume", action="store_true",
  help="Resume from previous progress",
 )
 args = parser.parse_args()

 tickers = args.tickers or SP500_TICKERS

 download_filings(
  tickers=tickers,
  start_year=args.start_year,
  end_year=args.end_year,
  dry_run=args.dry_run,
  resume=args.resume,
 )


if __name__ == "__main__":
 main()
