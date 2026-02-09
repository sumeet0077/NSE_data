
import os
import json
import logging
import datetime as dt
import pandas as pd
from pathlib import Path
from nse_bhav_pipeline import (
    aggregate_year, 
    START_DATE, 
    END_DATE, 
    daterange, 
    setup_logging, 
    METADATA_PATH, 
    PROCESSED_DAILY_ROOT, 
    FINAL_DATASET_DIR,
    LOG_PATH,
    RAW_ROOT
)

def finalize():
    setup_logging()
    logging.info("Starting aggregation only...")
    
    # Re-create final dir if needed (but handle with care if partial)
    if FINAL_DATASET_DIR.exists():
        for p in sorted(FINAL_DATASET_DIR.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        FINAL_DATASET_DIR.rmdir()
    FINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    all_years = sorted({d.year for d in daterange(START_DATE, END_DATE)})
    total_rows = 0
    all_symbols: set[str] = set()
    rows_2015_plus = 0
    deliv_2015_plus = 0
    min_trade_date: dt.date | None = None
    max_trade_date: dt.date | None = None
    feb1_2026_exists = False

    for year in all_years:
        (
            y_rows,
            y_unique_symbols,
            y_rows_2015_plus,
            y_deliv_2015_plus,
            y_min,
            y_max,
            y_feb1,
        ) = aggregate_year(year)
        
        if y_rows == 0:
            logging.info(f"No rows for year {year}")
            continue

        sample_df = pd.read_parquet(FINAL_DATASET_DIR / f"year={year}" / f"part-{year}.parquet", columns=["symbol"])
        all_symbols.update(sample_df["symbol"].astype(str).unique().tolist())

        total_rows += y_rows
        rows_2015_plus += y_rows_2015_plus
        deliv_2015_plus += y_deliv_2015_plus
        feb1_2026_exists = feb1_2026_exists or y_feb1

        if y_min and (min_trade_date is None or y_min < min_trade_date):
            min_trade_date = y_min
        if y_max and (max_trade_date is None or y_max > max_trade_date):
            max_trade_date = y_max

        logging.info("Year %d aggregated rows=%d unique_symbols=%d", year, y_rows, y_unique_symbols)

    deliv_pct_coverage_after_2015 = (deliv_2015_plus / rows_2015_plus * 100.0) if rows_2015_plus else 0.0

    metadata = {
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "min_trade_date": min_trade_date.isoformat() if min_trade_date else None,
        "max_trade_date": max_trade_date.isoformat() if max_trade_date else None,
        "row_count": total_rows,
        "unique_symbols": len(all_symbols),
        "rows_2015_onwards": rows_2015_plus,
        "rows_with_deliv_pct_2015_onwards": deliv_2015_plus,
        "deliv_pct_coverage_2015_onwards_percent": round(deliv_pct_coverage_after_2015, 4),
        "feb_1_2026_exists": feb1_2026_exists,
        "generated_at": dt.datetime.now().isoformat(),
        "output_dataset_dir": str(FINAL_DATASET_DIR.resolve()),
        "processed_daily_dir": str(PROCESSED_DAILY_ROOT.resolve()),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logging.info("Aggregation DONE.")

if __name__ == "__main__":
    finalize()
