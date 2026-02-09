#!/usr/bin/env python3
"""
Validation script to compare our adjusted_close with Yahoo Finance's Adj Close.
Samples random stocks from each year and reports discrepancies.
"""

import random
import pandas as pd
import pyarrow.dataset as ds
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import time
import json

# Configuration
SAMPLES_PER_YEAR = 20
TOLERANCE_PCT = 5.0  # Allow 5% tolerance for rounding/timing differences
OUTPUT_FILE = Path("validation_report.json")

def load_our_data():
    """Load our adjusted dataset."""
    dataset_path = Path("nse_master_adjusted_2014_onwards.parquet")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")
    # Only load columns needed for validation to avoid potential schema overflows in unused columns (like volume)
    columns = ["symbol", "series", "trade_date", "close", "adjusted_close"]
    df = dataset.to_table(columns=columns).to_pandas()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df

def get_unique_symbols_by_year(df):
    """Get unique symbols for each year."""
    df["year"] = df["trade_date"].dt.year
    symbols_by_year = {}
    for year in df["year"].unique():
        year_df = df[df["year"] == year]
        # Filter to EQ series only for cleaner comparison
        eq_df = year_df[year_df["series"] == "EQ"]
        symbols = eq_df["symbol"].unique().tolist()
        symbols_by_year[int(year)] = symbols
    return symbols_by_year

def sample_stocks(symbols_by_year, n=SAMPLES_PER_YEAR):
    """Sample n random stocks from each year."""
    sampled = {}
    for year, symbols in symbols_by_year.items():
        if len(symbols) >= n:
            sampled[year] = random.sample(symbols, n)
        else:
            sampled[year] = symbols
    return sampled

def fetch_yahoo_data(symbol, start_date, end_date):
    """Fetch historical data from Yahoo Finance."""
    ticker = f"{symbol}.NS"
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date, auto_adjust=False)
        if hist.empty:
            return None
        hist = hist.reset_index()
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
        return hist[["Date", "Close", "Adj Close"]]
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None

def compare_prices(our_df, symbol, year):
    """Compare our adjusted_close with Yahoo's Adj Close for a symbol in a year."""
    # Filter our data
    mask = (our_df["symbol"] == symbol) & (our_df["series"] == "EQ") & (our_df["trade_date"].dt.year == year)
    our_data = our_df[mask][["trade_date", "close", "adjusted_close"]].copy()
    our_data = our_data.sort_values("trade_date")
    
    if our_data.empty:
        return {"status": "no_data", "symbol": symbol, "year": year}
    
    # Get date range
    start_date = our_data["trade_date"].min() - timedelta(days=5)
    end_date = our_data["trade_date"].max() + timedelta(days=5)
    
    # Fetch Yahoo data
    yf_data = fetch_yahoo_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    if yf_data is None or yf_data.empty:
        return {"status": "yahoo_fetch_failed", "symbol": symbol, "year": year}
    
    # Merge on date
    our_data = our_data.rename(columns={"trade_date": "Date"})
    merged = our_data.merge(yf_data, on="Date", how="inner", suffixes=("_ours", "_yf"))
    
    if merged.empty:
        return {"status": "no_overlap", "symbol": symbol, "year": year}
    
    # Calculate percentage difference
    merged["pct_diff"] = abs(merged["adjusted_close"] - merged["Adj Close"]) / merged["Adj Close"] * 100
    
    # Find discrepancies
    discrepancies = merged[merged["pct_diff"] > TOLERANCE_PCT]
    
    result = {
        "status": "compared",
        "symbol": symbol,
        "year": year,
        "total_dates": len(merged),
        "discrepancy_count": len(discrepancies),
        "max_pct_diff": float(merged["pct_diff"].max()),
        "mean_pct_diff": float(merged["pct_diff"].mean()),
        "within_tolerance": len(discrepancies) == 0,
    }
    
    if len(discrepancies) > 0:
        # Sample some discrepancy examples
        sample_disc = discrepancies.head(3).to_dict(orient="records")
        result["sample_discrepancies"] = [
            {
                "date": str(r["Date"].date()),
                "our_adj_close": float(r["adjusted_close"]),
                "yf_adj_close": float(r["Adj Close"]),
                "pct_diff": float(r["pct_diff"])
            } for r in sample_disc
        ]
    
    return result

def run_validation():
    """Run the full validation."""
    print("Loading our dataset...")
    our_df = load_our_data()
    print(f"  Loaded {len(our_df)} rows")
    
    print("\nGetting unique symbols by year...")
    symbols_by_year = get_unique_symbols_by_year(our_df)
    for year, symbols in symbols_by_year.items():
        print(f"  {year}: {len(symbols)} symbols")
    
    print(f"\nSampling {SAMPLES_PER_YEAR} stocks per year...")
    sampled = sample_stocks(symbols_by_year)
    
    results = {
        "validation_date": datetime.now().isoformat(),
        "tolerance_pct": TOLERANCE_PCT,
        "samples_per_year": SAMPLES_PER_YEAR,
        "results_by_year": {},
        "summary": {
            "total_compared": 0,
            "total_passed": 0,
            "total_failed": 0,
            "total_errors": 0,
        }
    }
    
    for year, symbols in sorted(sampled.items()):
        print(f"\n=== Year {year} ===")
        year_results = []
        
        for i, symbol in enumerate(symbols):
            print(f"  [{i+1}/{len(symbols)}] Checking {symbol}...", end=" ")
            result = compare_prices(our_df, symbol, year)
            year_results.append(result)
            
            if result["status"] == "compared":
                if result["within_tolerance"]:
                    print(f"PASS (mean diff: {result['mean_pct_diff']:.2f}%)")
                    results["summary"]["total_passed"] += 1
                else:
                    print(f"FAIL (max diff: {result['max_pct_diff']:.2f}%)")
                    results["summary"]["total_failed"] += 1
                results["summary"]["total_compared"] += 1
            else:
                print(f"ERROR: {result['status']}")
                results["summary"]["total_errors"] += 1
            
            # Rate limiting for Yahoo Finance
            time.sleep(0.5)
        
        results["results_by_year"][year] = year_results
    
    # Calculate pass rate
    compared = results["summary"]["total_compared"]
    passed = results["summary"]["total_passed"]
    if compared > 0:
        results["summary"]["pass_rate_pct"] = round(passed / compared * 100, 2)
    else:
        results["summary"]["pass_rate_pct"] = 0
    
    # Save report
    OUTPUT_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n\nValidation complete. Report saved to {OUTPUT_FILE}")
    print(f"\nSummary:")
    print(f"  Total Compared: {compared}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {results['summary']['total_failed']}")
    print(f"  Errors: {results['summary']['total_errors']}")
    print(f"  Pass Rate: {results['summary']['pass_rate_pct']}%")
    
    return results

if __name__ == "__main__":
    run_validation()
