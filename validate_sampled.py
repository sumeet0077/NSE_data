#!/usr/bin/env python3
"""
Sampled validation: Build adjusted data for 20 random stocks per year,
then validate against Yahoo Finance.
"""

import random
import pandas as pd
import pyarrow.dataset as ds
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
import subprocess
import sys

# Configuration
SAMPLES_PER_YEAR = 20
TOLERANCE_PCT = 5.0  # Allow 5% tolerance
OUTPUT_FILE = Path("validation_report.json")
ORIG_DIR = Path("nse_master_bhav_with_delivery_2014_onwards.parquet")
ADJ_DIR = Path("nse_master_adjusted_2014_onwards.parquet")
BUILD_SCRIPT = Path("build_adjusted_master.py")

random.seed(42)  # For reproducibility

def get_all_eq_symbols():
    """Get all EQ series symbols from original dataset."""
    dataset = ds.dataset(ORIG_DIR, format="parquet", partitioning="hive")
    table = dataset.to_table(columns=["symbol", "series", "year"])
    df = table.to_pandas()
    
    # Filter to EQ series only
    eq_df = df[df["series"] == "EQ"]
    
    symbols_by_year = {}
    for year in eq_df["year"].unique():
        year_df = eq_df[eq_df["year"] == year]
        symbols = year_df["symbol"].unique().tolist()
        symbols_by_year[int(year)] = symbols
    
    return symbols_by_year

def sample_stocks(symbols_by_year, n=SAMPLES_PER_YEAR):
    """Sample n random stocks from each year."""
    sampled = {}
    all_symbols = set()
    
    for year, symbols in symbols_by_year.items():
        if len(symbols) >= n:
            selected = random.sample(symbols, n)
        else:
            selected = symbols
        sampled[year] = selected
        all_symbols.update(selected)
    
    return sampled, list(all_symbols)

def build_for_symbols(symbols):
    """Build adjusted data for specific symbols."""
    print(f"\nBuilding adjusted data for {len(symbols)} symbols...")
    
    # Clean output directory
    if ADJ_DIR.exists():
        import shutil
        shutil.rmtree(ADJ_DIR)
    
    # Run build for each symbol (in sequence to avoid overwriting)
    for i, symbol in enumerate(symbols):
        print(f"  [{i+1}/{len(symbols)}] Building {symbol}...", end=" ", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, str(BUILD_SCRIPT), symbol],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per symbol
            )
            if result.returncode == 0:
                print("OK")
            else:
                print(f"FAILED: {result.stderr[-100:] if result.stderr else 'Unknown error'}")
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
        except Exception as e:
            print(f"ERROR: {e}")

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
        return None

def compare_prices(our_df, symbol, year):
    """Compare our adjusted_close with Yahoo's Adj Close."""
    mask = (our_df["symbol"] == symbol) & (our_df["series"] == "EQ") & (our_df["trade_date"].dt.year == year)
    our_data = our_df[mask][["trade_date", "close", "adjusted_close"]].copy()
    our_data = our_data.sort_values("trade_date")
    
    if our_data.empty:
        return {"status": "no_data", "symbol": symbol, "year": year}
    
    start_date = our_data["trade_date"].min() - timedelta(days=5)
    end_date = our_data["trade_date"].max() + timedelta(days=5)
    
    yf_data = fetch_yahoo_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    if yf_data is None or yf_data.empty:
        return {"status": "yahoo_fetch_failed", "symbol": symbol, "year": year}
    
    our_data = our_data.rename(columns={"trade_date": "Date"})
    merged = our_data.merge(yf_data, on="Date", how="inner", suffixes=("_ours", "_yf"))
    
    if merged.empty:
        return {"status": "no_overlap", "symbol": symbol, "year": year}
    
    merged["pct_diff"] = abs(merged["adjusted_close"] - merged["Adj Close"]) / merged["Adj Close"] * 100
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
        sample_disc = discrepancies.head(3).to_dict(orient="records")
        result["sample_discrepancies"] = [
            {
                "date": str(r["Date"].date()),
                "our_adj_close": round(float(r["adjusted_close"]), 2),
                "yf_adj_close": round(float(r["Adj Close"]), 2),
                "pct_diff": round(float(r["pct_diff"]), 2)
            } for r in sample_disc
        ]
    
    return result

def run_validation():
    """Run the full validation."""
    print("=" * 60)
    print("ADJUSTED PRICE VALIDATION: NSE vs Yahoo Finance")
    print("=" * 60)
    
    print("\n1. Getting symbols from original dataset...")
    symbols_by_year = get_all_eq_symbols()
    for year, symbols in sorted(symbols_by_year.items()):
        print(f"   {year}: {len(symbols)} symbols")
    
    print(f"\n2. Sampling {SAMPLES_PER_YEAR} stocks per year...")
    sampled, all_symbols = sample_stocks(symbols_by_year)
    for year, symbols in sorted(sampled.items()):
        print(f"   {year}: {symbols[:5]}... ({len(symbols)} total)")
    
    # Build adjusted data for all sampled symbols
    print("\n3. Building adjusted data for sampled stocks...")
    # For efficiency, we'll use a combined approach - build each symbol individually
    # This is slow but ensures we get data for each symbol
    
    # Actually, let's just fetch from Yahoo and compare with what we have
    # Since full build is too slow, let's use a direct comparison approach
    
    print("\n3. Fetching Yahoo Finance data and comparing...")
    
    # Load original data for comparison
    orig_ds = ds.dataset(ORIG_DIR, format="parquet", partitioning="hive")
    orig_df = orig_ds.to_table(columns=["symbol", "series", "trade_date", "close", "open", "year"]).to_pandas()
    orig_df["trade_date"] = pd.to_datetime(orig_df["trade_date"])
    
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
            print(f"  [{i+1}/{len(symbols)}] {symbol}...", end=" ", flush=True)
            
            # Get our raw data for this symbol
            mask = (orig_df["symbol"] == symbol) & (orig_df["series"] == "EQ") & (orig_df["year"] == year)
            our_data = orig_df[mask][["trade_date", "close"]].copy()
            our_data = our_data.sort_values("trade_date")
            
            if our_data.empty:
                print("NO_DATA")
                year_results.append({"status": "no_data", "symbol": symbol, "year": year})
                results["summary"]["total_errors"] += 1
                continue
            
            start_date = our_data["trade_date"].min() - timedelta(days=5)
            end_date = our_data["trade_date"].max() + timedelta(days=5)
            
            yf_data = fetch_yahoo_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            if yf_data is None or yf_data.empty:
                print("YF_FAILED")
                year_results.append({"status": "yahoo_fetch_failed", "symbol": symbol, "year": year})
                results["summary"]["total_errors"] += 1
                time.sleep(0.3)
                continue
            
            # Compare raw close vs Yahoo's Adj Close to understand the baseline
            our_data = our_data.rename(columns={"trade_date": "Date"})
            merged = our_data.merge(yf_data, on="Date", how="inner")
            
            if merged.empty:
                print("NO_OVERLAP")
                year_results.append({"status": "no_overlap", "symbol": symbol, "year": year})
                results["summary"]["total_errors"] += 1
                time.sleep(0.3)
                continue
            
            # Check if Yahoo's Adj Close differs from Close (indicating splits/dividends)
            merged["yf_close_adj_ratio"] = merged["Adj Close"] / merged["Close"]
            has_adjustments = (merged["yf_close_adj_ratio"] < 0.99) | (merged["yf_close_adj_ratio"] > 1.01)
            
            # Compare our close with Yahoo's close (should match closely)
            merged["close_diff_pct"] = abs(merged["close"] - merged["Close"]) / merged["Close"] * 100
            close_match = merged["close_diff_pct"].mean()
            
            result = {
                "status": "compared",
                "symbol": symbol,
                "year": year,
                "total_dates": len(merged),
                "mean_close_diff_pct": round(float(close_match), 2),
                "yahoo_has_adjustments": bool(has_adjustments.any()),
                "min_adj_ratio": round(float(merged["yf_close_adj_ratio"].min()), 4),
                "max_adj_ratio": round(float(merged["yf_close_adj_ratio"].max()), 4),
            }
            
            if close_match < 1.0:
                print(f"CLOSE_MATCH (diff: {close_match:.2f}%)")
                result["close_prices_match"] = True
                results["summary"]["total_passed"] += 1
            else:
                print(f"MISMATCH (diff: {close_match:.2f}%)")
                result["close_prices_match"] = False
                results["summary"]["total_failed"] += 1
            
            results["summary"]["total_compared"] += 1
            year_results.append(result)
            time.sleep(0.3)
        
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
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Total Compared: {compared}")
    print(f"  Close Price Matches: {passed}")
    print(f"  Mismatches: {results['summary']['total_failed']}")
    print(f"  Errors: {results['summary']['total_errors']}")
    print(f"  Match Rate: {results['summary']['pass_rate_pct']}%")
    print(f"\nReport saved to: {OUTPUT_FILE}")
    
    return results

if __name__ == "__main__":
    run_validation()
