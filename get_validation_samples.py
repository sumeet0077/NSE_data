
import pandas as pd
import pyarrow.dataset as ds
from pathlib import Path
import random

ACTIONS_PATH = Path("nse_corporate_actions_equities_2014_onwards.parquet")
ADJUSTED_PATH = Path("nse_master_adjusted_2014_onwards.parquet")

def get_samples():
    # 1. Find candidates
    try:
        actions = pd.read_parquet(ACTIONS_PATH)
    except:
        print("Could not load actions.")
        return

    # Check which symbols are actually in our adjusted dataset
    dataset = ds.dataset(ADJUSTED_PATH, format="parquet", partitioning="hive")
    
    mask = actions["subject"].str.contains(r"split|bonus|demerger|spin", case=False, regex=True)
    candidates = actions.loc[mask, "symbol"].unique().tolist()
    random.shuffle(candidates)
    
    selected = []
    print(f"Searching for 2 valid candidates among {len(candidates)} potential symbols...")
    
    for cand in candidates:
        if len(selected) >= 2:
            break
            
        # Check if data exists - efficiently
        # Use a limit to avoid scanning everything if it exists
        try:
             # Just try to get 1 row
             table = dataset.head(1, filter=(ds.field("symbol") == cand))
             if table.num_rows > 0:
                 selected.append(cand)
                 print(f"  Found valid candidate: {cand}")
             else:
                 pass # print(f"  {cand}: No data")
        except Exception as e:
            # print(f"  Error checking {cand}: {e}")
            continue
            
    print(f"Selected Stocks: {', '.join(selected)}")

    # 2. Extract data
    dataset = ds.dataset(ADJUSTED_PATH, format="parquet", partitioning="hive")
    
    for sym in selected:
        print(f"\n=== {sym} ===")
        # Get Corporate Action details for context
        sym_actions = actions[(actions["symbol"] == sym) & mask][["exDate", "subject"]].sort_values("exDate")
        print("Key Events:")
        for _, row in sym_actions.iterrows():
            print(f"  {row['exDate']}: {row['subject']}")
            
        print("\nAnnual Adjusted Prices (Snapshot approx mid-year):")
        
        # Load all data for symbol (remove EQ filter at pyarrow level)
        # Load all data for symbol (remove EQ filter at pyarrow level)
        try:
            # table = dataset.to_table(filter=(ds.field("symbol") == sym) & (ds.field("series") == "EQ"))
            # Only read needed columns to avoid overflow in Volume
            table = dataset.to_table(filter=(ds.field("symbol") == sym), columns=["symbol", "series", "trade_date", "close", "adjusted_close"])
            df = table.to_pandas()
        except Exception as e:
            print(f"  Error loading data: {e}")
            continue
            
        if df.empty:
            print("  No data found (DataFrame empty).")
            continue
            
        # Filter for EQ primarily, fallback to others if needed
        if "EQ" in df["series"].unique():
             df = df[df["series"] == "EQ"]
        else:
             print(f"  Note: 'EQ' series not found. Available: {df['series'].unique()}")
             # Take the most common series
             top_series = df["series"].mode()[0]
             df = df[df["series"] == top_series]

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["year"] = df["trade_date"].dt.year
        df = df.sort_values("trade_date")
        
        # Pick one date per year (e.g., first trading day of July)
        years = sorted(df["year"].unique())
        for y in years:
            # Try to get July 1st, or first date after
            target = pd.Timestamp(f"{y}-07-01")
            year_data = df[df["year"] == y]
            
            # Find nearest date after July 1st
            after = year_data[year_data["trade_date"] >= target]
            if not after.empty:
                row = after.iloc[0]
            else:
                # If no data after July (e.g. current year incomplete, or delisted), take last available
                row = year_data.iloc[-1]
                
            print(f"  {y}: Date={row['trade_date'].date()} | Close={row['close']:.2f} | Adj Close={row['adjusted_close']:.2f}")

if __name__ == "__main__":
    get_samples()
