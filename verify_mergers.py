
import pandas as pd
import pyarrow.dataset as ds
from pathlib import Path
import datetime as dt

# Paths
ACTIONS_PATH = Path("nse_corporate_actions_equities_2014_onwards.parquet")
ADJUSTED_PATH = Path("nse_master_adjusted_2014_onwards.parquet")

def verify_mergers():
    # 1. Load Corporate Actions
    print("Loading corporate actions...")
    try:
        actions_df = pd.read_parquet(ACTIONS_PATH)
    except Exception as e:
        print(f"Error loading actions: {e}")
        return

    # 2. Filter for Merger/Demerger related keywords
    # Common keywords: "Demerger", "Amalgamation", "Arrangement", "Spin off"
    mask = actions_df["subject"].str.contains(r"demerger|merger|amalgamation|arrangement|spin|hive", case=False, regex=True)
    reorg_actions = actions_df[mask].copy()
    
    print(f"Found {len(reorg_actions)} potential merger/demerger events.")
    
    # 3. Select a few candidates (prefer recent ones or well-known ones if possible, but random is fine)
    # We want events that are likely to have caused a price drop.
    # We will pick 5 random ones and check if we have data for them.
    if reorg_actions.empty:
        print("No merger/demerger events found in cache.")
        return

    candidates = reorg_actions.sample(min(10, len(reorg_actions)))
    
    print(f"Checking {len(candidates)} candidates for data availability...")

    # Load Adjusted Data (lazy)
    dataset = ds.dataset(ADJUSTED_PATH, format="parquet", partitioning="hive")
    
    found_count = 0
    for idx, row in candidates.iterrows():
        if found_count >= 2:
            break
            
        symbol = row["symbol"]
        ex_date_str = row["exDate"]
        subject = row["subject"]
        
        try:
            ex_date = pd.to_datetime(ex_date_str).date()
        except:
            continue
            
        print(f"\nEvaluating: {symbol} (Ex-Date: {ex_date}) - {subject}")
        
        # Load data for this symbol around the ex-date
        # We need a window: ex_date - 10 days to ex_date + 10 days
        start_date = ex_date - dt.timedelta(days=15)
        end_date = ex_date + dt.timedelta(days=15)
        
        # Pyarrow filter (efficient)
        filter_expr = (ds.field("symbol") == symbol) & (ds.field("series") == "EQ") & (ds.field("trade_date") >= pd.Timestamp(start_date)) & (ds.field("trade_date") <= pd.Timestamp(end_date))
        
        try:
            table = dataset.to_table(filter=filter_expr)
            df = table.to_pandas()
        except Exception as e:
            print(f"  Error loading data: {e}")
            continue
            
        if df.empty:
            print("  No data found in window.")
            continue
            
        df = df.sort_values("trade_date")
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        
        # Check if we cover the ex-date boundary
        if df["trade_date"].min() >= ex_date or df["trade_date"].max() <= ex_date:
            print("  Insufficient data around ex-date.")
            continue
            
        # Find the drop
        # Locate the row strictly before ex_date and the row on/after ex_date
        pre_mask = df["trade_date"] < ex_date
        post_mask = df["trade_date"] >= ex_date
        
        if not pre_mask.any() or not post_mask.any():
            print("  Gap in data around ex-date.")
            continue
            
        pre_row = df[pre_mask].iloc[-1]
        post_row = df[post_mask].iloc[0]
        
        # Calculate drops
        raw_drop_pct = (post_row["close"] - pre_row["close"]) / pre_row["close"] * 100
        adj_drop_pct = (post_row["adjusted_close"] - pre_row["adjusted_close"]) / pre_row["adjusted_close"] * 100
        
        print(f"  Pre-Event ({pre_row['trade_date']}): Close={pre_row['close']:.2f}, AdjClose={pre_row['adjusted_close']:.2f}")
        print(f"  Post-Event ({post_row['trade_date']}): Close={post_row['close']:.2f}, AdjClose={post_row['adjusted_close']:.2f}")
        print(f"  Raw Change: {raw_drop_pct:.2f}%")
        print(f"  Adj Change: {adj_drop_pct:.2f}%")
        
        # If adjusted is smoother than raw (meaning raw dropped big, adj didn't), it worked.
        # OR if both dropped, maybe it wasn't adjusted (which is fine if it wasn't supposed to be).
        # But for demergers, typically Raw drops, Adj is "back-adjusted" so the drop disappears in the history? 
        # Wait, for Splits: Raw 1000->100. Adj History 100->100. So Pre-Event Adj is lower.
        # For Demerger: Raw 1000->800. Adj History 800->800. So Pre-Event Adj is lower.
        
        # Let's see if there's a difference between Close and AdjClose in the PRE-event period.
        if abs(pre_row["close"] - pre_row["adjusted_close"]) > 0.01:
             print("  => Adjustment DETECTED (Pre-event Close != Pre-event AdjClose)")
             found_count += 1
        else:
             print("  => No significant adjustment detected.")
             found_count += 1 # Still count it as an example

if __name__ == "__main__":
    verify_mergers()
