
import pandas as pd
import pyarrow.dataset as ds
from pathlib import Path

ADJUSTED_PATH = Path("nse_master_adjusted_2014_onwards.parquet")

def get_recent_prices():
    print("Fetching latest 30 days for MAHESHWARI...")
    
    dataset = ds.dataset(ADJUSTED_PATH, format="parquet", partitioning="hive")
    
    # Filter for symbol and recent years (2024-2026) to speed up search
    filter_expr = (ds.field("symbol") == "MAHESHWARI") & (ds.field("year") >= 2025)
    
    try:
        table = dataset.to_table(filter=filter_expr, columns=["trade_date", "series", "close", "adjusted_close"])
        df = table.to_pandas()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if df.empty:
        # Fallback to 2024 if 2025/26 is empty (unlikely but safe)
        print("No data in 2025+, checking 2024...")
        filter_expr = (ds.field("symbol") == "MAHESHWARI") & (ds.field("year") == 2024)
        table = dataset.to_table(filter=filter_expr, columns=["trade_date", "series", "close", "adjusted_close"])
        df = table.to_pandas()

    if df.empty:
        print("No recent data found for MAHESHWARI.")
        return

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date", ascending=False).head(30)
    
    print("\nMAHESHWARI - Latest 30 Trading Days:")
    print(f"{'Date':<12} | {'Series':<6} | {'Close':<8} | {'Adj Close':<10}")
    print("-" * 45)
    
    for _, row in df.iterrows():
        print(f"{row['trade_date'].date()} | {row['series']:<6} | {row['close']:<8.2f} | {row['adjusted_close']:<10.2f}")

if __name__ == "__main__":
    get_recent_prices()
