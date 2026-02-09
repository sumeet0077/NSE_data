
import pandas as pd
import pyarrow.dataset as ds
from pathlib import Path

ADJUSTED_PATH = Path("nse_master_adjusted_2014_onwards.parquet")

def check_maheshwari():
    print("Loading data for MAHESHWARI in 2020...")
    
    dataset = ds.dataset(ADJUSTED_PATH, format="parquet", partitioning="hive")
    
    # Filter for symbol and year 2020
    # Note: 'year' is a partition key, so this is efficient
    filter_expr = (ds.field("symbol") == "MAHESHWARI") & (ds.field("year") == 2020)
    
    try:
        # Load columns needed
        table = dataset.to_table(filter=filter_expr, columns=["symbol", "series", "trade_date", "close", "adjusted_close"])
        df = table.to_pandas()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if df.empty:
        print("No data found for MAHESHWARI in 2020.")
        return

    # Remove global EQ filter to avoid losing months where only BE exists
    # if "EQ" in df["series"].unique():
    #     df = df[df["series"] == "EQ"]
        
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date")
    
    print("\nMAHESHWARI Adjusted Prices (1st Available Trading Day of Each Month in 2020):")
    print(f"{'Date':<12} | {'Series':<6} | {'Close':<8} | {'Adj Close':<10} | {'Factor':<6}")
    print("-" * 55)
    
    for month in range(1, 13):
        month_data = df[df["trade_date"].dt.month == month]
        if month_data.empty:
            print(f"Month {month}: No Data")
            continue
            
        # Per-month series selection: Prefer EQ, else take first available
        if "EQ" in month_data["series"].values:
            row = month_data[month_data["series"] == "EQ"].iloc[0]
        else:
            row = month_data.iloc[0]
            
        factor = row["adjusted_close"] / row["close"]
        print(f"{row['trade_date'].date()} | {row['series']:<6} | {row['close']:<8.2f} | {row['adjusted_close']:<10.2f} | {factor:.4f}")

if __name__ == "__main__":
    check_maheshwari()
