
import pandas as pd
import pyarrow.dataset as ds
from pathlib import Path

def check_reliance():
    dataset_path = Path("nse_master_adjusted_2014_onwards.parquet")
    if not dataset_path.exists():
        print("Dataset not found!")
        return

    print(f"Reading dataset from {dataset_path}...")
    dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")
    
    table = dataset.to_table(filter=
        (ds.field("symbol") == "RELIANCE") & 
        (ds.field("year") == 2023)
    )
    df = table.to_pandas()
    
    # Filter dates
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    mask = (df["trade_date"] >= "2023-07-15") & (df["trade_date"] <= "2023-07-28")
    df = df[mask].sort_values("trade_date")
    
    # Calculate effective factor: adjusted_close / close
    df["eff_factor"] = df["adjusted_close"] / df["close"]
    
    print("\nReliance Data (July 2023):")
    cols = ["symbol", "series", "trade_date", "open", "close", "adjusted_close", "eff_factor"]
    cols = [c for c in cols if c in df.columns]
    
    print(df[cols].to_string(index=False))
    
    # Check specific adjustment
    # Ex-Date 2023-07-20 (Jio Financial demerger)
    # On 20th onwards, factor should be ~1.0 (no backward adjustment needed)
    # On 19th and before, factor should be < 1.0 (approx 0.9) because we backward-adjust
    try:
        row_19 = df[df["trade_date"] == "2023-07-19"].iloc[0]
        row_20 = df[df["trade_date"] == "2023-07-20"].iloc[0]
        
        f19 = row_19["eff_factor"]
        f20 = row_20["eff_factor"]
        
        print(f"\nEffective factor on 19th: {f19:.4f}")
        print(f"Effective factor on 20th: {f20:.4f}")
        
        # Backward adjustment means pre-demerger prices are multiplied by a factor < 1
        # So adjusted_close on 19th should be less than close on 19th
        if f19 < 0.95:
            print("\nSUCCESS: Demerger adjustment detected correctly!")
            print(f"  Pre-demerger prices are backward-adjusted by factor ~{f19:.4f}")
        elif f20 > 1.02:
            print("\nALTERNATIVE: Forward adjustment detected (post-demerger prices scaled up)")
        else:
            print("\nFAILURE: No significant adjustment pattern detected.")
            print("  This may indicate the demerger was not detected or data is incomplete.")
            
    except IndexError:
        print("\nData missing for specific dates.")

if __name__ == "__main__":
    check_reliance()
