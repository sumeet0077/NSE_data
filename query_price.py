import pandas as pd
import sys

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

try:
    print("Reading parquet...")
    df = pd.read_parquet('/Users/sumeetdas/Antigravity_NSE_Data/nse_master_adjusted_2014_onwards.parquet', filters=[('year', '=', 2026)])
    
    target_date = '2026-02-10'
    print(f"Filtering for date: {target_date}")
    
    # Ensure trade_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
        df['trade_date'] = pd.to_datetime(df['trade_date'])

    subset = df[df['trade_date'].dt.strftime('%Y-%m-%d') == target_date]
    
    targets = ['RELIANCE', 'STALLION', 'STEL']
    print(f"Filtering for symbols: {targets}")
    subset = subset[subset['symbol'].isin(targets)]
    
    if subset.empty:
        print("No data found for the specified criteria.")
    else:
        print(subset[['symbol', 'trade_date', 'close', 'adj_close']].to_string())

except Exception as e:
    print(f"Error: {e}")
