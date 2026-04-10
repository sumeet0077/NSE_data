import pandas as pd
import numpy as np
import duckdb
import os

def test_aegis_mapping():
    parquet_path = 'nse_master_adjusted_2014_onwards.parquet'
    con = duckdb.connect(':memory:')
    
    # 1. Pull data for both symbols
    query = f"""
        SELECT symbol, trade_date, adj_close, volume
        FROM read_parquet('{parquet_path}', union_by_name=True)
        WHERE symbol IN ('AEGISLOG', 'AEGISCHEM')
        ORDER BY trade_date
    """
    df = con.query(query).df()
    print(f"Total rows pulled: {len(df)}")
    print(f"Symbols present: {df['symbol'].unique()}")
    
    # 2. Simulate the mapping
    df['symbol'] = df['symbol'].replace('AEGISCHEM', 'AEGISLOG')
    print(f"After mapping, symbols: {df['symbol'].unique()}")
    
    # 3. Simulate Active Filter
    # Need to know the global latest date
    latest_date = con.query(f"SELECT MAX(trade_date) FROM read_parquet('{parquet_path}', union_by_name=True)").fetchone()[0]
    print(f"Global latest date: {latest_date}")
    
    active_symbols = con.query(f"SELECT DISTINCT symbol FROM read_parquet('{parquet_path}', union_by_name=True) WHERE trade_date = '{latest_date}'").df()['symbol'].tolist()
    
    print(f"Is AEGISLOG active? {'AEGISLOG' in active_symbols}")
    print(f"Is AEGISCHEM active? {'AEGISCHEM' in active_symbols}")
    
    # 4. Filter
    df = df[df['symbol'].isin(active_symbols)]
    print(f"Rows after active filter: {len(df)}")
    
    # 5. Calculate Metrics
    if not df.empty:
        group = df.sort_values('trade_date')
        high_52w = group.tail(252)['adj_close'].max()
        last_price = group['adj_close'].iloc[-1]
        dist = ((high_52w - last_price) / high_52w) * 100
        print(f"\nAEGISLOG Stats:")
        print(f"  Last Price: {last_price:.2f}")
        print(f"  52W High: {high_52w:.2f}")
        print(f"  Distance: {dist:.2f}%")

if __name__ == "__main__":
    test_aegis_mapping()
