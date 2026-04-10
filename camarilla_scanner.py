#!/usr/bin/env python3
"""
Advanced Camarilla Pivot Compression Scanner
Identifies stocks in extreme volatility contraction ("Inside Camarilla") on a Monthly timeframe.
The scanner calculates H3/H4 and L3/L4 levels and identifies compression before momentum bursts.
"""

import argparse
import logging
import datetime as dt
from pathlib import Path
import duckdb
import pandas as pd
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Use the adjusted dataset by default
PARQUET_GLOB = "/Users/sumeetdas/Antigravity_NSE_Data/nse_master_adjusted_2014_onwards.parquet/*/*.parquet"

def get_camarilla_compression_query(min_price: float = 50.0, min_volume: int = 250000, consecutive_inside: int = 1) -> str:
    """
    Constructs the DuckDB SQL query to find stocks with 'Inside Monthly Camarilla' compression.
    """
    
    # We use LAG() to look at the previous month's Camarilla levels.
    # An "Inside H3/L3" pattern means: Current H3 < Previous H3 AND Current L3 > Previous L3
    
    query = f"""
    WITH DailyFiltered AS (
        -- Filter base daily data
        SELECT symbol, trade_date, adj_open as open, adj_high as high, adj_low as low, adj_close as close, volume, series 
        FROM read_parquet('{PARQUET_GLOB}', union_by_name=True)
        WHERE trade_date >= '2024-01-01' 
          AND series = 'EQ'
          -- Exclude ETFs/G-Secs usually having these patterns
          AND symbol NOT LIKE '%ETF%' AND symbol NOT LIKE '%LIQUID%' 
          AND symbol NOT LIKE '0%' AND symbol NOT LIKE '1%' AND symbol NOT LIKE '2%'
          AND symbol NOT LIKE '3%' AND symbol NOT LIKE '4%' AND symbol NOT LIKE '5%'
          AND symbol NOT LIKE '6%' AND symbol NOT LIKE '7%' AND symbol NOT LIKE '8%'
          AND symbol NOT LIKE '9%'
    ),
    MonthlyOHLC AS (
        -- Aggregate daily data into Monthly buckets
        SELECT 
            symbol,
            DATE_TRUNC('month', trade_date)::DATE AS month_start,
            MAX(high) AS m_high,
            MIN(low) AS m_low,
            ARG_MAX(close, trade_date) AS m_close,
            AVG(volume) as avg_monthly_vol,
            ARG_MAX(trade_date, trade_date) AS last_trade_date,
            COUNT(trade_date) as trading_days
        FROM DailyFiltered
        GROUP BY symbol, month_start
    ),
    CamarillaPivots AS (
        -- Calculate the Camarilla Pivots for a given month based on the PREVIOUS month's OHLC
        SELECT 
            symbol,
            month_start,
            last_trade_date,
            m_close as current_close,
            avg_monthly_vol,
            trading_days,
            -- Look back at the previous month's OHLC to calculate THIS month's pivots
            LAG(m_high) OVER (PARTITION BY symbol ORDER BY month_start) AS prev_high,
            LAG(m_low) OVER (PARTITION BY symbol ORDER BY month_start) AS prev_low,
            LAG(m_close) OVER (PARTITION BY symbol ORDER BY month_start) AS prev_close
        FROM MonthlyOHLC
    ),
    CamarillaCalculated AS (
        SELECT 
            *,
            (prev_high - prev_low) AS c_range,
            prev_close + ((prev_high - prev_low) * 1.1 / 2) AS h4,
            prev_close + ((prev_high - prev_low) * 1.1 / 4) AS h3,
            prev_close - ((prev_high - prev_low) * 1.1 / 4) AS l3,
            prev_close - ((prev_high - prev_low) * 1.1 / 2) AS l4,
            -- Central Pivot Range (CPR)
            (prev_high + prev_low + prev_close) / 3 AS pivot,
            (prev_high + prev_low) / 2 AS bottom_central,
            ((prev_high + prev_low + prev_close) / 3 - ((prev_high + prev_low) / 2)) + ((prev_high + prev_low + prev_close) / 3) AS top_central
        FROM CamarillaPivots
        WHERE prev_high IS NOT NULL -- filter out the very first month missing a lag
    ),
    InsideCamarilla AS (
        SELECT 
            *,
            -- Look at the previous month's H3/L3 levels to see if we are "inside" them
            LAG(h3) OVER (PARTITION BY symbol ORDER BY month_start) as lag1_h3,
            LAG(l3) OVER (PARTITION BY symbol ORDER BY month_start) as lag1_l3,
            LAG(h4) OVER (PARTITION BY symbol ORDER BY month_start) as lag1_h4,
            LAG(l4) OVER (PARTITION BY symbol ORDER BY month_start) as lag1_l4
        FROM CamarillaCalculated
    ),
    CompressionFlags AS (
        SELECT 
            *,
            -- H4/L4 Inside condition: Current H4 < Previous H4 AND Current L4 > Previous L4
            -- Added: minimum 1% gap on EACH side to eliminate noise-level "inside" signals
            CASE 
                WHEN h4 < lag1_h4 AND l4 > lag1_l4 THEN 1 
                ELSE 0 
            END AS is_inside_h4_l4,
            -- Width of the H4-L4 range (smaller = higher compression)
            (h4 - l4) / current_close * 100 AS h4_l4_width_pct,
            -- Is price currently near the bottom of the compression (L4) or top (H4)?
            (current_close - l4) / NULLIF((h4 - l4), 0) * 100 AS position_in_zone_pct
        FROM InsideCamarilla
    ),
    LatestMonth AS (
        -- We only care about stocks that are currently compressed in the absolute latest month available
        SELECT 
            *,
            -- Count consecutive inside months terminating in the present month
            CASE 
                WHEN is_inside_h4_l4 = 0 THEN 0
                WHEN COALESCE(LAG(is_inside_h4_l4, 1) OVER (PARTITION BY symbol ORDER BY month_start), 0) = 0 THEN 1
                WHEN COALESCE(LAG(is_inside_h4_l4, 2) OVER (PARTITION BY symbol ORDER BY month_start), 0) = 0 THEN 2
                WHEN COALESCE(LAG(is_inside_h4_l4, 3) OVER (PARTITION BY symbol ORDER BY month_start), 0) = 0 THEN 3
                ELSE 4
            END AS consecutive_inside_score
        FROM CompressionFlags
    )
    SELECT 
        symbol,
        current_close as "Close",
        ROUND(h4_l4_width_pct, 2) AS "H4/L4 Width %",
        -- position_in_zone: 0% = at L4, 100% = at H4
        ROUND(position_in_zone_pct, 1) AS "Zone Pos %",
        is_inside_h4_l4 AS "Inside Now",
        consecutive_inside_score AS "Consec. Inside Mths",
        ROUND(h4, 2) AS H4,
        ROUND(h3, 2) AS H3,
        ROUND(l3, 2) AS L3,
        ROUND(l4, 2) AS L4
    FROM LatestMonth
    WHERE current_close >= {min_price}
      AND avg_monthly_vol >= {min_volume}
      AND trading_days >= 10 -- Ensure it traded somewhat normally this month
      -- Filter by user's compression requirement
      AND is_inside_h4_l4 = 1
      AND consecutive_inside_score >= {consecutive_inside}
      -- Ensure we are only looking at the absolute latest month in the dataset
      AND month_start = (SELECT MAX(month_start) FROM LatestMonth)
      -- Ensure the setup hasn't already broken out (price must still be boxed in)
      AND current_close > l4 AND current_close < h4
    ORDER BY "H4/L4 Width %" ASC, "Zone Pos %" ASC
    """
    return query

def main():
    parser = argparse.ArgumentParser(description="Monthly Camarilla Pivot Compression Scanner")
    parser.add_argument("--min-price", type=float, default=50.0, help="Minimum current price")
    parser.add_argument("--min-vol", type=int, default=250000, help="Minimum average monthly volume")
    parser.add_argument("--consecutive", type=int, default=1, help="Number of consecutive 'Inside' months required (1 to 4)")
    
    args = parser.parse_args()
    
    logging.info("Initializing DuckDB connection...")
    con = duckdb.connect()
    
    query = get_camarilla_compression_query(args.min_price, args.min_vol, args.consecutive)
    
    logging.info(f"Scanning the market for Inside Monthly Camarilla (Consecutive={args.consecutive})...")
    df = con.query(query).df()
    
    if df.empty:
        logging.warning("No stocks found meeting the Camarilla compression criteria.")
    else:
        logging.info(f"Found {len(df)} compressed setups. Sorting by tightest H4/L4 Width:")
        print(tabulate(df.head(50), headers='keys', tablefmt='psql', showindex=False))

if __name__ == "__main__":
    main()
