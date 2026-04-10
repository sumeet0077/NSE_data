import duckdb
import pandas as pd
import logging
import argparse
import sys

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class DuckDBScanner:
    def __init__(self, parquet_file="nse_master_adjusted_2014_onwards.parquet"):
        self.parquet_file = parquet_file
        self.con = duckdb.connect(database=':memory:')
        self.read_cmd = f"read_parquet('{self.parquet_file}', union_by_name=True)"

    def _build_base_query(self):
        """
        CTE for technical indicators used across all strategies.
        Includes: SMAs, Volume, Volatility, 52W High, R² Linearity, Regression Slope.
        """
        return f"""
        WITH Numbered AS (
            SELECT 
                *,
                -- Integer index needed for REGR_R2/REGR_SLOPE (x-axis = time as integer)
                ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trade_date) as row_idx
            FROM {self.read_cmd}
        ),
        BaseData AS (
            SELECT 
                symbol, 
                trade_date, 
                close, 
                high,
                low,
                volume, 
                deliv_pct,
                row_idx,

                -- Trend Indicators (SMAs)
                AVG(close) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as sma_20,
                AVG(close) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as sma_50,
                AVG(close) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) as sma_200,
                
                -- Volume Stats
                AVG(volume) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as vol_sma_20,
                
                -- Volatility
                MAX(high) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as high_20,
                MIN(low) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as low_20,
                
                -- 52 Week High (approx 252 trading days)
                MAX(high) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 252 PRECEDING AND CURRENT ROW) as high_252,
                
                -- Linear Regression (50-day window)
                -- R² measures how "linear" the price movement is (1.0 = perfect line)
                REGR_R2(close, row_idx) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as r_squared,
                -- Slope measures direction and steepness (positive = uptrend)
                REGR_SLOPE(close, row_idx) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as trend_slope,

                -- Price Change
                (close - LAG(close, 1) OVER (PARTITION BY symbol ORDER BY trade_date)) / LAG(close, 1) OVER (PARTITION BY symbol ORDER BY trade_date) * 100 as pct_change
            FROM Numbered
        ),
        EnhancedIndicators AS (
            SELECT 
                *,
                -- Relative Volume
                (volume / NULLIF(vol_sma_20, 0)) as rvol,
                
                -- Proximity to 52W High (%)
                ((close - high_252) / high_252) * 100 as dist_from_high,

                -- Where is price relative to SMA20? (negative = below)
                ((close - sma_20) / sma_20) * 100 as dist_from_sma20,

                -- Range compression (Squeeze proxy)
                ((high_20 - low_20) / sma_20) * 100 as range_compression_pct
            FROM BaseData
            WHERE trade_date = (SELECT MAX(trade_date) FROM {self.read_cmd})
        )
        """

    def run_scan(self, strategy='MOMENTUM', min_price=30, min_volume=100000, rvol_th=1.5, deliv_th=25):
        """
        Executes the scan based on the selected strategy.
        Strategies:
        - MOMENTUM: Trend Alignment + Volume Blast (Breakouts)
        - VCP: Trend Alignment + Low Volatility (Contraction/Squeeze)
        - PULLBACK: Uptrend + Price near SMA20/50 (Buy the dip)
        - LINEAR_PULLBACK: Mathematically linear uptrend + Pullback to 20-day MA
        """
        logging.info(f"🚀 Running Scan with Strategy: {strategy}")
        
        base_cte = self._build_base_query()
        
        if strategy == 'MOMENTUM':
            # Rule: Uptrend + Near Highs + Volume Burst
            filters = f"""
                AND close > sma_20 AND sma_20 > sma_50 AND sma_50 > sma_200
                AND dist_from_high > -25
                AND rvol > {rvol_th} 
                AND deliv_pct > {deliv_th}
                AND pct_change > 0
            ORDER BY rvol DESC
            """
        elif strategy == 'VCP':
            # Rule: Uptrend + Consolidation Near Highs + Low Vol
            filters = f"""
                AND close > sma_200
                AND dist_from_high > -20
                AND range_compression_pct < 10
                AND rvol < 1.0
            ORDER BY range_compression_pct ASC
            """
        elif strategy == 'PULLBACK':
            # Rule: Primary Uptrend + Short term dip
            filters = f"""
                AND close > sma_200
                AND close > sma_50
                AND close < sma_20
                AND close > (sma_20 * 0.95)
                AND pct_change < 0
            ORDER BY rvol DESC
            """
        elif strategy == 'LINEAR_PULLBACK':
            # ===================================================================
            # LINEAR TREND + PULLBACK TO 20-DAY MA
            # ===================================================================
            # Empirical basis:
            #   - R² (coefficient of determination) measures how closely price
            #     follows a straight line over time. R² > 0.65 = strong linearity.
            #   - REGR_SLOPE > 0 confirms the line slopes upward.
            #   - Pullback: price has dipped TO or slightly below the 20-day SMA,
            #     but NOT crashed through the 50-day SMA (trend intact).
            #
            # What this catches:
            #   Stocks moving up in a smooth, steady channel that have temporarily
            #   pulled back to their short-term average — a classic "buy the dip
            #   in a trend" setup.
            #
            # What this rejects:
            #   - Choppy, oscillating stocks (low R²)
            #   - Parabolic/exponential moves (often lower R² due to curvature)
            #   - Stocks in downtrends (negative slope)
            #   - Stocks that have crashed through support (below SMA50)
            # ===================================================================
            filters = f"""
                AND r_squared > 0.65          -- Price follows a straight line (65%+ variance explained)
                AND trend_slope > 0.10        -- Meaningful upward slope (rejects liquid funds ~0.02)
                AND close > sma_200           -- Primary trend intact
                AND close > sma_50            -- Intermediate trend intact
                AND dist_from_sma20 BETWEEN -3 AND 1  -- Price is at or just below SMA20 (pullback zone)
                -- Exclude ETFs/Funds
                AND symbol NOT LIKE '%%LIQUID%%'
                AND symbol NOT LIKE '%%GOLD%%'
                AND symbol NOT LIKE '%%SILVER%%'
                AND symbol NOT LIKE '%%NIFTY%%'
                AND symbol NOT LIKE '%%BEES%%'
                AND symbol NOT LIKE '%%ETF%%'
                AND symbol NOT LIKE '%%SETF%%'
                AND symbol NOT LIKE 'LIQ%%'
                AND symbol NOT LIKE 'EGOLD'
                AND symbol NOT LIKE '%%BIRET%%'
            ORDER BY r_squared DESC
            """
        else:
            logging.error(f"Unknown Strategy: {strategy}")
            return pd.DataFrame()

        # Select columns based on strategy
        if strategy == 'LINEAR_PULLBACK':
            select_cols = """
                symbol, 
                trade_date, 
                close, 
                ROUND(r_squared, 3) as r_squared,
                ROUND(trend_slope, 2) as slope,
                ROUND(dist_from_sma20, 2) as dist_sma20_pct,
                ROUND(rvol, 2) as rvol,
                ROUND(deliv_pct, 2) as deliv_pct,
                ROUND(sma_20, 2) as sma_20,
                ROUND(sma_50, 2) as sma_50
            """
        else:
            select_cols = """
                symbol, 
                trade_date, 
                close, 
                pct_change as chg_pct,
                rvol,
                deliv_pct,
                range_compression_pct as squeeze_idx,
                sma_20,
                sma_50
            """

        final_query = f"""
        {base_cte}
        SELECT {select_cols}
        FROM EnhancedIndicators
        WHERE 
            close > {min_price} 
            AND volume > {min_volume}
            {filters}
        LIMIT 50;
        """
        
        try:
            return self.con.query(final_query).df()
        except Exception as e:
            logging.error(f"Query Failed: {e}")
            return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="DuckDB NSE Scanner")
    parser.add_argument("--strategy", type=str, default="MOMENTUM", 
                        choices=['MOMENTUM', 'VCP', 'PULLBACK', 'LINEAR_PULLBACK'], 
                        help="Scan Strategy")
    parser.add_argument("--price", type=float, default=30, help="Min Price")
    parser.add_argument("--volume", type=int, default=100000, help="Min Volume")
    parser.add_argument("--rvol", type=float, default=1.5, help="RVOL Threshold (for Momentum)")
    parser.add_argument("--deliv", type=float, default=25, help="Delivery %% Threshold")
    
    args = parser.parse_args()
    
    scanner = DuckDBScanner()
    df = scanner.run_scan(
        strategy=args.strategy,
        min_price=args.price,
        min_volume=args.volume,
        rvol_th=args.rvol,
        deliv_th=args.deliv
    )
    
    if not df.empty:
        try:
            from tabulate import tabulate
            print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".2f"))
        except ImportError:
            print(df.to_string())
        print(f"\n✅ Found {len(df)} matches.")
    else:
        print("⚠️ No matches found.")

if __name__ == "__main__":
    main()

