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
                adj_close as close, 
                adj_high as high, 
                adj_low as low, 
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

    def _run_ema_crossover(self, min_price=30, min_volume=100000):
        """
        EMA CROSSOVER FROM BELOW
        ========================
        Finds stocks where today's close crossed ABOVE the 20-day EMA,
        having been BELOW it yesterday.

        Why hybrid approach?
          EMA is recursive: EMA_t = close * k + EMA_(t-1) * (1-k)
          This CANNOT be computed in pure SQL. We use DuckDB for fast data
          loading, then pandas ewm() for accurate EMA computation.

        Crossover logic:
          - Yesterday: close < EMA(20)  (was below)
          - Today:     close > EMA(20)  (crossed above)
          - Extra:     close > SMA(50)  (intermediate uptrend filter)
        """
        # Step 1: Use DuckDB to load last 60 trading days of OHLCV for all stocks
        #         (60 days gives ~40 days of EMA warm-up + 20 days of valid signal)
        query = f"""
            SELECT symbol, trade_date, adj_close as close, adj_high as high, adj_low as low, volume, deliv_pct
            FROM {self.read_cmd}
            WHERE trade_date >= (
                SELECT MAX(trade_date) - INTERVAL '90 days' FROM {self.read_cmd}
            )
            AND close > {min_price}
            ORDER BY symbol, trade_date
        """

        try:
            df = self.con.query(query).df()
        except Exception as e:
            logging.error(f"Data load failed: {e}")
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        # Step 2: Compute true 20-day EMA per symbol using pandas
        results = []
        for symbol, group in df.groupby('symbol'):
            g = group.sort_values('trade_date').copy()

            if len(g) < 25:  # Need enough data for EMA warm-up
                continue

            # True EMA: pandas ewm(span=20) uses k = 2/(20+1)
            g['ema_20'] = g['close'].ewm(span=20, adjust=False).mean()
            g['sma_50'] = g['close'].rolling(50, min_periods=30).mean()
            g['avg_vol_20'] = g['volume'].rolling(20).mean()

            # Get last two rows (yesterday and today)
            if len(g) < 2:
                continue
            today = g.iloc[-1]
            yesterday = g.iloc[-2]

            # Crossover: yesterday BELOW ema, today ABOVE ema
            if yesterday['close'] < yesterday['ema_20'] and today['close'] > today['ema_20']:
                # Volume filter
                if today['volume'] < min_volume:
                    continue
                # Optional: intermediate trend filter (close > SMA50)
                if pd.notna(today['sma_50']) and today['close'] < today['sma_50']:
                    continue

                rvol = today['volume'] / today['avg_vol_20'] if today['avg_vol_20'] > 0 else 0
                pct_chg = (today['close'] - yesterday['close']) / yesterday['close'] * 100
                dist_ema = (today['close'] - today['ema_20']) / today['ema_20'] * 100

                results.append({
                    'symbol': symbol,
                    'trade_date': today['trade_date'],
                    'close': round(today['close'], 2),
                    'ema_20': round(today['ema_20'], 2),
                    'dist_ema_pct': round(dist_ema, 2),
                    'chg_pct': round(pct_chg, 2),
                    'rvol': round(rvol, 2),
                    'deliv_pct': round(today['deliv_pct'], 2) if pd.notna(today['deliv_pct']) else 0,
                    'sma_50': round(today['sma_50'], 2) if pd.notna(today['sma_50']) else 0,
                })

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        # Exclude ETFs/Funds
        etf_patterns = ['LIQUID', 'GOLD', 'SILVER', 'NIFTY', 'BEES', 'ETF', 'SETF', 'BIRET']
        # Exclude government securities (e.g., 648GS2035)
        result_df = result_df[~result_df['symbol'].str.match(r'^\d')]
        for pat in etf_patterns:
            result_df = result_df[~result_df['symbol'].str.contains(pat, case=False)]

        return result_df.sort_values('rvol', ascending=False).head(50).reset_index(drop=True)

    def run_scan(self, strategy='MOMENTUM', min_price=30, min_volume=100000, rvol_th=1.5, deliv_th=25):
        """
        Executes the scan based on the selected strategy.
        Strategies:
        - MOMENTUM: Trend Alignment + Volume Blast (Breakouts)
        - VCP: Trend Alignment + Low Volatility (Contraction/Squeeze)
        - PULLBACK: Uptrend + Price near SMA20/50 (Buy the dip)
        - LINEAR_PULLBACK: Mathematically linear uptrend + Pullback to 20-day MA
        - EMA_CROSSOVER: Close crosses above 20 EMA from below
        """
        logging.info(f"🚀 Running Scan with Strategy: {strategy}")

        # EMA_CROSSOVER uses a separate hybrid DuckDB+pandas path
        if strategy == 'EMA_CROSSOVER':
            return self._run_ema_crossover(min_price=min_price, min_volume=min_volume)
        
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
                        choices=['MOMENTUM', 'VCP', 'PULLBACK', 'LINEAR_PULLBACK', 'EMA_CROSSOVER'], 
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

