import os
import logging
import argparse
import pandas as pd
import numpy as np
import duckdb

from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("linearity_scanner.log"),
        logging.StreamHandler()
    ]
)

class LinearityScanner:
    # Registry of symbol changes to merge history (New Symbol: Old Symbol)
    SYMBOL_MAPPING = {
        'AEGISLOG': 'AEGISCHEM',
        'ACUTAAS': 'AMIORG',
        'ZYDUSLIFE': 'CADILAHC',
        'MOTHERSON': 'MOTHERSUMI',
        'TATACONSUM': 'TATAGLOBAL'
    }

    def __init__(self, config):
        self.config = config
        self.con = duckdb.connect(database=':memory:')
        self.parquet_path = config.get('parquet_path', 'nse_master_adjusted_2014_onwards.parquet')
        
    def load_data(self):
        """
        Load recent data for all stocks using DuckDB.
        """
        logging.info(f"Loading data from {self.parquet_path}...")
        
        # Pull ~350 days to ensure we have enough data for 200-day Moving Averages
        query = f"""
            SELECT symbol, trade_date, adj_close, volume
            FROM read_parquet('{self.parquet_path}', union_by_name=True)
            WHERE trade_date >= (
                SELECT MAX(trade_date) - INTERVAL '730 days' 
                FROM read_parquet('{self.parquet_path}', union_by_name=True)
            )
            ORDER BY symbol, trade_date
        """
        try:
            df = self.con.query(query).df()
            
            # Apply Symbol Mapping for History Merging
            for new_sym, old_sym in self.SYMBOL_MAPPING.items():
                df['symbol'] = df['symbol'].replace(old_sym, new_sym)
            
            # Active Stock Filter: Only keep symbols trading on the latest available date
            latest_date = df['trade_date'].max()
            active_symbols = df[df['trade_date'] == latest_date]['symbol'].unique()
            df = df[df['symbol'].isin(active_symbols)]
            
            # Ensure chronological order and drop any potential duplicates from mapping
            df = df.sort_values(['symbol', 'trade_date']).drop_duplicates(subset=['symbol', 'trade_date'])
            
            logging.info(f"Loaded {len(df)} rows across {df['symbol'].nunique()} symbols (after mapping & active filter).")
            return df
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            return pd.DataFrame()

    def compute_metrics(self, group):
        """
        Compute linearity metrics for a single stock's price series.
        """
        n = self.config['lookback']
        if len(group) < n:
            return None
        
        # Get the last N rows
        data = group.tail(n).copy()
        prices = data['adj_close'].values
        volumes = data['volume'].values
        
        # 1. R-Squared & Slope
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        r_value = np.corrcoef(x, prices)[0, 1]
        r2 = r_value**2
        
        # 2. Kaufman's Efficiency Ratio (ER)
        # ER = (Close_n - Close_0) / Sum(|PriceChange_i|)
        price_diffs = np.abs(np.diff(prices))
        total_volatility = np.sum(price_diffs)
        net_change = prices[-1] - prices[0]
        er = net_change / total_volatility if total_volatility != 0 else 0
        
        # 3. Net Price Change %
        net_change_pct = (net_change / prices[0]) * 100
        
        # 4. Max Drawdown % (MAE)
        rolling_max = np.maximum.accumulate(prices)
        drawdowns = (prices - rolling_max) / rolling_max * 100
        max_dd = np.abs(np.min(drawdowns))
        
        # 5. Average Volume and Turnover (Daily Value)
        avg_vol = np.mean(volumes)
        # Turnover = Avg(Price * Volume)
        daily_values = prices * volumes
        avg_daily_value = np.mean(daily_values)

        # 6. 1-Day Change %
        # Take change from the absolute last two days in the full group
        one_day_chg_pct = ((prices[-1] - prices[-2]) / prices[-2] * 100) if len(prices) > 1 else 0

        # 7. MA Relationship (20, 50, 200 SMA)
        # Calculate signals based on the full group to ensure indices exist
        full_prices = group['adj_close']
        sma_20 = full_prices.rolling(window=20, min_periods=15).mean().iloc[-1]
        sma_50 = full_prices.rolling(window=50, min_periods=40).mean().iloc[-1]
        sma_200 = full_prices.rolling(window=200, min_periods=180).mean().iloc[-1]
        
        last_price = prices[-1]

        # 8. MA Relationship (Yes/No)
        a_200 = "Yes" if not np.isnan(sma_200) and last_price >= sma_200 else ("No" if not np.isnan(sma_200) else "N/A")
        a_50  = "Yes" if not np.isnan(sma_50)  and last_price >= sma_50  else ("No" if not np.isnan(sma_50)  else "N/A")
        a_20  = "Yes" if not np.isnan(sma_20)  and last_price >= sma_20  else ("No" if not np.isnan(sma_20)  else "N/A")

        # 9. 20-day EMA Distance
        ema_20_series = data['adj_close'].ewm(span=20, adjust=False).mean()
        last_ema_20 = ema_20_series.iloc[-1]
        dist_ema_20_pct = ((prices[-1] - last_ema_20) / last_ema_20) * 100
        
        # 9. Max Single Day Drop
        daily_pct_change = np.diff(prices) / prices[:-1] * 100
        max_single_day_drop = np.min(daily_pct_change) if len(daily_pct_change) > 0 else 0
        
        # 10. Composite Score
        score = (0.55 * r2) + (0.35 * max(0, er)) + (0.10 * min(1, net_change_pct / 20))
        
        # 11. RS vs Nifty50 (20-day)
        nifty_20d_return = self.config.get('nifty_20d_return')
        rs_vs_nifty = np.nan
        full_prices = group['adj_close'].values
        if nifty_20d_return and len(full_prices) >= 21:
            stock_20d_return = full_prices[-1] / full_prices[-21]
            rs_vs_nifty = stock_20d_return / nifty_20d_return
        
        # 12. TradingView Link
        base_symbol = group['symbol'].iloc[0]
        # TradingView uses _ instead of & and - in many NSE symbols
        tv_symbol = base_symbol.replace('&', '_').replace('-', '_')
        tv_link = f"https://www.tradingview.com/chart/?symbol=NSE:{tv_symbol}"
        
        # 13. TV Helper
        tv_helper = f"NSE:{tv_symbol},"
        
        return {
            'symbol': base_symbol,
            'tv_link': tv_link,
            'TV Helper': tv_helper,
            'lookback_days': n,
            'r2': round(r2, 2),
            'er': round(er, 2),
            'slope': round(slope, 2),
            'net_change_pct': round(net_change_pct, 2),
            'one_day_chg_pct': round(one_day_chg_pct, 2),
            'Above 200MA': a_200,
            'Above 50MA': a_50,
            'Above 20MA': a_20,
            'max_drawdown_pct': round(max_dd, 2),
            'avg_volume': int(avg_vol),
            'Traded Value (Cr)': round(avg_daily_value / 10_000_000, 2),
            'ema_20_dist_pct': round(dist_ema_20_pct, 2),
            'max_single_day_drop': round(max_single_day_drop, 2),
            'composite_score': round(score, 2),
            'RS (vs Nifty50)': round(rs_vs_nifty, 3) if not np.isnan(rs_vs_nifty) else '',
            'last_close': round(prices[-1], 2),
            'date_range': f"{data['trade_date'].iloc[0].date()} to {data['trade_date'].iloc[-1].date()}"
        }

    def run_scan(self):
        df = self.load_data()
        if df.empty:
            return pd.DataFrame()
        
        results = []
        unique_symbols = df['symbol'].unique()
        total = len(unique_symbols)
        
        logging.info(f"Scanning {total} symbols...")
        
        for i, (symbol, group) in enumerate(df.groupby('symbol')):
            if (i+1) % 100 == 0:
                logging.info(f"Progress: {i+1}/{total} symbols processed.")
            
            metrics = self.compute_metrics(group)
            if metrics:
                results.append(metrics)
        
        results_df = pd.DataFrame(results)
        return results_df

    def filter_and_rank(self, results_df):
        if results_df.empty:
            return pd.DataFrame()
        
        # Apply Filters (ONLY Turnover and Symbol Pattern as per new request)
        mask = (
            (results_df['Traded Value (Cr)'] >= self.config['min_turnover'] / 10_000_000)
        )
        
        filtered_df = results_df[mask].copy()
        
        # Exclude ETFs, BEES, Funds, Gilt Funds, SGBs, Indices, PP, RE, etc.
        etf_patterns = [
            'LIQUID', 'LIQID', 'GOLD', 'SILVER', 'NIFTY', 'BEES', 'ETF', 'SETF', 'BIRET', 
            'GILT', 'GS', 'FUND', 'INVIT', 'REIT', 'CASH', 'SGB', 'INDEX', 'MIDCAP', 
            'SENSEX', 'BANKNIFTY', 'FINNIFTY', '-PP', '-RE'
        ]
        # Also exclude symbols starting with a number (common for G-Secs)
        filtered_df = filtered_df[~filtered_df['symbol'].str.match(r'^\d')]
        for pat in etf_patterns:
            filtered_df = filtered_df[~filtered_df['symbol'].str.contains(pat, case=False)]
            
        # Rank by composite score
        filtered_df = filtered_df.sort_values(by='composite_score', ascending=False)
        
        return filtered_df

    def save_charts(self, filtered_df, raw_df):
        """
        Save charts for the top 10 stocks.
        """
        output_dir = self.config['output_dir'] / "results_charts"
        output_dir.mkdir(exist_ok=True)
        
        top_10 = filtered_df.head(10)
        logging.info(f"Generating charts for top {len(top_10)} stocks...")
        
        for _, row in top_10.iterrows():
            symbol = row['symbol']
            group = raw_df[raw_df['symbol'] == symbol].tail(row['lookback_days'])
            
            prices = group['adj_close'].values
            x = np.arange(len(prices))
            slope, intercept = np.polyfit(x, prices, 1)
            r_value = np.corrcoef(x, prices)[0, 1]
            reg_line = slope * x + intercept
            
            plt.figure(figsize=(10, 6))
            plt.plot(group['trade_date'], prices, label='Adj Close', color='blue', linewidth=2)
            plt.plot(group['trade_date'], reg_line, label=f'Regression (R2={row["r2"]})', color='red', linestyle='--')
            plt.title(f"{symbol} - Linearity Scan (Score: {row['composite_score']})")
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            safe_symbol = symbol.replace('.', '_')
            plt.savefig(output_dir / f"{safe_symbol}.png")
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="NSE Stock Linearity Scanner")
    parser.add_argument("--min_r2", type=float, default=0.85, help="Minimum R-squared")
    parser.add_argument("--min_er", type=float, default=0.70, help="Minimum Efficiency Ratio")
    parser.add_argument("--min_net_change", type=float, default=8.0, help="Minimum net price change %%")
    parser.add_argument("--min_volume", type=int, default=500000, help="Minimum average volume")
    parser.add_argument("--min_turnover", type=float, default=10000000.0, help="Minimum average daily turnover (Price * Volume)")
    parser.add_argument("--max_drop", type=float, default=-5.0, help="Max single day drop allowed")
    parser.add_argument("--charts", action="store_true", help="Generate charts for top 10")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    # Handle Desktop path
    output_path = Path(args.output_dir).expanduser()
    if "Desktop" in args.output_dir:
        output_path = Path("~/Desktop").expanduser()
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    xlsx_file = output_path / "Linearity Scans.xlsx"
    tmp_file = output_path / "Linearity Scans.tmp.xlsx"
    
    # Fetch Nifty50 20d return
    nifty_20d_return = None
    try:
        import yfinance as yf
        nifty_df = yf.download('^NSEI', period='60d', interval='1d', progress=False)
        nifty_close = nifty_df['Close'].dropna().sort_index()
        if len(nifty_close) >= 21:
            n_curr = float(nifty_close.iloc[-1].iloc[0]) if isinstance(nifty_close.iloc[-1], pd.Series) else float(nifty_close.iloc[-1])
            n_20d  = float(nifty_close.iloc[-21].iloc[0]) if isinstance(nifty_close.iloc[-21], pd.Series) else float(nifty_close.iloc[-21])
            nifty_20d_return = n_curr / n_20d
    except Exception as e:
        logging.warning(f"Failed to fetch Nifty data for RS: {e}")
    
    try:
        with pd.ExcelWriter(tmp_file, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            for lookback in [60, 120]:
                config = {
                    'lookback': lookback,
                    'min_r2': args.min_r2,
                    'min_er': args.min_er,
                    'min_net_change': args.min_net_change,
                    'min_volume': args.min_volume,
                    'min_turnover': args.min_turnover,
                    'max_single_day_drop_limit': args.max_drop,
                    'parquet_path': 'nse_master_adjusted_2014_onwards.parquet',
                    'nifty_20d_return': nifty_20d_return,
                    'output_dir': output_path
                }
                
                logging.info(f"--- Running Linearity Scan (Lookback: {lookback}) ---")
                scanner = LinearityScanner(config)
                raw_df = scanner.load_data()
                
                if raw_df.empty:
                    logging.error(f"No data found for {lookback}D. Skipping.")
                    continue

                results_df = scanner.run_scan()
                filtered_df = scanner.filter_and_rank(results_df)
                
                sheet_name = f'Lookback {lookback}'
                filtered_df.to_excel(writer, index=False, sheet_name=sheet_name)
                worksheet = writer.sheets[sheet_name]
                
                if not filtered_df.empty:
                    (max_row, max_col) = filtered_df.shape
                    column_settings = [{'header': str(column)} for column in filtered_df.columns]
                    
                    header_fmt = workbook.add_format({
                        'bold': True,
                        'align': 'center',
                        'valign': 'vcenter',
                        'border': 1,
                        'bg_color': '#F2F2F2'
                    })

                    for col_num, value in enumerate(filtered_df.columns.values):
                        worksheet.write(0, col_num, str(value), header_fmt)
                    
                    worksheet.add_table(0, 0, max_row, max_col - 1, {
                        'columns': column_settings,
                        'style': 'Table Style Light 1',
                        'header_row': True
                    })
                    
                    worksheet.freeze_panes(1, 0)
                    
                    decimal_cols = {'r2', 'er', 'slope', 'net_change_pct', 'one_day_chg_pct', 
                                    'max_drawdown_pct', 'Traded Value (Cr)', 'ema_20_dist_pct', 
                                    'max_single_day_drop', 'composite_score', 'last_close', 'RS (vs Nifty50)'}
                    fmt2d = workbook.add_format({'num_format': '0.00'})
                    fmt3d = workbook.add_format({'num_format': '0.000'})
                    
                    for i, col in enumerate(filtered_df.columns):
                        column_len = max(filtered_df[col].astype(str).map(len).max(), len(str(col))) + 2
                        if col == 'RS (vs Nifty50)':
                            worksheet.set_column(i, i, min(column_len, 50), fmt3d)
                        elif col in decimal_cols:
                            worksheet.set_column(i, i, min(column_len, 50), fmt2d)
                        else:
                            worksheet.set_column(i, i, min(column_len, 50))
                            
                    # --- Generate TV Watchlists Sheet ---
                    symbols_list = filtered_df['symbol'].tolist()
                    tv_symbols = [f"NSE:{str(s).replace('&', '_').replace('-', '_')}" for s in symbols_list]
                    chunks = [",".join(tv_symbols[i:i+30]) for i in range(0, len(tv_symbols), 30)]
                    tv_df = pd.DataFrame({'TV Watchlist (Copy 30 at a time)': chunks})
                    tv_sheet_name = f'TV Watchlist ({lookback}D)'
                    tv_df.to_excel(writer, index=False, sheet_name=tv_sheet_name)
                    writer.sheets[tv_sheet_name].set_column(0, 0, 150)
                
                print(f"\n--- TOP LINEAR TRENDS ({lookback}D) ---")
                if not filtered_df.empty:
                    print(filtered_df[['symbol', 'r2', 'one_day_chg_pct', 'Above 200MA', 'Above 50MA', 'Above 20MA']].head(15).to_string(index=False))
                else:
                    print("No matches found.")
            
        if xlsx_file.exists():
            xlsx_file.unlink()
        tmp_file.rename(xlsx_file)
        logging.info(f"Consolidated Linearity scans saved to {xlsx_file}")
        
        # Copy to Desktop if directory exists and path is different
        desktop_dir = Path.home() / "Desktop" / "Linearity Scan"
        desktop_file = desktop_dir / "Linearity Scans.xlsx"
        if desktop_dir.exists() and xlsx_file.resolve() != desktop_file.resolve():
            import shutil
            shutil.copy2(str(xlsx_file), str(desktop_file))
        
        # open automatically 
        import subprocess
        subprocess.Popen(['open', str(xlsx_file)])

    except Exception as e:
        logging.error(f"Failed to save Excel file: {e}")
        if tmp_file.exists():
            tmp_file.unlink()

if __name__ == "__main__":
    main()
