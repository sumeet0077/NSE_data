import os
import logging
import argparse
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from datetime import datetime

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("52week_high_scanner.log"),
        logging.StreamHandler()
    ]
)

class FiftyTwoWeekHighScanner:
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
        Load historical data using DuckDB.
        """
        logging.info(f"Loading data from {self.parquet_path}...")
        
        # Pull ~800 calendar days to ensure we have a full 52-week high lookback
        query = f"""
            SELECT symbol, trade_date, adj_close, volume
            FROM read_parquet('{self.parquet_path}', union_by_name=True)
            WHERE trade_date >= (
                SELECT MAX(trade_date) - INTERVAL '800 days' 
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
        Compute 52-week high metrics.
        """
        # We need at least 252 trading days for a proper 52-week high
        # But we can work with less for newer listings if we want to show current high
        if len(group) < 10:
            return None
        
        # 52-Week High Calculation (252 trading days)
        # Using adj_close to account for corporate actions
        year_data = group.tail(252)
        high_52w = year_data['adj_close'].max()
        last_price = group['adj_close'].iloc[-1]
        
        dist_from_52w_high_pct = ((high_52w - last_price) / high_52w) * 100
        
        # Average Daily Turnover (Last 20 days for current liquidity)
        recent_data = group.tail(20)
        avg_daily_value = (recent_data['adj_close'] * recent_data['volume']).mean()
        
        # Simple MA signals for context
        sma_20 = group['adj_close'].rolling(20).mean().iloc[-1]
        sma_50 = group['adj_close'].rolling(50).mean().iloc[-1]
        sma_200 = group['adj_close'].rolling(200, min_periods=180).mean().iloc[-1]
        
        ma_signals = []
        ma_signals.append("Above 200 MA" if not np.isnan(sma_200) and last_price >= sma_200 else "Below 200 MA")
        ma_signals.append("Above 50 MA" if not np.isnan(sma_50) and last_price >= sma_50 else "Below 50 MA")
        ma_signals.append("Above 20 MA" if not np.isnan(sma_20) and last_price >= sma_20 else "Below 20 MA")
        ma_status = ", ".join(ma_signals)

        # TradingView Link
        base_symbol = group['symbol'].iloc[0]
        tv_symbol = base_symbol.replace('&', '_').replace('-', '_')
        tv_link = f"https://www.tradingview.com/chart/?symbol=NSE:{tv_symbol}"
        
        return {
            'symbol': base_symbol,
            'tv_link': tv_link,
            'last_price': round(last_price, 2),
            'high_52w': round(high_52w, 2),
            'dist_from_52w_high_pct': round(dist_from_52w_high_pct, 2),
            'avg_daily_turnover_lakhs': round(avg_daily_value / 100000, 2),
            'ma_status': ma_status,
            'latest_date': group['trade_date'].iloc[-1].date()
        }

    def run_scan(self):
        df = self.load_data()
        if df.empty:
            return pd.DataFrame()
        
        results = []
        unique_symbols = df['symbol'].unique()
        total = len(unique_symbols)
        
        logging.info(f"Scanning {total} symbols for 52-week highs...")
        
        for i, (symbol, group) in enumerate(df.groupby('symbol')):
            if (i+1) % 500 == 0:
                logging.info(f"Progress: {i+1}/{total} symbols processed.")
            
            metrics = self.compute_metrics(group)
            if metrics:
                results.append(metrics)
        
        results_df = pd.DataFrame(results)
        return results_df

    def filter_and_rank(self, results_df):
        if results_df.empty:
            return pd.DataFrame()
        
        # Apply User Filters:
        # 1. Within 25% of 52-week high
        # 2. Avg Daily Turnover >= 50 Lakhs
        mask = (
            (results_df['dist_from_52w_high_pct'] <= 25) &
            (results_df['avg_daily_turnover_lakhs'] >= 50)
        )
        
        filtered_df = results_df[mask].copy()
        
        # Exclude non-equity patterns, PP, and RE symbols
        etf_patterns = [
            'LIQUID', 'LIQID', 'GOLD', 'SILVER', 'NIFTY', 'BEES', 'ETF', 'SETF', 'BIRET', 
            'GILT', 'GS', 'FUND', 'INVIT', 'REIT', 'CASH', 'SGB', 'INDEX', 'MIDCAP', 
            'SENSEX', 'BANKNIFTY', 'FINNIFTY', '-PP', '-RE'
        ]
        filtered_df = filtered_df[~filtered_df['symbol'].str.match(r'^\d')]
        for pat in etf_patterns:
            filtered_df = filtered_df[~filtered_df['symbol'].str.contains(pat, case=False)]
            
        # Sort by distance to high (closest first)
        filtered_df = filtered_df.sort_values(by='dist_from_52w_high_pct')
        
        return filtered_df

def main():
    parser = argparse.ArgumentParser(description="NSE 52-Week High Scanner")
    parser.add_argument("--threshold", type=float, default=25.0, help="Max distance from 52w high %%")
    parser.add_argument("--min_turnover_lakhs", type=float, default=50.0, help="Min avg daily turnover in Lakhs")
    parser.add_argument("--output_dir", type=str, default="~/Desktop/52week high scanner", help="Output directory")
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'parquet_path': 'nse_master_adjusted_2014_onwards.parquet',
        'output_dir': output_path,
        'threshold': args.threshold,
        'min_turnover': args.min_turnover_lakhs
    }
    
    scanner = FiftyTwoWeekHighScanner(config)
    results_df = scanner.run_scan()
    filtered_df = scanner.filter_and_rank(results_df)
    
    # Save to Excel
    today = datetime.now().strftime("%Y%m%d")
    xlsx_file = output_path / f"52week_high_scan_{today}.xlsx"
    
    try:
        with pd.ExcelWriter(xlsx_file, engine='xlsxwriter') as writer:
            filtered_df.to_excel(writer, index=False, sheet_name='52W High Scan')
            
            workbook = writer.book
            worksheet = writer.sheets['52W High Scan']
            
            (max_row, max_col) = filtered_df.shape
            column_settings = [{'header': column} for column in filtered_df.columns]
            
            header_fmt = workbook.add_format({
                'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1, 'bg_color': '#E6F0FF'
            })
            for col_num, value in enumerate(filtered_df.columns.values):
                worksheet.write(0, col_num, value, header_fmt)
            
            worksheet.add_table(0, 0, max_row, max_col - 1, {
                'columns': column_settings,
                'style': 'Table Style Light 1',
                'header_row': True
            })
            worksheet.freeze_panes(1, 0)
            
            for i, col in enumerate(filtered_df.columns):
                column_len = max(filtered_df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, min(column_len, 50))

        logging.info(f"Results saved to {xlsx_file}")
    except Exception as e:
        logging.error(f"Failed to save Excel file: {e}")
    
    print(f"\n--- STOCKS WITHIN {args.threshold}% OF 52W HIGH (Min {args.min_turnover_lakhs}L Turnover) ---")
    print(filtered_df[['symbol', 'dist_from_52w_high_pct', 'last_price', 'high_52w', 'avg_daily_turnover_lakhs']].head(50).to_string(index=False))

if __name__ == "__main__":
    main()
