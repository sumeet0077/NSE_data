import pandas as pd
import numpy as np
# import pandas_ta as ta  # Removed due to Python 3.14 incompatibility
# Fallback to manual implementation if pandas_ta not present (for portability)

def calculate_indicators(df):
    """
    Compute technical indicators for momentum scanning.
    Assumes df is sorted by trade_date.
    """
    # 1. EMA 20
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # 2. Daily Range & ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_14'] = df['tr'].rolling(window=14).mean()
    df['range_pct'] = (df['high'] - df['low']) / df['open'] * 100
    
    # 3. Volume Moving Average
    df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
    
    # 4. Pivot Points (Camarilla & CPR) requires Monthly/Weekly data
    # Approximating Daily CPR for directional bias
    pivot = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
    bc = (df['high'].shift(1) + df['low'].shift(1)) / 2
    tc = (pivot - bc) + pivot
    df['cpr_width'] = abs(tc - bc)
    df['cpr_narrow'] = df['cpr_width'] < (df['close'] * 0.002) # Arbitrary threshold for "narrow"
    
    return df

def detect_vcp(df, window=20):
    """
    Detects Volatility Contraction Pattern (simplified).
    Look for decreasing volatility (range) over the window.
    """
    recent = df.tail(window).copy()
    # Check if max range is decreasing in chunks
    ranges = recent['range_pct'].values
    # Rough check: is the trend of ranges down?
    # Or simply: recent range is very small compared to avg
    is_tight = ranges[-1] < (ranges.mean() * 0.5)
    return is_tight

def score_setup(row, full_df_tail):
    """
    Assigns a confluence score (0-5) based on patterns.
    """
    score = 0
    reasons = []
    
    # 1. Trend Filter: Price > EMA 20 > EMA 50 > EMA 200
    if (row['close'] > row['ema_20']) and (row['ema_20'] > row['ema_50']):
        score += 1
        reasons.append("Strong Uptrend")
        
    # 2. Liquidity (Avg Vol > 100k)
    if row['vol_ma_20'] > 100000:
        # Pass filter, maybe add to score if HUGE volume
        pass
    else:
        return 0, [] # Skip illiquid
        
    # 3. Momentum Burst (Range Expansion + Vol Surge)
    # Check vs yesterday
    if (row['range_pct'] > full_df_tail.iloc[-2]['range_pct'] * 2) and \
       (row['volume'] > row['vol_ma_20'] * 1.5) and \
       (row['close'] > row['open']):
        score += 2
        reasons.append("Momentum Burst (Vol+Range)")
        
    # 4. Mean Reversion (Near 20 EMA bounce)
    dist_ema = abs(row['close'] - row['ema_20']) / row['close']
    if dist_ema < 0.02 and row['close'] > row['ema_20']:
        score += 1
        reasons.append("Near 20EMA")
        
    # 5. CPR Narrow (Coiling)
    if row['cpr_narrow']:
        score += 1
        reasons.append("Narrow CPR")
        
    return score, reasons

def run_scan(year=None):
    if year is None:
        year = pd.Timestamp.now().year
    
    # Load Master Data
    df = pd.read_parquet('nse_master_adjusted_2014_onwards.parquet', filters=[('year', '>=', year-1)])
    
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    # Filter for Small/Midcap proxy (Price > 20 and Price < 5000) ?? 
    # Or just process all and let user filter. Process all.
    
    results = []
    
    # Group by Symbol
    grouped = df.sort_values('trade_date').groupby('symbol')
    
    for symbol, group in grouped:
        if len(group) < 50: continue # Skip new lists
        
        # Calculate Indicators
        g = calculate_indicators(group.copy())
        
        # Check latest date
        latest = g.iloc[-1]
        
        # Run Scoring
        score, reasons = score_setup(latest, g.tail(5))
        
        if score >= 3: # Minimum A grade
            results.append({
                'Symbol': symbol,
                'Date': latest['trade_date'],
                'Close': latest['close'],
                'Score': score,
                'Patterns': ", ".join(reasons),
                'Vol_Avg': int(latest['vol_ma_20'])
            })
            
    return pd.DataFrame(results).sort_values('Score', ascending=False)
