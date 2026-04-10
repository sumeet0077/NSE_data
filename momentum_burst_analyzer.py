"""
Momentum Burst Probability Analyzer
=====================================
Takes the union of LINEAR_PULLBACK + EMA_CROSSOVER candidates and scores
them on multiple empirically-backed factors to estimate the probability
of a momentum burst in the next 1-5 trading days.

Empirical Basis:
  - Minervini (Trade Like a Stock Market Wizard): Tight price + volume dry-up → explosion
  - O'Neil (CANSLIM): Relative Strength + Volume confirmation
  - Bollinger (Squeeze): Low bandwidth → expansion is inevitable
  - Institutional Footprint: High delivery % = smart money accumulation
"""

import duckdb
import pandas as pd
import numpy as np
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

PARQUET = "nse_master_adjusted_2014_onwards.parquet"
READ_CMD = f"read_parquet('{PARQUET}', union_by_name=True)"

# ETF exclusion patterns
ETF_PATTERNS = ['LIQUID', 'GOLD', 'SILVER', 'NIFTY', 'BEES', 'ETF', 'SETF', 'BIRET']


def get_candidate_universe():
    """
    Union of LINEAR_PULLBACK + EMA_CROSSOVER scan results.
    Returns a list of unique symbols.
    """
    from scanner_engine_v2 import DuckDBScanner
    scanner = DuckDBScanner(PARQUET)

    lp = scanner.run_scan(strategy='LINEAR_PULLBACK')
    ema = scanner.run_scan(strategy='EMA_CROSSOVER')

    symbols = set()
    if not lp.empty:
        symbols.update(lp['symbol'].tolist())
    if not ema.empty:
        symbols.update(ema['symbol'].tolist())

    logging.info(f"Candidate universe: {len(symbols)} stocks (LP={len(lp)}, EMA={len(ema)})")
    return sorted(symbols), lp, ema


def compute_all_metrics(symbols):
    """
    For each candidate symbol, compute a rich set of metrics from raw price data.
    Uses 250 trading days of history for robust calculations.
    """
    con = duckdb.connect(database=':memory:')

    sym_list = "', '".join(symbols)
    query = f"""
        SELECT symbol, trade_date, open, high, low, close, volume, deliv_pct
        FROM {READ_CMD}
        WHERE symbol IN ('{sym_list}')
        AND trade_date >= (SELECT MAX(trade_date) - INTERVAL '400 days' FROM {READ_CMD})
        ORDER BY symbol, trade_date
    """
    df = con.query(query).df()

    # Also get NIFTY 50 for relative strength calculation
    nifty_query = f"""
        SELECT trade_date, close as nifty_close
        FROM {READ_CMD}
        WHERE symbol = 'NIFTY 50'
        AND trade_date >= (SELECT MAX(trade_date) - INTERVAL '400 days' FROM {READ_CMD})
        ORDER BY trade_date
    """
    try:
        nifty = con.query(nifty_query).df()
    except:
        nifty = pd.DataFrame()

    results = []
    for symbol, g in df.groupby('symbol'):
        g = g.sort_values('trade_date').reset_index(drop=True)
        if len(g) < 60:
            continue

        close = g['close']
        volume = g['volume']
        high = g['high']
        low = g['low']
        today = g.iloc[-1]

        # === CORE INDICATORS ===

        # 1. EMA 20 (true recursive)
        ema_20 = close.ewm(span=20, adjust=False).mean()

        # 2. SMAs
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()

        # 3. Volume metrics
        vol_sma_20 = volume.rolling(20).mean()
        rvol = (today['volume'] / vol_sma_20.iloc[-1]) if vol_sma_20.iloc[-1] > 0 else 0

        # === DERIVED METRICS (Empirically Backed) ===

        # 4. R² Linearity (50-day) — How straight-line is the trend?
        if len(g) >= 50:
            recent = g.tail(50)
            x = np.arange(50)
            y = recent['close'].values
            corr = np.corrcoef(x, y)[0, 1]
            r_squared = corr ** 2
            slope = np.polyfit(x, y, 1)[0]
        else:
            r_squared = 0
            slope = 0

        # 5. Volume Contraction Ratio — Is volume drying up before the move?
        #    Compare last 5-day avg volume to 20-day avg volume
        #    Ratio < 1 = volume drying up (bullish for pending breakout)
        vol_5 = volume.tail(5).mean()
        vol_20 = vol_sma_20.iloc[-1]
        vol_contraction = vol_5 / vol_20 if vol_20 > 0 else 1

        # 6. Bollinger Band Width (Squeeze Index)
        #    Narrower bands = more compression = bigger expected move
        bb_std = close.rolling(20).std()
        bb_upper = sma_20 + 2 * bb_std
        bb_lower = sma_20 - 2 * bb_std
        bb_width = ((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / sma_20.iloc[-1]) * 100 if sma_20.iloc[-1] > 0 else 0

        # 7. Rate of Change (ROC) — Price momentum over multiple timeframes
        roc_5 = ((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]) * 100 if len(g) > 6 else 0
        roc_20 = ((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]) * 100 if len(g) > 21 else 0
        roc_50 = ((close.iloc[-1] - close.iloc[-51]) / close.iloc[-51]) * 100 if len(g) > 51 else 0

        # 8. Distance from 52W High (Overhead Supply)
        high_252 = high.tail(252).max()
        dist_52w_high = ((close.iloc[-1] - high_252) / high_252) * 100

        # 9. Relative Strength vs Nifty (50-day)
        #    Stock ROC vs Market ROC → if stock outperforms = high RS
        rs_score = 0
        if not nifty.empty and len(nifty) > 51:
            nifty_roc_50 = ((nifty['nifty_close'].iloc[-1] - nifty['nifty_close'].iloc[-51]) / nifty['nifty_close'].iloc[-51]) * 100
            rs_score = roc_50 - nifty_roc_50  # Alpha over market

        # 10. Delivery % (Institutional Footprint)
        deliv_today = today['deliv_pct'] if pd.notna(today['deliv_pct']) else 0
        avg_deliv_5 = g['deliv_pct'].tail(5).mean() if pd.notna(g['deliv_pct'].tail(5).mean()) else 0

        # 11. Price Acceleration (2nd derivative of trend)
        #     Is the slope accelerating or decelerating?
        #     Compare slope of last 20 days vs slope of previous 20 days
        if len(g) >= 40:
            slope_recent = np.polyfit(np.arange(20), g['close'].iloc[-20:].values, 1)[0]
            slope_prior = np.polyfit(np.arange(20), g['close'].iloc[-40:-20].values, 1)[0]
            acceleration = slope_recent - slope_prior
        else:
            acceleration = 0

        # 12. ATR% (Volatility normalized by price)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean().iloc[-1]
        atr_pct = (atr_14 / close.iloc[-1]) * 100

        # 13. Candle Body Ratio (last candle strength)
        body = abs(today['close'] - today['open'])
        wick = today['high'] - today['low']
        body_ratio = (body / wick) * 100 if wick > 0 else 0

        # 14. EMA Crossover Signal (did it just cross?)
        ema_cross = 1 if (len(g) >= 2 and 
                         g['close'].iloc[-2] < ema_20.iloc[-2] and 
                         today['close'] > ema_20.iloc[-1]) else 0

        # 15. Linear pullback signal
        lp_signal = 1 if (r_squared > 0.65 and slope > 0 and 
                         today['close'] > sma_50.iloc[-1] if pd.notna(sma_50.iloc[-1]) else False) else 0

        results.append({
            'symbol': symbol,
            'close': round(today['close'], 2),
            'ema_20': round(ema_20.iloc[-1], 2),
            'sma_50': round(sma_50.iloc[-1], 2) if pd.notna(sma_50.iloc[-1]) else 0,
            # Core factors
            'r_squared': round(r_squared, 3),
            'trend_slope': round(slope, 2),
            'rvol': round(rvol, 2),
            'deliv_pct': round(deliv_today, 1),
            'avg_deliv_5d': round(avg_deliv_5, 1),
            # Derived factors
            'vol_contraction': round(vol_contraction, 2),
            'bb_width': round(bb_width, 2),
            'roc_5d': round(roc_5, 2),
            'roc_20d': round(roc_20, 2),
            'roc_50d': round(roc_50, 2),
            'dist_52w_high': round(dist_52w_high, 2),
            'rs_vs_nifty': round(rs_score, 2),
            'acceleration': round(acceleration, 3),
            'atr_pct': round(atr_pct, 2),
            'body_ratio': round(body_ratio, 1),
            'ema_cross': ema_cross,
            'lp_signal': lp_signal,
        })

    return pd.DataFrame(results)


def score_momentum_burst(metrics_df):
    """
    Composite scoring model for momentum burst probability.
    
    Each factor is scored 0-10 based on empirical ranges, then weighted
    by how predictive it is of short-term momentum bursts.

    Weights based on empirical evidence:
      - Volume Contraction (spring effect): HIGH weight
      - R² linearity: HIGH weight (predictable trends tend to resume)
      - Relative Strength: HIGH weight (leaders lead)
      - Bollinger Squeeze: MEDIUM-HIGH (tight → explosive)
      - RVOL on signal day: MEDIUM (confirmation)
      - Delivery %: MEDIUM (institutional backing)
      - Distance from 52W high: MEDIUM (no overhead supply)
      - EMA/LP dual signal: BONUS
    """

    df = metrics_df.copy()

    # --- Score each factor 0-10 ---

    # 1. R² Linearity (higher = better, >0.65 is baseline, >0.85 is excellent)
    df['s_linearity'] = np.clip((df['r_squared'] - 0.5) / 0.5 * 10, 0, 10)

    # 2. Volume Contraction (LOWER = better — coiled spring)
    #    <0.6 = excellent (volume dried up), >1.2 = no contraction
    df['s_vol_contraction'] = np.clip((1.3 - df['vol_contraction']) / 0.7 * 10, 0, 10)

    # 3. Bollinger Band Width (LOWER = tighter = better)
    #    <5% = very tight, >15% = loose
    df['s_bb_squeeze'] = np.clip((15 - df['bb_width']) / 10 * 10, 0, 10)

    # 4. Relative Strength vs Nifty (higher = better)
    #    >20 = strong outperformer, <0 = underperformer
    df['s_rs'] = np.clip(df['rs_vs_nifty'] / 4, 0, 10)

    # 5. RVOL on signal day (1.5x = good, 3x+ = excellent)
    df['s_rvol'] = np.clip(df['rvol'] / 0.5, 0, 10)

    # 6. Delivery % (>50% = strong, >70% = excellent)
    df['s_delivery'] = np.clip(df['deliv_pct'] / 10, 0, 10)

    # 7. Distance from 52W High (closer = less supply above)
    #    0% = at high, -10% = moderate, -25% = far
    df['s_proximity'] = np.clip((25 + df['dist_52w_high']) / 2.5, 0, 10)

    # 8. Positive ROC 50d (uptrend momentum)
    df['s_roc50'] = np.clip(df['roc_50d'] / 5, 0, 10)

    # 9. Acceleration bonus (2nd derivative > 0 = accelerating trend)
    df['s_accel'] = np.clip(df['acceleration'] * 5, 0, 10)

    # 10. Dual Signal Bonus (both scanners flagged it = confluence)
    df['s_dual'] = (df['ema_cross'] + df['lp_signal']) * 5  # 0, 5, or 10

    # --- Weighted Composite Score ---
    weights = {
        's_vol_contraction': 0.18,  # Spring effect — most predictive
        's_linearity':       0.15,  # Predictable trend likely to resume
        's_rs':              0.14,  # Leaders lead — relative strength
        's_bb_squeeze':      0.12,  # Tight bands → expansion
        's_rvol':            0.10,  # Volume confirmation
        's_delivery':        0.08,  # Institutional backing
        's_proximity':       0.08,  # Close to highs — no supply
        's_roc50':           0.05,  # Underlying momentum
        's_accel':           0.05,  # Trend accelerating
        's_dual':            0.05,  # Confluence bonus
    }

    df['momentum_score'] = sum(df[col] * w for col, w in weights.items())
    df['momentum_score'] = (df['momentum_score'] * 10).round(1)  # Scale to 0-100

    return df.sort_values('momentum_score', ascending=False)


def main():
    print("=" * 70)
    print("  MOMENTUM BURST PROBABILITY ANALYZER")
    print("  Empirical Multi-Factor Scoring Model")
    print("=" * 70)
    print()

    # Step 1: Get candidates
    symbols, lp_df, ema_df = get_candidate_universe()
    if not symbols:
        print("⚠️ No candidates found from either scanner.")
        return

    # Step 2: Compute all metrics
    logging.info("Computing 15 technical metrics for each candidate...")
    metrics = compute_all_metrics(symbols)
    if metrics.empty:
        print("⚠️ Could not compute metrics.")
        return

    # Step 3: Score and rank
    scored = score_momentum_burst(metrics)

    # Step 4: Display results
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    # --- Summary Table ---
    print("\n" + "=" * 70)
    print("  📊 MOMENTUM BURST RANKINGS (Highest Probability First)")
    print("=" * 70 + "\n")

    display_cols = ['symbol', 'close', 'momentum_score', 'r_squared', 'rvol',
                    'vol_contraction', 'bb_width', 'rs_vs_nifty', 'deliv_pct',
                    'dist_52w_high', 'ema_cross', 'lp_signal']

    display_df = scored[display_cols].copy()
    display_df.columns = ['Symbol', 'Close', 'Score', 'R²', 'RVOL',
                          'VolContr', 'BBW%', 'RS', 'Deliv%',
                          'Dist52W%', 'EMA✓', 'LP✓']

    if tabulate:
        print(tabulate(display_df, headers='keys', tablefmt='psql',
                       floatfmt=".2f", showindex=False))
    else:
        print(display_df.to_string(index=False))

    # --- Top Picks Deep Dive ---
    top_n = min(3, len(scored))
    print(f"\n{'=' * 70}")
    print(f"  🎯 TOP {top_n} PICKS — DEEP ANALYSIS")
    print(f"{'=' * 70}")

    for i, (_, row) in enumerate(scored.head(top_n).iterrows()):
        signals = []
        if row['ema_cross']:
            signals.append("EMA20 Crossover")
        if row['lp_signal']:
            signals.append("Linear Pullback")

        print(f"\n  #{i+1}  {row['symbol']}  —  Score: {row['momentum_score']}/100")
        print(f"  {'─' * 50}")
        print(f"  Close: ₹{row['close']:.2f}  |  EMA20: ₹{row['ema_20']:.2f}  |  SMA50: ₹{row['sma_50']:.2f}")
        print(f"  Signals: {', '.join(signals) if signals else 'None'}")
        print(f"")
        print(f"  Trend Quality:")
        print(f"    R² Linearity:    {row['r_squared']:.3f}  {'🟢' if row['r_squared'] > 0.7 else '🟡' if row['r_squared'] > 0.5 else '🔴'}")
        print(f"    Trend Slope:     {row['trend_slope']:.2f}/day")
        print(f"    Acceleration:    {row['acceleration']:.3f}  {'↗️ Accelerating' if row['acceleration'] > 0 else '↘️ Decelerating'}")
        print(f"")
        print(f"  Volume Profile:")
        print(f"    RVOL Today:      {row['rvol']:.2f}x  {'🟢' if row['rvol'] > 1.5 else '🟡'}")
        print(f"    Vol Contraction: {row['vol_contraction']:.2f}  {'🟢 Coiled' if row['vol_contraction'] < 0.8 else '🟡 Normal' if row['vol_contraction'] < 1.2 else '🔴 High'}")
        print(f"    Delivery:        {row['deliv_pct']:.1f}%%  {'🟢' if row['deliv_pct'] > 50 else '🟡'}")
        print(f"")
        print(f"  Momentum Factors:")
        print(f"    BB Width:        {row['bb_width']:.2f}%%  {'🟢 Tight' if row['bb_width'] < 8 else '🟡 Moderate' if row['bb_width'] < 15 else '🔴 Wide'}")
        print(f"    ROC 5d/20d/50d:  {row['roc_5d']:.1f}%% / {row['roc_20d']:.1f}%% / {row['roc_50d']:.1f}%%")
        print(f"    RS vs Nifty:     {row['rs_vs_nifty']:+.2f}  {'🟢 Outperforming' if row['rs_vs_nifty'] > 5 else '🟡'}")
        print(f"    Dist 52W High:   {row['dist_52w_high']:.1f}%%  {'🟢 Near Highs' if row['dist_52w_high'] > -10 else '🟡'}")
        print(f"    ATR%%:            {row['atr_pct']:.2f}%%  (daily volatility)")
        print(f"    Candle Body:     {row['body_ratio']:.0f}%%  {'🟢 Strong' if row['body_ratio'] > 60 else '🟡 Mixed'}")

    print(f"\n{'=' * 70}")
    print(f"  ⚠️  This is a quantitative screen, NOT financial advice.")
    print(f"  Always verify with price action, news, and risk management.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
