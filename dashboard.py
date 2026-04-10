import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import datetime
# Import our new engine
try:
    from scanner_engine_v2 import DuckDBScanner
except ImportError:
    # Use dummy if not found (for local testing without engine)
    scanner_engine = None

# --- Configuration ---
st.set_page_config(page_title="Pro Momentum Scanner (OCI)", layout="wide", initial_sidebar_state="expanded")
DATA_PATH = "nse_master_adjusted_2014_onwards.parquet"

# --- Data Loading ---
@st.cache_data(ttl=3600)
def load_data(year=None):
    try:
        if year:
            df = pd.read_parquet(DATA_PATH, filters=[('year', '=', year)])
        else:
            current_year = datetime.date.today().year
            df = pd.read_parquet(DATA_PATH, filters=[('year', '>=', current_year - 1)])
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df
    except Exception as e:
        return pd.DataFrame()

# --- Main App ---
st.title("🚀 Pro Momentum Scanner (Autonomous Agent)")

# Tabs for Modes
tab1, tab2 = st.tabs(["🔍 Live Scanner", "📊 Chart Viewer"])

with tab1:
    st.header("Daily Momentum Burst Scan")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.info("Scans for: VCP, EMA bounces, CPR contraction, Range Expansion.")
        
        # Manual Filters
        st.markdown("### ⚙️ Scan Parameters")
        min_price = st.number_input("Min Price (₹)", value=30, step=10)
        min_volume = st.number_input("Min Volume", value=100000, step=50000)
        rvol_thresh = st.slider("Min Relative Volume (x)", 1.0, 10.0, 1.5, 0.1)
        deliv_thresh = st.slider("Min Delivery %", 0, 100, 25, 5)
        
        scan_year = st.number_input("Scan Data Year", 2024, 2030, 2026)
        
        if st.button("RUN SCAN 🚀", type="primary"):
            with st.spinner("Running DuckDB High-Performance Scan..."):
                try:
                    # Initialize and run scanner
                    scanner = DuckDBScanner(DATA_PATH)
                    res = scanner.run_scan(
                        min_price=min_price, 
                        min_volume=min_volume, 
                        rvol_threshold=rvol_thresh, 
                        delivery_threshold=deliv_thresh
                    )
                    
                    if not res.empty:
                        st.success(f"Scan Complete! Found {len(res)} A+ Setups")
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Top Pick", res.iloc[0]['symbol'])
                        with col2:
                            st.metric("Highest Volume Surge", f"{res['rvol'].max()}x")
                        with col3:
                            st.metric("Avg Delivery %", f"{res['delivery_pct'].mean():.1f}%")

                        # Create clickable links for TradingView
                        res['chart_link'] = "https://in.tradingview.com/chart/?symbol=NSE:" + res['symbol']

                        # Formatting for display
                        st.dataframe(
                            res.style.background_gradient(subset=['rvol'], cmap='Reds')
                               .background_gradient(subset=['delivery_pct'], cmap='Greens')
                               .format({
                                   'adj_close': '{:.2f}', 
                                   'sma_20': '{:.2f}', 
                                   'sma_50': '{:.2f}', 
                                   'sma_200': '{:.2f}',
                                   'rvol': '{:.2f}x',
                                   'delivery_pct': '{:.2f}%'
                               }),
                            column_config={
                                "chart_link": st.column_config.LinkColumn(
                                    "Chart", display_text="Open in TV"
                                )
                            },
                            use_container_width=True
                        )
                        st.session_state['scan_results'] = res # Store results for chart viewer
                    else:
                        st.warning("No stocks matched the strict V2 criteria. Try lowering reliability filters?")
                except Exception as e:
                    st.error(f"Scan failed: {e}")

    with col2:
        if 'scan_results' in st.session_state:
            res = st.session_state['scan_results']
            if not res.empty:
                st.dataframe(res.style.background_gradient(subset=['rvol'], cmap='Reds'), use_container_width=True)
                
                # Setup Details
                st.subheader("Top Setup Analysis")
                top_pick = st.selectbox("Select Candidate", res['symbol'].unique()) # Changed 'Symbol' to 'symbol' based on new output
                
                # Get details from main data
                if 'data' not in st.session_state:
                    st.session_state['data'] = load_data(scan_year)
                
                df = st.session_state['data']
                setup_data = df[df['symbol'] == top_pick].sort_values('trade_date').tail(60)
                
                # Plot
                fig = go.Figure(data=[go.Candlestick(x=setup_data['trade_date'],
                                open=setup_data['adj_open'], high=setup_data['adj_high'],
                                low=setup_data['adj_low'], close=setup_data['adj_close'], name='OHLC')])
                fig.update_layout(height=400, title=f"{top_pick} - Momentum Setup (Adjusted)")
                st.plotly_chart(fig, use_container_width=True)
                
                # st.markdown(f"**Strategy Confluence:** {res[res['symbol']==top_pick].iloc[0]['Patterns']}")
            else:
                st.warning("No A+ setups found matching current strict criteria.")

with tab2:
    # Original Chart Viewer Logic
    st.header("Deep Dive Charting")
    # ... (Keep original logic here but simplified)
    if 'data' not in st.session_state:
        if st.button("Load Full History"):
            st.session_state['data'] = load_data()
    
    if 'data' in st.session_state:
        df = st.session_state['data']
        sym = st.selectbox("Symbol", df['symbol'].unique())
        s_df = df[df['symbol'] == sym].tail(100)
        st.line_chart(s_df.set_index('trade_date')['adj_close'])
