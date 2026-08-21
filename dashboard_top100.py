"""
Top-100 Regime Scanner Dashboard
===================================
Streamlit UI wrapping regime_scanner.py. Scans the top N coins on
CoinGecko, fits the calm/volatile regime model to each, and shows one
ranked table: whichever coin is theoretically closest to flipping regimes
sits at the top, with a live countdown. Calm rows are green, volatile
rows are red.

SETUP
-----
pip install streamlit plotly requests pandas numpy hmmlearn --break-system-packages

RUN
---
streamlit run dashboard_top100.py
"""

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from regime_scanner import scan_top_coins, fit_regime_from_prices, fetch_price_history
import plotly.graph_objects as go


st.set_page_config(page_title="Top Coins Regime Scanner", layout="wide")

st.title("Top Coins: Calm vs Volatile Regime Scanner")
st.caption(
    "Punctuated-equilibrium Markov model applied across the top coins by market cap. "
    "Sorted by theoretical time-to-flip -- most urgent at the top."
)

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------

st.sidebar.header("Scan settings")
n_coins = st.sidebar.slider("Number of top coins to scan", 10, 100, 100, step=10)
days_history = st.sidebar.selectbox("History length per coin", [180, 365, 730], index=1)
request_delay = st.sidebar.slider("Delay between API calls (sec)", 0.5, 3.0, 1.3, step=0.1,
                                   help="Higher = slower scan but safer against CoinGecko rate limits.")

run_scan = st.sidebar.button("Run scan", type="primary")

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Scanning {n_coins} coins takes roughly {n_coins * request_delay:.0f}-"
    f"{n_coins * (request_delay + 1):.0f} seconds due to API rate limiting. "
    "Results are cached in this session until you run a new scan."
)


# ---------------------------------------------------------------------
# Run scan (only when button pressed -- avoids re-scanning on every widget interaction)
# ---------------------------------------------------------------------

if run_scan:
    progress_bar = st.progress(0, text="Starting scan...")
    status_text = st.empty()

    def _update_progress(i, n, label):
        progress_bar.progress(i / n, text=f"Fetching {label} ({i}/{n})...")
        status_text.text(f"Last processed: {label}")

    with st.spinner("Scanning..."):
        table = scan_top_coins(
            n=n_coins,
            days=days_history,
            request_delay=request_delay,
            progress_callback=_update_progress,
        )
    st.session_state["scan_table"] = table
    st.session_state["scan_time"] = datetime.now(timezone.utc)
    progress_bar.empty()
    status_text.empty()


# ---------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------

if "scan_table" not in st.session_state:
    st.info("Set your scan options in the sidebar and click **Run scan** to begin.")
    st.stop()

table = st.session_state["scan_table"]
scan_time = st.session_state["scan_time"]

if table.empty:
    st.warning("No coins returned enough data to fit a reliable model. Try increasing history length.")
    st.stop()

st.caption(f"Last scanned: {scan_time.strftime('%Y-%m-%d %H:%M:%S UTC')} -- {len(table)} coins with valid fits")

# Live countdown: recompute "days remaining" relative to right now, not scan time
now = datetime.now(timezone.utc)
elapsed_since_scan = (now - scan_time).total_seconds() / 86400  # days
display_table = table.copy()
display_table["days_remaining"] = (display_table["median_days_to_flip"] - elapsed_since_scan).round(1)
display_table["days_remaining"] = display_table["days_remaining"].clip(lower=0)
display_table = display_table.sort_values("days_remaining", ascending=True).reset_index(drop=True)


def highlight_state(row):
    color = "#d3f9d8" if row["state"] == "calm" else "#ffe3e3"
    return [f"background-color: {color}"] * len(row)


display_cols = ["rank", "symbol", "name", "price", "state", "confidence_pct",
                 "streak_days", "days_remaining", "flip_date"]
col_labels = {
    "rank": "Mkt Cap Rank", "symbol": "Symbol", "name": "Name", "price": "Price (USD)",
    "state": "State", "confidence_pct": "Confidence %", "streak_days": "Current Streak (days)",
    "days_remaining": "Days Until Theoretical Flip", "flip_date": "Flip Date (median)",
}

styled = (
    display_table[display_cols]
    .rename(columns=col_labels)
    .style.apply(lambda r: highlight_state({"state": display_table.loc[r.name, "state"]}), axis=1)
    .format({"Price (USD)": "${:,.4f}", "Confidence %": "{:.1f}%", "Days Until Theoretical Flip": "{:.1f}"})
)

st.dataframe(styled, use_container_width=True, height=min(45 * (len(display_table) + 1), 900))

st.markdown(
    "**Green rows** = currently calm (stasis). **Red rows** = currently volatile (punctuation). "
    "The list is sorted so whichever coin is theoretically soonest to flip regimes -- in either "
    "direction -- appears at the top. This is a probabilistic midpoint (50/50 odds), not a prediction."
)

st.markdown("---")

# ---------------------------------------------------------------------
# Coin detail view
# ---------------------------------------------------------------------

st.subheader("Coin detail")
selected_symbol = st.selectbox("Select a coin for a detailed chart", display_table["symbol"] + " - " + display_table["name"])
selected_id = display_table.loc[
    (display_table["symbol"] + " - " + display_table["name"]) == selected_symbol, "id"
].iloc[0]

if st.button("Load detail chart (fetches fresh history for this coin)"):
    with st.spinner(f"Fetching history for {selected_symbol}..."):
        price_df = fetch_price_history(selected_id, days=days_history)
        fit = fit_regime_from_prices(price_df)

    if fit is None:
        st.warning("Not enough data to fit a reliable model for this coin.")
    else:
        d = fit["df"]
        fig = go.Figure()
        for s_val, s_name, color in [(0, "Calm", "#4c6ef5"), (1, "Volatile", "#d6336c")]:
            mask = fit["hidden_states"] == s_val
            fig.add_trace(go.Scatter(
                x=d.loc[mask, "date"], y=d.loc[mask, "close"],
                mode="markers", name=s_name, marker=dict(size=5, color=color),
            ))
        fig.update_layout(height=420, title=f"{selected_symbol} price, colored by regime",
                           margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Current state", fit["state_label"], f"{fit['confidence']*100:.1f}% confidence")
        c2.metric("Current streak", f"{fit['streak_days']} days")
        c3.metric("Median days to flip", f"{fit['median_days_to_flip']:.1f}")
