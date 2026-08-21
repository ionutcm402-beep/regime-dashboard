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
import time

import pandas as pd
import streamlit as st

from regime_scanner import (
    scan_top_coins, fit_regime_from_prices, fetch_price_history, search_coins, fetch_coin_details,
    fetch_coin_news, fetch_youtube_videos,
)
import plotly.graph_objects as go


st.set_page_config(page_title="Top Coins Regime Scanner", page_icon="🪙", layout="wide")


def format_compact_number(value, prefix=""):
    """Abbreviate large numbers (1.45T, 282.0B, 44.9M) so they fit in a metric box."""
    if pd.isna(value):
        return "—"
    value = float(value)
    for threshold, suffix in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(value) >= threshold:
            return f"{prefix}{value / threshold:,.2f}{suffix}"
    return f"{prefix}{value:,.0f}"


def render_content_enrichment(coin_symbol, coin_name, youtube_key):
    """
    Shared section for a coin detail view: recent news headlines, top YouTube
    videos, and a link to search X (Twitter) manually -- since live X search
    requires a paid API tier we're not using. Used by both the quick-search
    card and the full coin-detail profile page.
    """
    news_col, video_col = st.columns(2)

    with news_col:
        st.markdown("**Recent news**")
        news = fetch_coin_news(coin_symbol)
        if not news:
            st.caption("No recent headlines found (or the news source is temporarily unavailable).")
        else:
            for article in news:
                date_str = article["published"].strftime("%d/%m/%Y") if article["published"] else ""
                st.markdown(
                    f"[{article['title']}]({article['url']})  \n"
                    f"<span style='font-size:12px; color:#868e96;'>{article['source']} · {date_str}</span>",
                    unsafe_allow_html=True,
                )

    with video_col:
        st.markdown("**Top videos**")
        if not youtube_key:
            st.caption("Add a YouTube API key in the sidebar to see relevant videos here.")
        else:
            videos = fetch_youtube_videos(f"{coin_name} crypto", youtube_key)
            if not videos:
                st.caption("No videos found (or the YouTube API request failed).")
            else:
                for v in videos:
                    vcol1, vcol2 = st.columns([1, 2])
                    with vcol1:
                        if v["thumbnail"]:
                            st.image(v["thumbnail"], use_container_width=True)
                    with vcol2:
                        st.markdown(
                            f"[{v['title']}]({v['url']})  \n"
                            f"<span style='font-size:12px; color:#868e96;'>{v['channel']}</span>",
                            unsafe_allow_html=True,
                        )

    x_query = coin_name.replace(" ", "+")
    st.link_button(f"Search X for \"{coin_name}\" ↗", f"https://x.com/search?q={x_query}&src=typed_query&f=live")


st.title("🪙 Top Coins: Calm vs Volatile Regime Scanner")
st.caption(
    "Punctuated-equilibrium Markov model applied across the top coins by market cap. "
    "Sorted by theoretical time-to-flip -- most urgent at the top."
)

st.sidebar.header("🔑 API keys")
api_key = st.sidebar.text_input(
    "CoinGecko API key (optional, recommended)",
    type="password",
    help="Free at coingecko.com/en/developers/dashboard. Without a key you're "
         "sharing the anonymous public rate limit with everyone else on this "
         "cloud host's IP, so expect more failed coins. A free 'Demo' key "
         "gives you your own dedicated, much higher limit.",
)
youtube_api_key = st.sidebar.text_input(
    "YouTube API key (optional)",
    type="password",
    help="Free from console.cloud.google.com -- enable the 'YouTube Data API v3' "
         "and create an API key. Without this, the top-videos section on coin "
         "detail pages is simply skipped.",
)

# ---------------------------------------------------------------------
# Quick single-coin search (independent of the full scan below)
# ---------------------------------------------------------------------

st.markdown("### 🔍 Quick look-up: check any single coin")
search_col1, search_col2 = st.columns([3, 1])
with search_col1:
    search_query = st.text_input(
        "Search by name or symbol", placeholder="e.g. dogecoin, PEPE, chainlink...",
        label_visibility="collapsed",
    )
with search_col2:
    search_clicked = st.button("🔎 Search", use_container_width=True)

if search_clicked and search_query:
    with st.spinner(f"Searching for '{search_query}'..."):
        try:
            matches = search_coins(search_query)
        except Exception as e:
            matches = []
            st.error(f"Search failed: {e}")
    st.session_state["search_matches"] = matches
    st.session_state["search_query_used"] = search_query

if st.session_state.get("search_matches"):
    matches = st.session_state["search_matches"]
    if not matches:
        st.warning(f"No coins found matching '{st.session_state.get('search_query_used', '')}'.")
    else:
        options = [f"{m['name']} ({m['symbol']})" for m in matches]
        picked = st.selectbox("Matches found -- pick one:", options, key="search_pick")
        picked_coin = matches[options.index(picked)]

        if st.button("📊 Load regime for this coin"):
            with st.spinner(f"Fetching history for {picked_coin['name']}..."):
                try:
                    price_df = fetch_price_history(picked_coin["id"], days=365)
                    fit = fit_regime_from_prices(price_df)
                except Exception as e:
                    fit = None
                    price_df = None
                    st.error(f"Couldn't fetch data: {e}")

            if fit is None and price_df is not None:
                st.warning("Not enough price history to fit a reliable model for this coin.")
            elif fit is not None:
                mood = "Calm" if fit["state_label"] == "calm" else "Volatile"
                mood_color = "#2f9e44" if fit["state_label"] == "calm" else "#e03131"

                st.markdown(
                    f"## {picked_coin['name']} ({picked_coin['symbol']}) -- currently "
                    f"<span style='color:{mood_color}'>{mood}</span>",
                    unsafe_allow_html=True,
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Price", f"${fit['latest_price']:,.4f}")
                c2.metric("State", mood, f"{fit['confidence']*100:.1f}% confidence")
                c3.metric("Current streak", f"{fit['streak_days']} days")
                c4.metric("Median days to flip", f"{fit['median_days_to_flip']:.1f}")

                d = fit["df"]
                fig = go.Figure()
                for s_val, s_name, color in [(0, "Calm", "#4c6ef5"), (1, "Volatile", "#d6336c")]:
                    mask = fit["hidden_states"] == s_val
                    fig.add_trace(go.Scatter(
                        x=d.loc[mask, "date"], y=d.loc[mask, "close"],
                        mode="markers", name=s_name, marker=dict(size=5, color=color),
                    ))
                fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10),
                                   legend=dict(orientation="h", y=1.05))
                st.plotly_chart(fig, use_container_width=True)

                render_content_enrichment(picked_coin["symbol"], picked_coin["name"], youtube_api_key)

st.markdown("---")
st.markdown("### 📋 Full market scan")

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------

st.sidebar.header("⚙️ Scan settings")
n_coins = st.sidebar.number_input(
    "Number of top coins to scan", min_value=1, max_value=250, value=20, step=1,
    help="How many coins by market cap rank to include in the scan.",
)
days_history = st.sidebar.selectbox("History length per coin", [180, 365, 730], index=1)
request_delay = st.sidebar.slider("Delay between API calls (sec)", 0.5, 3.0, 1.3, step=0.1,
                                   help="Higher = slower scan but safer against CoinGecko rate limits.")

st.sidebar.markdown("**Coin categories to include**")
st.sidebar.caption(
    "Best-effort classification by coin, not a live category feed. "
    "Coins that don't match any group below (\"Other\") are always included."
)

CATEGORY_COINS = {
    "Stablecoins": {
        "tether", "usd-coin", "usds", "dai", "binance-usd", "true-usd", "frax",
        "usdd", "gemini-dollar", "paypal-usd", "first-digital-usd", "usde",
        "ethena-usde", "usdb", "fdusd", "pyusd", "tusd", "usdp", "susds",
    },
    "Meme coins": {
        "dogecoin", "shiba-inu", "pepe", "floki", "bonk", "dogwifcoin",
        "brett-based", "mog-coin", "book-of-meme", "turbo", "memecoin",
    },
    "Layer 1s": {
        "bitcoin", "ethereum", "solana", "cardano", "avalanche-2", "polkadot",
        "near", "aptos", "sui", "cosmos", "algorand", "internet-computer",
        "tron", "binancecoin", "litecoin",
    },
    "DeFi tokens": {
        "uniswap", "aave", "compound-governance-token", "curve-dao-token",
        "maker", "lido-dao", "pancakeswap-token", "sushi", "1inch",
        "havven", "yearn-finance",
    },
    "Exchange tokens": {
        "binancecoin", "okb", "leo-token", "kucoin-shares", "gate-token",
        "huobi-token", "bitget-token",
    },
}

category_include = {}
for cat_name in CATEGORY_COINS:
    default_checked = cat_name != "Stablecoins"  # matches old default: stablecoins excluded, rest included
    category_include[cat_name] = st.sidebar.checkbox(cat_name, value=default_checked, key=f"cat_{cat_name}")

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
    timer_text = st.empty()

    scan_start = time.time()

    def _update_progress(i, n, label):
        elapsed = time.time() - scan_start
        progress_bar.progress(i / n, text=f"Fetching {label} ({i}/{n})...")
        status_text.text(f"Last processed: {label}")
        timer_text.markdown(f"⏱️ **Elapsed: {elapsed:.1f} sec**")

    with st.spinner("Scanning..."):
        try:
            table = scan_top_coins(
                n=n_coins,
                days=days_history,
                request_delay=request_delay,
                progress_callback=_update_progress,
                api_key=api_key if api_key else None,
            )
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            timer_text.empty()
            st.error(
                f"Scan failed: {e}\n\n"
                "This is usually a temporary CoinGecko API issue (rate limit, "
                "downtime, or a network hiccup). Try again in a minute, or "
                "reduce the number of coins / increase the delay between calls."
            )
            st.stop()

    scan_duration = time.time() - scan_start
    st.session_state["scan_table"] = table
    st.session_state["category_include"] = category_include
    st.session_state["scan_time"] = datetime.now(timezone.utc)
    st.session_state["scan_duration_sec"] = scan_duration
    progress_bar.empty()
    status_text.empty()
    timer_text.empty()
    st.toast(f"✅ Scan finished in {scan_duration:.1f} seconds ({len(table)} coins)", icon="⏱️")


# ---------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------

if "scan_table" not in st.session_state:
    st.info("Set your scan options in the sidebar and click **Run scan** to begin.")
    st.stop()

table = st.session_state["scan_table"]

def _coin_category(coin_id):
    for cat_name, ids in CATEGORY_COINS.items():
        if coin_id in ids:
            return cat_name
    return "Other"

saved_category_include = st.session_state.get("category_include", {c: (c != "Stablecoins") for c in CATEGORY_COINS})
table = table.copy()
table["category"] = table["id"].map(_coin_category)
# A coin is kept if it's "Other" (uncategorized -- always shown) or if its
# category checkbox is checked.
keep_mask = table["category"].map(lambda c: c == "Other" or saved_category_include.get(c, True))
table = table[keep_mask].reset_index(drop=True)

scan_time = st.session_state["scan_time"]

if table.empty:
    st.warning("No coins returned enough data to fit a reliable model. Try increasing history length.")
    st.stop()

st.caption(
    f"Last scanned: {scan_time.strftime('%Y-%m-%d %H:%M:%S UTC')} -- {len(table)} coins with valid fits "
    f"-- took {st.session_state.get('scan_duration_sec', 0):.1f} sec"
)

# Live countdown: recompute "days remaining" relative to right now, not scan time
now = datetime.now(timezone.utc)
elapsed_since_scan = (now - scan_time).total_seconds() / 86400  # days
display_table = table.copy()
display_table["days_remaining"] = (display_table["median_days_to_flip"] - elapsed_since_scan).round(1)
display_table["days_remaining"] = display_table["days_remaining"].clip(lower=0)
display_table = display_table.sort_values("days_remaining", ascending=True).reset_index(drop=True)

# Clean, modern labels -- no inline emoji in dense data cells (background
# tint on the row already signals calm/volatile clearly enough).
if "change_24h_pct" not in display_table.columns:
    display_table["change_24h_pct"] = pd.NA  # defensive: handles stale cached scans safely
display_table["state"] = display_table["state"].str.upper()
display_table["change_24h_pct_display"] = display_table["change_24h_pct"].map(
    lambda v: f"+{v:.2f}%" if pd.notna(v) and v >= 0 else (f"{v:.2f}%" if pd.notna(v) else "—")
)

view_mode = st.radio("View", ["Table", "Cards"], horizontal=True, label_visibility="collapsed")


display_cols = ["rank", "symbol", "name", "category", "price", "change_24h_pct_display", "state", "confidence_pct",
                 "streak_days", "days_remaining", "flip_date"]
col_labels = {
    "rank": "Mkt Cap Rank", "symbol": "Symbol", "name": "Name", "category": "Category", "price": "Price (USD)",
    "change_24h_pct_display": "24h Change",
    "state": "State", "confidence_pct": "Confidence %", "streak_days": "Current Streak (days)",
    "days_remaining": "Days Until Theoretical Flip", "flip_date": "Flip Date (median)",
}

def highlight_state(row):
    # Subtle row tint: light green for calm, light red for volatile.
    color = "#e6f7ec" if row["State"] == "CALM" else "#fdecec"
    return [f"background-color: {color}"] * len(row)

def color_change(val):
    # Colored text (not emoji) for the 24h change column -- green/red like a
    # normal fintech dashboard.
    if val == "—":
        return "color: #868e96"
    return "color: #2f9e44; font-weight: 600" if not val.startswith("-") else "color: #e03131; font-weight: 600"

styled = (
    display_table[display_cols]
    .rename(columns=col_labels)
    .style.apply(highlight_state, axis=1)
    .map(color_change, subset=["24h Change"])
    .format({"Price (USD)": "${:,.4f}",
              "Confidence %": "{:.1f}%", "Days Until Theoretical Flip": "{:.1f}"})
    .set_properties(**{"text-align": "center"})
    .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
)

if view_mode == "Table":
    st.dataframe(styled, use_container_width=True, height=min(45 * (len(display_table) + 1), 900))
else:
    st.markdown(
        """
        <style>
        .coin-card {
            background: white; border: 1px solid #e9ecef; border-radius: 12px;
            padding: 14px; transition: box-shadow 0.15s ease, transform 0.15s ease;
        }
        .coin-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.08); transform: translateY(-2px); }
        </style>
        """,
        unsafe_allow_html=True,
    )
    n_cols = 4
    rows_of_coins = [display_table.iloc[i:i + n_cols] for i in range(0, len(display_table), n_cols)]
    for chunk in rows_of_coins:
        cols = st.columns(n_cols)
        for col, (_, coin) in zip(cols, chunk.iterrows()):
            with col:
                badge_bg = "#e6f7ec" if coin["state"] == "CALM" else "#fdecec"
                badge_color = "#2f9e44" if coin["state"] == "CALM" else "#e03131"
                change_color = "#868e96" if coin["change_24h_pct_display"] == "—" else (
                    "#2f9e44" if not coin["change_24h_pct_display"].startswith("-") else "#e03131"
                )
                logo_html = f'<img src="{coin["image"]}" width="28" style="border-radius:50%;">' if pd.notna(coin.get("image")) else ""
                st.markdown(
                    f"""
                    <div class="coin-card">
                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                            {logo_html}
                            <div>
                                <div style="font-weight:600; font-size:14px;">{coin['name']}</div>
                                <div style="font-size:11px; color:#868e96;">{coin['category']}</div>
                            </div>
                        </div>
                        <div style="font-size:17px; font-weight:600; margin-bottom:2px;">${coin['price']:,.4f}</div>
                        <div style="font-size:12px; color:{change_color}; margin-bottom:10px;">{coin['change_24h_pct_display']} 24h</div>
                        <div style="background:{badge_bg}; color:{badge_color}; font-size:12px; font-weight:600; padding:5px 0; border-radius:6px; text-align:center;">
                            {coin['state']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("View details", key=f"card_view_{coin['id']}", use_container_width=True):
                    st.session_state["detail_coin_id"] = coin["id"]
                    st.session_state["detail_coin_label"] = f"{coin['symbol']} - {coin['name']}"
                    st.rerun()

st.markdown(
    "**Green rows** = currently calm (stasis). **Red rows** = currently volatile (punctuation). "
    "The list is sorted so whichever coin is theoretically soonest to flip regimes -- in either "
    "direction -- appears at the top. This is a probabilistic midpoint (50/50 odds), not a prediction."
)

st.markdown("---")

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Coin detail view -- rich profile page, styled after a single AutoTrader listing
# ---------------------------------------------------------------------

st.subheader("Coin detail")

label_options = (display_table["symbol"] + " - " + display_table["name"]).tolist()
id_by_label = dict(zip(label_options, display_table["id"]))

default_index = 0
if st.session_state.get("detail_coin_label") in label_options:
    default_index = label_options.index(st.session_state["detail_coin_label"])

selected_label = st.selectbox("Select a coin", label_options, index=default_index)
selected_id = id_by_label[selected_label]

if st.button("Load coin profile"):
    with st.spinner(f"Fetching profile for {selected_label}..."):
        try:
            price_df = fetch_price_history(selected_id, days=days_history)
            fit = fit_regime_from_prices(price_df)
        except Exception as e:
            st.error(f"Couldn't fetch price history for this coin: {e}")
            st.stop()
        details = fetch_coin_details(selected_id, api_key=api_key if api_key else None)

    if fit is None:
        st.warning("Not enough price data to fit a reliable model for this coin.")
    else:
        coin_row = display_table.loc[display_table["id"] == selected_id].iloc[0]
        mood = "Calm" if fit["state_label"] == "calm" else "Volatile"
        mood_bg = "#e6f7ec" if fit["state_label"] == "calm" else "#fdecec"
        mood_color = "#2f9e44" if fit["state_label"] == "calm" else "#e03131"

        header_col1, header_col2 = st.columns([3, 1])
        with header_col1:
            logo_html = f'<img src="{coin_row["image"]}" width="40" style="border-radius:50%; vertical-align:middle; margin-right:10px;">' if pd.notna(coin_row.get("image")) else ""
            st.markdown(
                f"<div style='display:flex; align-items:center;'>{logo_html}"
                f"<div><div style='font-size:20px; font-weight:600;'>{coin_row['name']} ({coin_row['symbol']})</div>"
                f"<div style='font-size:13px; color:#868e96;'>{coin_row['category']}  ·  rank #{coin_row['rank']}</div></div></div>",
                unsafe_allow_html=True,
            )
        with header_col2:
            st.markdown(
                f"<div style='background:{mood_bg}; color:{mood_color}; font-weight:600; text-align:center; "
                f"padding:10px; border-radius:8px;'>{mood.upper()}<br>"
                f"<span style='font-size:12px; font-weight:400;'>{fit['streak_days']} day streak</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<div style='font-size:26px; font-weight:600; margin-top:14px;'>${fit['latest_price']:,.4f}</div>",
            unsafe_allow_html=True,
        )

        d = fit["df"]
        fig = go.Figure()
        for s_val, s_name, color in [(0, "Calm", "#4c6ef5"), (1, "Volatile", "#d6336c")]:
            mask = fit["hidden_states"] == s_val
            fig.add_trace(go.Scatter(
                x=d.loc[mask, "date"], y=d.loc[mask, "close"],
                mode="markers", name=s_name, marker=dict(size=5, color=color),
            ))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)

        # Key specs strip -- prefer fresh details from fetch_coin_details, fall back to scan-time data
        specs_source = details if details else {}
        market_cap = specs_source.get("market_cap") or coin_row.get("market_cap")
        ath = specs_source.get("ath") or coin_row.get("ath")
        circ_supply = specs_source.get("circulating_supply") or coin_row.get("circulating_supply")
        max_supply = specs_source.get("max_supply") or coin_row.get("max_supply")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Market cap", format_compact_number(market_cap, prefix="$"))
        s2.metric("Circulating supply", format_compact_number(circ_supply))
        s3.metric("All-time high", f"${ath:,.4f}" if pd.notna(ath) else "—")
        s4.metric("Max supply", format_compact_number(max_supply) if pd.notna(max_supply) else "No cap")

        link_col1, link_col2 = st.columns(2)
        with link_col1:
            st.link_button("View on CoinGecko ↗", f"https://www.coingecko.com/en/coins/{selected_id}", use_container_width=True)
        with link_col2:
            if details and details.get("homepage"):
                st.link_button("Official website ↗", details["homepage"], use_container_width=True)
            else:
                st.caption("No official website listed.")

        if details and details.get("description"):
            with st.expander("About this coin"):
                st.write(details["description"])
        elif details is None:
            st.caption("Couldn't fetch extended profile data (description, website) -- showing chart and scan-time stats only.")

        st.markdown("---")
        render_content_enrichment(coin_row["symbol"], coin_row["name"], youtube_api_key)
