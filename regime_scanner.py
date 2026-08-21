"""
Multi-Coin Regime Scanner -- Engine
=====================================
Fetches the top N coins by market cap from CoinGecko, fits the
punctuated-equilibrium (Markov regime) model to each one's price history,
and produces a single ranked table: which coins are calm, which are
volatile, and -- for calm coins -- a theoretical "days until flip" countdown
based on that coin's own historical dwell-time statistics.

This module contains the reusable logic (no UI). dashboard_top100.py wraps
this in a Streamlit table. You can also run this file directly for a
plain-text/CSV report without the dashboard.
"""

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from regime_model import fit_hmm


COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
COINGECKO_CHART_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"
COINGECKO_SEARCH_URL = "https://api.coingecko.com/api/v3/search"
COINGECKO_DETAIL_URL = "https://api.coingecko.com/api/v3/coins/{id}"

# Phase 2: content enrichment
CRYPTOCOMPARE_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

MIN_OBSERVATIONS = 40  # minimum daily data points needed for a trustworthy HMM fit


# ---------------------------------------------------------------------
# CoinGecko fetch helpers (network -- run these on your own machine)
# ---------------------------------------------------------------------

def _request_with_retry(url, params, api_key=None, max_retries=4, base_wait=8):
    """
    GET with retry-on-429 (rate limit) using exponential backoff. Respects a
    Retry-After header if the server sends one. Raises on any other HTTP error,
    or if retries are exhausted.
    """
    headers = {}
    if api_key:
        # CoinGecko's free "Demo" API key header
        headers["x-cg-demo-api-key"] = api_key

    for attempt in range(max_retries + 1):
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        # Rate limited -- wait and retry
        retry_after = resp.headers.get("Retry-After")
        wait = float(retry_after) if retry_after else base_wait * (2 ** attempt)
        if attempt < max_retries:
            time.sleep(wait)
        else:
            resp.raise_for_status()  # exhausted retries -> raise the 429


def fetch_coin_details(coin_id, api_key=None):
    """
    Fetch the rich per-coin profile: description, homepage link, and current
    market stats (price, market cap, ATH, supply). Used on-demand for a
    single coin's detail view -- NOT called during the bulk scan, since that
    would be one extra API call per coin and slow the scan down a lot.
    Returns None if the fetch fails.
    """
    params = {
        "localization": "false", "tickers": "false", "community_data": "false",
        "developer_data": "false", "sparkline": "false", "market_data": "true",
    }
    try:
        resp = _request_with_retry(COINGECKO_DETAIL_URL.format(id=coin_id), params, api_key=api_key)
    except Exception:
        return None
    data = resp.json()

    md = data.get("market_data", {}) or {}
    description = (data.get("description", {}) or {}).get("en", "") or ""
    # Trim to a reasonable length -- some descriptions run thousands of words
    if len(description) > 600:
        description = description[:600].rsplit(".", 1)[0] + "."

    homepage_list = (data.get("links", {}) or {}).get("homepage", []) or []
    homepage = next((h for h in homepage_list if h), None)

    return {
        "description": description,
        "homepage": homepage,
        "current_price": (md.get("current_price") or {}).get("usd"),
        "market_cap": (md.get("market_cap") or {}).get("usd"),
        "ath": (md.get("ath") or {}).get("usd"),
        "ath_date": (md.get("ath_date") or {}).get("usd"),
        "circulating_supply": md.get("circulating_supply"),
        "total_supply": md.get("total_supply"),
        "max_supply": md.get("max_supply"),
    }


def fetch_coin_news(symbol, limit=3):
    """
    Fetch recent news headlines mentioning this coin, via CryptoCompare's free
    news API (no key required). Returns a list of dicts: title, url, source,
    published (datetime). Returns [] on any failure -- news is a nice-to-have,
    never worth crashing the page over.
    """
    try:
        params = {"lang": "EN", "categories": symbol.upper()}
        resp = requests.get(CRYPTOCOMPARE_NEWS_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    articles = data.get("Data", [])[:limit]
    out = []
    for a in articles:
        published = None
        if a.get("published_on"):
            published = datetime.fromtimestamp(a["published_on"], tz=timezone.utc)
        out.append({
            "title": a.get("title"),
            "url": a.get("url"),
            "source": (a.get("source_info") or {}).get("name") or a.get("source"),
            "published": published,
        })
    return out


def fetch_youtube_videos(query, api_key, limit=3):
    """
    Fetch the top N most relevant YouTube videos for a search query. Requires
    a free Google/YouTube Data API v3 key -- returns [] if no key is given or
    the request fails, rather than raising (never crash the page over an
    optional enrichment feature).
    """
    if not api_key:
        return []
    try:
        params = {
            "part": "snippet", "q": query, "type": "video",
            "maxResults": limit, "order": "relevance", "key": api_key,
        }
        resp = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    out = []
    for item in data.get("items", [])[:limit]:
        video_id = (item.get("id") or {}).get("videoId")
        if not video_id:
            continue
        snippet = item.get("snippet", {})
        thumb = (snippet.get("thumbnails", {}).get("medium") or snippet.get("thumbnails", {}).get("default") or {}).get("url")
        out.append({
            "title": snippet.get("title"),
            "channel": snippet.get("channelTitle"),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "thumbnail": thumb,
        })
    return out


def search_coins(query, api_key=None, limit=8):
    """
    Search CoinGecko for coins matching a free-text query (name or symbol).
    Returns a list of dicts: id, symbol, name, market_cap_rank, thumb.
    """
    if not query or not query.strip():
        return []
    params = {"query": query.strip()}
    resp = _request_with_retry(COINGECKO_SEARCH_URL, params, api_key=api_key)
    data = resp.json()
    coins = data.get("coins", [])[:limit]
    return [
        {
            "id": c.get("id"),
            "symbol": (c.get("symbol") or "").upper(),
            "name": c.get("name"),
            "market_cap_rank": c.get("market_cap_rank"),
            "thumb": c.get("thumb"),
        }
        for c in coins
    ]


def fetch_top_coins(n=100, vs_currency="usd", api_key=None):
    """Returns a list of dicts: id, symbol, name, market_cap_rank, current_price."""
    coins = []
    per_page = 100  # CoinGecko max per page
    pages_needed = (n + per_page - 1) // per_page
    for page in range(1, pages_needed + 1):
        params = {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": min(per_page, n - len(coins)),
            "page": page,
            "sparkline": "false",
        }
        resp = _request_with_retry(COINGECKO_MARKETS_URL, params, api_key=api_key)
        coins.extend(resp.json())
        if len(coins) >= n:
            break
    return coins[:n]


def fetch_price_history(coin_id, days=365, vs_currency="usd", api_key=None):
    """Returns a DataFrame with columns: date, close -- daily granularity."""
    params = {"vs_currency": vs_currency, "days": days}
    resp = _request_with_retry(COINGECKO_CHART_URL.format(id=coin_id), params, api_key=api_key)
    data = resp.json()
    prices = data.get("prices", [])
    if not prices:
        return pd.DataFrame(columns=["date", "close"])
    df = pd.DataFrame(prices, columns=["ts_ms", "close"])
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms")
    return df[["date", "close"]]


# ---------------------------------------------------------------------
# Core regime-fitting logic (pure computation, testable offline)
# ---------------------------------------------------------------------

def fit_regime_from_prices(price_df, vol_window=12):
    """
    price_df: DataFrame with columns date, close (sorted ascending by date).
    Returns a dict summary, or None if there's not enough data to fit reliably.
    """
    df = price_df.copy().dropna(subset=["close"]).reset_index(drop=True)
    df["log_ret"] = np.log(df["close"]).diff()
    df["realized_vol"] = df["log_ret"].rolling(vol_window).std()
    df = df.dropna().reset_index(drop=True)

    if len(df) < MIN_OBSERVATIONS:
        return None

    X = df[["log_ret", "realized_vol"]].values
    try:
        model, hidden_states, state_probs, P, means, covars = fit_hmm(X, n_states=2)
    except Exception:
        return None

    current_state = hidden_states[-1]
    confidence = state_probs[-1][current_state]

    # current streak length
    s = hidden_states
    run = 1
    i = len(s) - 2
    while i >= 0 and s[i] == current_state:
        run += 1
        i -= 1

    p_stay = P[current_state, current_state]
    # guard against p_stay==1 (degenerate fit) which would break log()
    p_stay = min(p_stay, 0.999999)
    median_days_from_now = np.log(0.5) / np.log(p_stay)
    mean_days_from_now = 1 / (1 - p_stay)

    return {
        "current_state": int(current_state),          # 0 = calm, 1 = volatile
        "state_label": "calm" if current_state == 0 else "volatile",
        "confidence": float(confidence),
        "streak_days": int(run),
        "median_days_to_flip": float(median_days_from_now),
        "mean_days_to_flip": float(mean_days_from_now),
        "latest_price": float(df["close"].iloc[-1]),
        "latest_date": df["date"].iloc[-1],
        "n_obs": len(df),
        "hidden_states": hidden_states,   # kept for optional detail charts
        "df": df,                          # kept for optional detail charts
    }


def build_ranking_table(coin_meta_list, price_histories, today=None):
    """
    coin_meta_list: list of dicts from fetch_top_coins (id, symbol, name, market_cap_rank, current_price)
    price_histories: dict {coin_id: price_df}
    Returns a DataFrame ranked so the coin theoretically closest to flipping is first.
    """
    if today is None:
        today = datetime.now(timezone.utc)

    rows = []
    for meta in coin_meta_list:
        cid = meta["id"]
        price_df = price_histories.get(cid)
        if price_df is None or price_df.empty:
            continue
        fit = fit_regime_from_prices(price_df)
        if fit is None:
            continue

        flip_date_median = today + timedelta(days=fit["median_days_to_flip"])

        rows.append({
            "rank": meta.get("market_cap_rank"),
            "symbol": meta.get("symbol", "").upper(),
            "name": meta.get("name"),
            "id": cid,
            "image": meta.get("image"),
            "price": fit["latest_price"],
            "change_24h_pct": meta.get("price_change_percentage_24h"),
            "market_cap": meta.get("market_cap"),
            "ath": meta.get("ath"),
            "ath_date": meta.get("ath_date"),
            "circulating_supply": meta.get("circulating_supply"),
            "total_supply": meta.get("total_supply"),
            "max_supply": meta.get("max_supply"),
            "state": fit["state_label"],
            "confidence_pct": round(fit["confidence"] * 100, 1),
            "streak_days": fit["streak_days"],
            "median_days_to_flip": round(fit["median_days_to_flip"], 1),
            "flip_date": flip_date_median.strftime("%d/%m/%Y"),
            "n_obs": fit["n_obs"],
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Sort so soonest theoretical flip is first, regardless of current state --
    # this naturally surfaces both "volatile coins about to calm down" and
    # "calm coins about to break" at the top together, ranked by urgency.
    out = out.sort_values("median_days_to_flip", ascending=True).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------
# Orchestration (network -- run on your own machine)
# ---------------------------------------------------------------------

def scan_top_coins(n=100, days=365, request_delay=1.3, progress_callback=None, api_key=None):
    """
    Full pipeline: fetch top N coins, fetch each one's history, fit regimes,
    return the ranked table. request_delay throttles calls to stay under
    CoinGecko's free-tier rate limit. Failed fetches retry automatically
    with backoff before being counted as a real failure.
    progress_callback(i, n, coin_symbol) is called after each coin, if provided.
    """
    coin_meta_list = fetch_top_coins(n=n, api_key=api_key)
    price_histories = {}

    for i, meta in enumerate(coin_meta_list):
        cid = meta["id"]
        try:
            price_histories[cid] = fetch_price_history(cid, days=days, api_key=api_key)
        except Exception as e:
            price_histories[cid] = pd.DataFrame(columns=["date", "close"])
            if progress_callback:
                progress_callback(i + 1, len(coin_meta_list), f"{meta.get('symbol','?').upper()} (FAILED: {e})")
        else:
            if progress_callback:
                progress_callback(i + 1, len(coin_meta_list), meta.get("symbol", "?").upper())
        time.sleep(request_delay)

    return build_ranking_table(coin_meta_list, price_histories)


if __name__ == "__main__":
    def _print_progress(i, n, label):
        print(f"[{i}/{n}] {label}")

    table = scan_top_coins(n=100, progress_callback=_print_progress)
    table.to_csv("regime_scan_results.csv", index=False)
    print(table.head(20).to_string())
