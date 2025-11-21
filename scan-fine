import math
import os
from datetime import datetime, timedelta, timezone
from io import BytesIO

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:  # Fallback, no timezone support
    ZoneInfo = None

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


# ------------- Utility & math helpers ------------- #

def parse_tickers(raw: str) -> List[str]:
    if not raw:
        return []
    separators = [",", ";", " "]
    for sep in separators:
        raw = raw.replace(sep, ",")
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    seen = set()
    unique = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def mid_price(row: pd.Series) -> float:
    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    last = row.get("lastPrice", np.nan)
    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
        return float((bid + ask) / 2)
    if pd.notna(last) and last > 0:
        return float(last)
    if pd.notna(bid) and bid > 0:
        return float(bid)
    if pd.notna(ask) and ask > 0:
        return float(ask)
    return np.nan


def get_nearest_expiration(t: yf.Ticker, max_dte: int) -> Tuple[Optional[str], Optional[int]]:
    try:
        expirations = t.options
    except Exception:
        return None, None
    if not expirations:
        return None, None

    today = datetime.now(timezone.utc).date()
    candidates = []
    for e in expirations:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
        except Exception:
            continue
        dte = (d - today).days
        if 0 < dte <= max_dte:
            candidates.append((dte, e))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    dte, exp = candidates[0]
    return exp, dte


def realized_volatility(history: pd.DataFrame, lookback: int = 30) -> Optional[float]:
    if history is None or history.empty:
        return None
    prices = history["Close"].dropna().tail(lookback)
    if len(prices) < 2:
        return None
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if log_returns.empty:
        return None
    daily_vol = log_returns.std()
    annual_vol = float(daily_vol * np.sqrt(252))
    return annual_vol * 100.0


def annualize_return(simple_return_pct: float, dte: int) -> float:
    if dte <= 0:
        return np.nan
    return simple_return_pct * (365.0 / float(dte))


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def bs_greeks_and_pop(
    S: float,
    K: float,
    T_years: float,
    r: float,
    sigma: float,
    option_type: str,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Black–Scholes delta, theta (per day) and POP for the SHORT leg.
    POP is computed using risk-neutral probabilities based on d2.

    Returns (delta, theta_per_day, pop_pct)
    """
    if S <= 0 or K <= 0 or T_years <= 0 or sigma <= 0:
        return None, None, None

    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T_years) / (sigma * math.sqrt(T_years))
        d2 = d1 - sigma * math.sqrt(T_years)
    except Exception:
        return None, None, None

    if option_type.lower() == "call":
        delta = norm_cdf(d1)
        theta_annual = -(S * sigma * norm_pdf(d1)) / (2.0 * math.sqrt(T_years)) - r * K * math.exp(-r * T_years) * norm_cdf(d2)
        # For a short call, POP = P(S_T < K) = N(-d2)
        pop_pct = norm_cdf(-d2) * 100.0
    else:  # put
        delta = norm_cdf(d1) - 1.0
        theta_annual = -(S * sigma * norm_pdf(d1)) / (2.0 * math.sqrt(T_years)) + r * K * math.exp(-r * T_years) * norm_cdf(-d2)
        # For a short put, POP = P(S_T > K) = N(d2)
        pop_pct = norm_cdf(d2) * 100.0

    theta_per_day = theta_annual / 365.0
    return float(delta), float(theta_per_day), float(pop_pct)


def is_us_rth() -> bool:
    """
    Approximate US equity regular trading hours:
    Mon–Fri, 09:30–16:00 US/Eastern.
    """
    if ZoneInfo is None:
        return True  # can't check; don't block

    now = datetime.now(ZoneInfo("US/Eastern"))
    if now.weekday() >= 5:  # Sat/Sun
        return False
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_time <= now <= close_time


def add_export_buttons(df: pd.DataFrame, label: str, key_prefix: str):
    """Add CSV and Excel download buttons for a DataFrame."""
    if df is None or df.empty:
        return

    csv_data = df.to_csv(index=False).encode("utf-8")

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    excel_buffer.seek(0)
    excel_data = excel_buffer.read()

    col_csv, col_xlsx = st.columns(2)
    with col_csv:
        st.download_button(
            label=f"⬇️ Download CSV – {label}",
            data=csv_data,
            file_name=f"{label.replace(' ', '_')}.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv",
        )
    with col_xlsx:
        st.download_button(
            label=f"⬇️ Download Excel – {label}",
            data=excel_data,
            file_name=f"{label.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key_prefix}_xlsx",
        )


# ------------- Strategy scanners ------------- #

def scan_put_credit_spreads(
    underlying_price: float,
    puts: pd.DataFrame,
    dte: int,
    min_option_volume: int,
    min_oi: int,
    risk_free_rate: float,
    account_size: float,
    max_risk_per_trade_pct: float,
    max_capital_per_ticker_pct: float,
    min_pop_pct: float,
) -> pd.DataFrame:
    """Bullish put credit spreads (short higher strike, long lower strike)."""
    if puts is None or puts.empty or underlying_price <= 0:
        return pd.DataFrame()

    df = puts.copy()
    df["mid"] = df.apply(mid_price, axis=1)
    df = df.dropna(subset=["mid"])

    df = df[(df["strike"] < underlying_price)]
    df = df[(df["volume"] >= min_option_volume) & (df["openInterest"] >= min_oi)]
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("strike").reset_index(drop=True)

    max_risk_per_trade = account_size * max_risk_per_trade_pct / 100.0
    max_capital_per_ticker = account_size * max_capital_per_ticker_pct / 100.0

    rows = []
    T = dte / 365.0

    for i in range(1, len(df)):
        long_row = df.iloc[i - 1]
        short_row = df.iloc[i]

        short_strike = float(short_row["strike"])
        long_strike = float(long_row["strike"])

        if not (0.8 * underlying_price <= short_strike <= 0.98 * underlying_price):
            continue

        credit = short_row["mid"] - long_row["mid"]
        width = short_strike - long_strike
        if width <= 0 or credit <= 0:
            continue

        max_loss = width - credit
        if max_loss <= 0:
            continue

        risk_per_spread = max_loss * 100.0  # 1 spread = 100 shares
        if risk_per_spread <= 0:
            continue

        sigma = short_row.get("impliedVolatility", np.nan)
        if pd.isna(sigma):
            delta, theta, pop_pct = None, None, None
        else:
            delta, theta, pop_pct = bs_greeks_and_pop(
                S=underlying_price,
                K=short_strike,
                T_years=T,
                r=risk_free_rate,
                sigma=float(sigma),
                option_type="put",
            )

        # POP filter
        if pop_pct is not None and pop_pct < min_pop_pct:
            continue

        roi_pct = (credit / max_loss) * 100.0
        ann_pct = annualize_return(roi_pct, dte)

        max_allowed_risk = min(max_risk_per_trade, max_capital_per_ticker)
        suggested_contracts = int(max_allowed_risk // risk_per_spread) if risk_per_spread > 0 else 0

        rows.append(
            {
                "strategy": "Put Credit Spread",
                "short_strike": short_strike,
                "long_strike": long_strike,
                "spread_width": width,
                "net_credit": round(credit, 2),
                "max_loss": round(max_loss, 2),
                "risk_per_spread_$": round(risk_per_spread, 2),
                "roi_pct": round(roi_pct, 1),
                "annualized_roi_pct": round(ann_pct, 1),
                "short_delta": round(delta, 3) if delta is not None else np.nan,
                "short_theta_per_day": round(theta, 3) if theta is not None else np.nan,
                "pop_pct": round(pop_pct, 1) if pop_pct is not None else np.nan,
                "short_iv_pct": round(float(sigma) * 100.0, 1) if pd.notna(sigma) else np.nan,
                "short_volume": int(short_row.get("volume", 0)),
                "short_open_interest": int(short_row.get("openInterest", 0)),
                "suggested_contracts": suggested_contracts,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("annualized_roi_pct", ascending=False).reset_index(drop=True)
    return result


def scan_call_credit_spreads(
    underlying_price: float,
    calls: pd.DataFrame,
    dte: int,
    min_option_volume: int,
    min_oi: int,
    risk_free_rate: float,
    account_size: float,
    max_risk_per_trade_pct: float,
    max_capital_per_ticker_pct: float,
    min_pop_pct: float,
) -> pd.DataFrame:
    """Bearish call credit spreads (short lower strike, long higher strike)."""
    if calls is None or calls.empty or underlying_price <= 0:
        return pd.DataFrame()

    df = calls.copy()
    df["mid"] = df.apply(mid_price, axis=1)
    df = df.dropna(subset=["mid"])

    df = df[(df["strike"] > underlying_price)]
    df = df[(df["volume"] >= min_option_volume) & (df["openInterest"] >= min_oi)]
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("strike").reset_index(drop=True)

    max_risk_per_trade = account_size * max_risk_per_trade_pct / 100.0
    max_capital_per_ticker = account_size * max_capital_per_ticker_pct / 100.0

    rows = []
    T = dte / 365.0

    for i in range(0, len(df) - 1):
        short_row = df.iloc[i]
        long_row = df.iloc[i + 1]

        short_strike = float(short_row["strike"])
        long_strike = float(long_row["strike"])

        if not (1.02 * underlying_price <= short_strike <= 1.2 * underlying_price):
            continue

        credit = short_row["mid"] - long_row["mid"]
        width = long_strike - short_strike
        if width <= 0 or credit <= 0:
            continue

        max_loss = width - credit
        if max_loss <= 0:
            continue

        risk_per_spread = max_loss * 100.0
        if risk_per_spread <= 0:
            continue

        sigma = short_row.get("impliedVolatility", np.nan)
        if pd.isna(sigma):
            delta, theta, pop_pct = None, None, None
        else:
            delta, theta, pop_pct = bs_greeks_and_pop(
                S=underlying_price,
                K=short_strike,
                T_years=T,
                r=risk_free_rate,
                sigma=float(sigma),
                option_type="call",
            )

        # POP filter
        if pop_pct is not None and pop_pct < min_pop_pct:
            continue

        roi_pct = (credit / max_loss) * 100.0
        ann_pct = annualize_return(roi_pct, dte)

        max_allowed_risk = min(max_risk_per_trade, max_capital_per_ticker)
        suggested_contracts = int(max_allowed_risk // risk_per_spread) if risk_per_spread > 0 else 0

        rows.append(
            {
                "strategy": "Call Credit Spread",
                "short_strike": short_strike,
                "long_strike": long_strike,
                "spread_width": width,
                "net_credit": round(credit, 2),
                "max_loss": round(max_loss, 2),
                "risk_per_spread_$": round(risk_per_spread, 2),
                "roi_pct": round(roi_pct, 1),
                "annualized_roi_pct": round(ann_pct, 1),
                "short_delta": round(delta, 3) if delta is not None else np.nan,
                "short_theta_per_day": round(theta, 3) if theta is not None else np.nan,
                "pop_pct": round(pop_pct, 1) if pop_pct is not None else np.nan,
                "short_iv_pct": round(float(sigma) * 100.0, 1) if pd.notna(sigma) else np.nan,
                "short_volume": int(short_row.get("volume", 0)),
                "short_open_interest": int(short_row.get("openInterest", 0)),
                "suggested_contracts": suggested_contracts,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("annualized_roi_pct", ascending=False).reset_index(drop=True)
    return result


def scan_covered_calls(
    underlying_price: float,
    calls: pd.DataFrame,
    dte: int,
    min_option_volume: int,
    min_oi: int,
    account_size: float,
    max_capital_per_ticker_pct: float,
) -> pd.DataFrame:
    """Covered call candidates: short 1 OTM call per 100 shares."""
    if calls is None or calls.empty or underlying_price <= 0:
        return pd.DataFrame()

    df = calls.copy()
    df["mid"] = df.apply(mid_price, axis=1)
    df = df.dropna(subset=["mid"])

    df = df[(df["strike"] > underlying_price)]
    df = df[(df["volume"] >= min_option_volume) & (df["openInterest"] >= min_oi)]
    if df.empty:
        return pd.DataFrame()

    df = df[(df["strike"] >= 1.02 * underlying_price) & (df["strike"] <= 1.15 * underlying_price)]
    if df.empty:
        return pd.DataFrame()

    max_capital_per_ticker = account_size * max_capital_per_ticker_pct / 100.0
    capital_per_contract = underlying_price * 100.0
    max_contracts_ticker = int(max_capital_per_ticker // capital_per_contract) if capital_per_contract > 0 else 0

    rows = []
    for _, row in df.iterrows():
        strike = float(row["strike"])
        premium = float(row["mid"])
        if premium <= 0:
            continue

        simple_yield_pct = (premium / underlying_price) * 100.0
        annualized_yield_pct = annualize_return(simple_yield_pct, dte)
        sigma = row.get("impliedVolatility", np.nan)

        rows.append(
            {
                "strategy": "Covered Call",
                "strike": strike,
                "premium": round(premium, 2),
                "yield_pct": round(simple_yield_pct, 1),
                "annualized_yield_pct": round(annualized_yield_pct, 1),
                "iv_pct": round(float(sigma) * 100.0, 1) if pd.notna(sigma) else np.nan,
                "volume": int(row.get("volume", 0)),
                "open_interest": int(row.get("openInterest", 0)),
                "capital_per_contract_$": round(capital_per_contract, 2),
                "max_contracts_given_capital": max_contracts_ticker,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("annualized_yield_pct", ascending=False).reset_index(drop=True)
    return result


# ------------- External APIs (Finnhub / Alpha Vantage) ------------- #

def get_finnhub_api_key(user_input: str) -> Optional[str]:
    key = None
    if "FINNHUB_API_KEY" in st.secrets:
        key = st.secrets["FINNHUB_API_KEY"]
    elif user_input:
        key = user_input.strip()
    return key or None


def get_alphavantage_api_key(user_input: str) -> Optional[str]:
    key = None
    if "ALPHAVANTAGE_API_KEY" in st.secrets:
        key = st.secrets["ALPHAVANTAGE_API_KEY"]
    elif user_input:
        key = user_input.strip()
    return key or None


def fetch_economic_calendar_finnhub(api_key: str, days_ahead: int = 7) -> Optional[pd.DataFrame]:
    if not api_key:
        return None
    try:
        today = datetime.utcnow().date()
        future = today + timedelta(days=days_ahead)
        url = "https://finnhub.io/api/v1/calendar/economic"
        params = {
            "from": today.isoformat(),
            "to": future.isoformat(),
            "token": api_key,
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        events = data.get("economicCalendar", [])
        if not events:
            return None
        df = pd.DataFrame(events)
        keep_cols = [c for c in df.columns if c in ("time", "country", "event", "actual", "forecast", "previous", "impact")]
        if keep_cols:
            df = df[keep_cols]
        return df
    except Exception:
        return None


def fetch_alpha_overview(symbol: str, api_key: str) -> Optional[dict]:
    if not api_key or not symbol:
        return None
    try:
        url = "https://www.alphavantage.co/query"
        params = {"function": "OVERVIEW", "symbol": symbol, "apikey": api_key}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data or "Symbol" not in data:
            return None
        return data
    except Exception:
        return None


# ------------- Streamlit App ------------- #

st.set_page_config(
    page_title="Options Income Scanner",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Options Income Scanner")
st.markdown(
    """
Scan for **Put Credit Spreads**, **Call Credit Spreads**, and **Covered Calls**  
based on free Yahoo Finance data (via `yfinance`) plus optional Finnhub / Alpha Vantage context.

> ⚠️ **Educational only — not financial advice.**  
> Always confirm prices and Greeks in your broker before trading.
"""
)

with st.sidebar:
    st.header("⚙️ Scanner Settings")

    tickers_raw = st.text_input(
        "Enter ticker symbols",
        value="AAPL, MSFT, TSLA",
        help="Separate with comma, semicolon, or space.",
    )

    strategies = st.multiselect(
        "Strategies",
        options=["Put Credit Spreads", "Call Credit Spreads", "Covered Calls"],
        default=["Put Credit Spreads", "Call Credit Spreads", "Covered Calls"],
    )

    dte_choice = st.radio(
        "Target time to expiration",
        options=["7 days (weekly)", "14 days (up to 2 weeks)"],
        index=0,
    )
    max_dte = 7 if "7" in dte_choice else 14

    only_rth = st.checkbox(
        "Only scan during regular trading hours (US)",
        value=True,
        help="Skip scans outside 09:30–16:00 US/Eastern to avoid after-hours prices.",
    )

    st.subheader("Liquidity filters")
    min_stock_volume = st.number_input(
        "Min average daily stock volume",
        min_value=0,
        value=500_000,
        step=100_000,
        help="Skip illiquid underlyings.",
    )
    min_option_volume = st.number_input(
        "Min option contract volume",
        min_value=0,
        value=50,
        step=10,
    )
    min_oi = st.number_input(
        "Min option open interest",
        min_value=0,
        value=100,
        step=10,
    )

    st.subheader("POP filter")
    min_pop_pct = st.slider(
        "Min POP (%) for credit spreads",
        min_value=50.0,
        max_value=90.0,
        value=60.0,
        step=1.0,
        help="Filter out spreads with probability of profit below this level.",
    )

    st.subheader("Risk & Position Sizing")
    account_size = st.number_input(
        "Account size ($)",
        min_value=1000.0,
        value=10_000.0,
        step=1_000.0,
    )
    max_risk_per_trade_pct = st.slider(
        "Max risk per trade (% of account)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
    )
    max_capital_per_ticker_pct = st.slider(
        "Max capital at risk per ticker (% of account)",
        min_value=5.0,
        max_value=60.0,
        value=20.0,
        step=5.0,
    )
    risk_free_rate_input = st.number_input(
        "Risk-free rate (annual, %)",
        min_value=0.0,
        max_value=15.0,
        value=4.0,
        step=0.25,
        help="Used for Black–Scholes Greeks & POP.",
    )
    risk_free_rate = risk_free_rate_input / 100.0

    st.subheader("🔑 Optional API Keys")
    finnhub_key_input = st.text_input(
        "Finnhub API key (optional)",
        type="password",
        help="For economic calendar. Prefer configuring as st.secrets['FINNHUB_API_KEY'].",
    )
    alphavantage_key_input = st.text_input(
        "Alpha Vantage API key (optional)",
        type="password",
        help="For fundamentals. Prefer configuring as st.secrets['ALPHAVANTAGE_API_KEY'].",
    )

    max_rows_per_strategy = st.slider(
        "Max results per strategy & ticker (display only)",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
    )

    run_scan = st.button("🚀 Run Scan")

tickers = parse_tickers(tickers_raw)
finnhub_api_key = get_finnhub_api_key(finnhub_key_input)
alphavantage_api_key = get_alphavantage_api_key(alphavantage_key_input)

# --------- Macro / Market Context --------- #
st.markdown("## 🌍 Macro & Market Context")
macro_col1, macro_col2, macro_col3 = st.columns([1.2, 1.2, 1.6])

with macro_col1:
    st.markdown("**VIX (Fear Index)**")
    try:
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="3mo")
        if not vix_hist.empty:
            vix_last = float(vix_hist["Close"].iloc[-1])
            vix_5d = float(vix_hist["Close"].iloc[-1] - vix_hist["Close"].iloc[-5]) if len(vix_hist) >= 5 else 0.0
            st.metric("VIX", f"{vix_last:.2f}", f"{vix_5d:+.2f} vs 5d ago")
            st.line_chart(vix_hist["Close"].tail(30), height=120)
        else:
            st.info("No VIX data.")
    except Exception:
        st.info("Could not load VIX.")

with macro_col2:
    st.markdown("**SPY (Market Proxy)**")
    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="3mo")
        if not spy_hist.empty:
            spy_last = float(spy_hist["Close"].iloc[-1])
            spy_5d_pct = float((spy_hist["Close"].iloc[-1] / spy_hist["Close"].iloc[-5] - 1) * 100.0) if len(spy_hist) >= 5 else 0.0
            st.metric("SPY", f"${spy_last:.2f}", f"{spy_5d_pct:+.1f}% vs 5d ago")
            st.line_chart(spy_hist["Close"].tail(30), height=120)
        else:
            st.info("No SPY data.")
    except Exception:
        st.info("Could not load SPY.")

with macro_col3:
    st.markdown("**Upcoming Economic Events (Finnhub)**")
    if finnhub_api_key:
        econ_df = fetch_economic_calendar_finnhub(finnhub_api_key, days_ahead=7)
        if econ_df is None or econ_df.empty:
            st.info("No economic calendar data (API limits or plan restrictions).")
        else:
            st.caption("Next ~7 days, top events")
            st.dataframe(econ_df.head(15), use_container_width=True, height=180)
    else:
        st.info("Provide a Finnhub API key in the sidebar to load economic calendar.")

if alphavantage_api_key and tickers:
    st.markdown("#### Fundamentals snapshot (Alpha Vantage)")
    base_symbol = tickers[0]
    av_data = fetch_alpha_overview(base_symbol, alphavantage_api_key)
    if av_data:
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            st.metric("Symbol", av_data.get("Symbol", base_symbol))
        with fcol2:
            st.metric("Market Cap", av_data.get("MarketCapitalization", "N/A"))
        with fcol3:
            st.metric("P/E", av_data.get("PERatio", "N/A"))
        with fcol4:
            st.metric("Dividend Yield", av_data.get("DividendYield", "N/A"))
    else:
        st.info("Could not load Alpha Vantage fundamentals (rate limits or invalid key).")

st.markdown("---")

# --------- RTH filter --------- #
if not run_scan:
    st.info("Enter tickers and click **Run Scan** in the sidebar to start.")
    st.stop()

if not tickers:
    st.error("Please enter at least one ticker symbol.")
    st.stop()

if only_rth and not is_us_rth():
    st.warning(
        "Session filter is ON and current time is outside US regular trading hours "
        "(09:30–16:00 US/Eastern, Mon–Fri). Turn this off in the sidebar if you still want to scan."
    )
    st.stop()

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Universe")
    st.write(", ".join(tickers))
with col2:
    st.subheader("Config")
    st.write(f"Max DTE: **{max_dte} days**")
    st.write(f"Min stock volume: **{min_stock_volume:,}**")
    st.write(f"Risk per trade: **{max_risk_per_trade_pct:.1f}%** of ${account_size:,.0f}")
    st.write(f"Max per ticker: **{max_capital_per_ticker_pct:.1f}%** of ${account_size:,.0f}")
    st.write(f"Min POP: **{min_pop_pct:.0f}%**")

# --------- Per-ticker analysis --------- #

for symbol in tickers:
    st.markdown("---")
    st.header(f"🔍 {symbol}")

    try:
        ticker = yf.Ticker(symbol)
    except Exception as e:
        st.error(f"Could not create Ticker for {symbol}: {e}")
        continue

    # --- Price & volume info --- #
    try:
        hist = ticker.history(period="90d")
    except Exception as e:
        hist = pd.DataFrame()
        st.warning(f"Could not load price history for {symbol}: {e}")

    last_price = None
    avg_volume = None
    hv_30 = None

    if not hist.empty:
        last_price = float(hist["Close"].iloc[-1])
        avg_volume = int(hist["Volume"].tail(20).mean())
        hv_30 = realized_volatility(hist, lookback=30)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Last Price", f"${last_price:,.2f}" if last_price else "N/A")
    with m2:
        st.metric("20-day Avg Volume", f"{avg_volume:,}" if avg_volume else "N/A")
    with m3:
        st.metric("30-day Realized Vol", f"{hv_30:.1f}%" if hv_30 else "N/A")

    if avg_volume is not None and avg_volume < min_stock_volume:
        st.warning(
            f"Skipping options scan for {symbol} — average volume {avg_volume:,} "
            f"is below the minimum of {min_stock_volume:,}."
        )
        continue

    exp, dte = get_nearest_expiration(ticker, max_dte=max_dte)
    if not exp or not dte:
        st.warning(f"No expiration within {max_dte} days found for {symbol}.")
        continue

    st.write(f"Using expiration **{exp}** (≈ **{dte} days** to expiry)")

    # --- Options chain --- #
    try:
        chain = ticker.option_chain(exp)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
    except Exception as e:
        st.error(f"Could not load options chain for {symbol}: {e}")
        continue

    tab_summary, tab_puts, tab_calls, tab_news = st.tabs(
        ["📊 Summary", "🟢 Put Credit Spreads", "🔴 Call Credit / Covered Calls", "📰 News"]
    )

    # --- Summary tab --- #
    with tab_summary:
        st.subheader("Raw Options Snapshot")
        st.caption("Top 10 calls & puts by volume for quick inspection.")
        if not calls.empty:
            top_calls = calls.sort_values("volume", ascending=False).head(10)
            st.write("Top Calls (by volume)")
            st.dataframe(
                top_calls[
                    [
                        "contractSymbol",
                        "strike",
                        "lastPrice",
                        "bid",
                        "ask",
                        "volume",
                        "openInterest",
                        "impliedVolatility",
                    ]
                ]
            )
        if not puts.empty:
            top_puts = puts.sort_values("volume", ascending=False).head(10)
            st.write("Top Puts (by volume)")
            st.dataframe(
                top_puts[
                    [
                        "contractSymbol",
                        "strike",
                        "lastPrice",
                        "bid",
                        "ask",
                        "volume",
                        "openInterest",
                        "impliedVolatility",
                    ]
                ]
            )

    # --- Put credit spreads --- #
    with tab_puts:
        if "Put Credit Spreads" in strategies:
            st.subheader("Put Credit Spread Candidates (Bullish)")
            pcs = scan_put_credit_spreads(
                underlying_price=last_price if last_price else 0,
                puts=puts,
                dte=dte,
                min_option_volume=min_option_volume,
                min_oi=min_oi,
                risk_free_rate=risk_free_rate,
                account_size=account_size,
                max_risk_per_trade_pct=max_risk_per_trade_pct,
                max_capital_per_ticker_pct=max_capital_per_ticker_pct,
                min_pop_pct=min_pop_pct,
            )
            if pcs.empty:
                st.info("No put credit spread candidates passed the filters.")
            else:
                view_pcs = pcs.head(max_rows_per_strategy)
                st.dataframe(view_pcs)
                st.caption(
                    "Δ and θ are for the **short put**. POP is approximate probability of profit "
                    "based on Black–Scholes (risk-neutral)."
                )
                add_export_buttons(pcs, f"{symbol}_put_credit_spreads", f"{symbol}_pcs")
        else:
            st.info("Put credit spreads disabled in settings.")

    # --- Call credit spreads & covered calls --- #
    with tab_calls:
        if "Call Credit Spreads" in strategies:
            st.subheader("Call Credit Spread Candidates (Bearish)")
            ccs = scan_call_credit_spreads(
                underlying_price=last_price if last_price else 0,
                calls=calls,
                dte=dte,
                min_option_volume=min_option_volume,
                min_oi=min_oi,
                risk_free_rate=risk_free_rate,
                account_size=account_size,
                max_risk_per_trade_pct=max_risk_per_trade_pct,
                max_capital_per_ticker_pct=max_capital_per_ticker_pct,
                min_pop_pct=min_pop_pct,
            )
            if ccs.empty:
                st.info("No call credit spread candidates passed the filters.")
            else:
                view_ccs = ccs.head(max_rows_per_strategy)
                st.dataframe(view_ccs)
                st.caption(
                    "Δ and θ are for the **short call**. POP is approximate probability of profit "
                    "based on Black–Scholes (risk-neutral)."
                )
                add_export_buttons(ccs, f"{symbol}_call_credit_spreads", f"{symbol}_ccs")
        else:
            st.info("Call credit spreads disabled in settings.")

        st.markdown("---")

        if "Covered Calls" in strategies:
            st.subheader("Covered Call Candidates")
            cov = scan_covered_calls(
                underlying_price=last_price if last_price else 0,
                calls=calls,
                dte=dte,
                min_option_volume=min_option_volume,
                min_oi=min_oi,
                account_size=account_size,
                max_capital_per_ticker_pct=max_capital_per_ticker_pct,
            )
            if cov.empty:
                st.info("No covered call candidates passed the filters.")
            else:
                view_cov = cov.head(max_rows_per_strategy)
                st.dataframe(view_cov)
                st.caption(
                    "Each covered call assumes 100 shares owned. "
                    "Max contracts is limited by your per-ticker capital setting."
                )
                add_export_buttons(cov, f"{symbol}_covered_calls", f"{symbol}_cov")
        else:
            st.info("Covered calls disabled in settings.")

    # --- News tab --- #
    with tab_news:
        st.subheader(f"Recent Headlines for {symbol}")
        try:
            news_items = ticker.news or []
        except Exception:
            news_items = []
        if not news_items:
            st.info("No news found from Yahoo Finance.")
        else:
            max_age_days = 7
            cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
            shown = 0
            for item in news_items:
                ts = item.get("providerPublishTime") or item.get("publishTime")
                dt = None
                if ts:
                    try:
                        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    except Exception:
                        dt = None
                if dt and dt < cutoff:
                    continue
                title = item.get("title", "No title")
                publisher = item.get("publisher", "Unknown")
                link = item.get("link")
                time_str = dt.strftime("%Y-%m-%d %H:%M UTC") if dt else "Unknown time"

                st.markdown(f"**{title}**")
                st.caption(f"{publisher} · {time_str}")
                if link:
                    st.markdown(f"[Open article]({link})")
                st.markdown("---")
                shown += 1
                if shown >= 10:
                    break
            if shown == 0:
                st.info("No news in the last 7 days.")
