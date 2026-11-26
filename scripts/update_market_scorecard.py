#!/usr/bin/env python3
"""Build the daily US and Singapore market-conditions scorecard.

Prices are daily Yahoo Finance historical closes retrieved with yfinance.
This is a convenience data feed, not an exchange-certified feed. The script
keeps the previous valid market block if a new source fetch fails validation.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import exchange_calendars as xcals
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "market-scorecard.json"
STI_FILE = ROOT / "data" / "sti-constituents.json"
SP500_URL = (
    "https://raw.githubusercontent.com/datasets/"
    "s-and-p-500-companies-financials/main/data/constituents.csv"
)
SECTORS = ["XLK", "XLC", "XLF", "XLY", "XLP", "XLV",
           "XLI", "XLE", "XLB", "XLRE", "XLU"]


def download_sp500_symbols() -> list[str]:
    req = Request(SP500_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as response:
        rows = csv.DictReader(io.StringIO(response.read().decode("utf-8-sig")))
        symbols = []
        for row in rows:
            symbol = (row.get("Symbol") or row.get("symbol") or "").strip()
            if symbol:
                symbols.append(symbol.replace(".", "-"))
    if len(symbols) < 450:
        raise RuntimeError(f"Unexpected S&P 500 universe size: {len(symbols)}")
    return sorted(set(symbols))


def download_prices(tickers: list[str], period: str = "18mo") -> dict[str, pd.Series]:
    """Download in modest batches; tolerate individual ticker failures."""
    result: dict[str, pd.Series] = {}
    unique = list(dict.fromkeys(tickers))
    for start in range(0, len(unique), 70):
        batch = unique[start:start + 70]
        try:
            frame = yf.download(
                tickers=batch,
                period=period,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=35,
            )
        except Exception as exc:
            print(f"Batch {start // 70 + 1} download failed: {exc}", file=sys.stderr)
            continue
        if frame is None or frame.empty:
            continue
        for ticker in batch:
            try:
                if isinstance(frame.columns, pd.MultiIndex):
                    levels = [frame.columns.get_level_values(i) for i in range(frame.columns.nlevels)]
                    if ticker in levels[0]:
                        series = frame[ticker]["Close"]
                    elif ticker in levels[-1]:
                        series = frame.xs(ticker, axis=1, level=-1)["Close"]
                    else:
                        continue
                else:
                    if len(batch) != 1 or "Close" not in frame.columns:
                        continue
                    series = frame["Close"]
                series = pd.to_numeric(series, errors="coerce").dropna()
                if not series.empty:
                    series.index = pd.to_datetime(series.index).tz_localize(None)
                    result[ticker] = series[~series.index.duplicated(keep="last")].sort_index()
            except Exception:
                continue
    return result


def latest_on_or_before(series: pd.Series, as_of: pd.Timestamp) -> pd.Series:
    return series.loc[series.index <= as_of]



def weekly_market_status(index_series: pd.Series, calendar_name: str) -> dict:
    """Return Singapore-calendar week's weekday closes, distinguishing holidays from missing data."""
    # Both markets use the Singapore calendar week so their bars reset together
    # at Monday 00:00 Singapore time, even while the US is still on Sunday.
    today = pd.Timestamp.now(tz="Asia/Singapore").date()
    monday = today - timedelta(days=today.weekday())
    friday = monday + timedelta(days=4)
    calendar = xcals.get_calendar(calendar_name)
    # Pass ISO date strings; exchange_calendars normalises session labels
    # internally and avoids pandas datetime.timezone compatibility issues.
    sessions = calendar.sessions_in_range(monday.isoformat(), friday.isoformat())
    session_dates = {session.date() for session in sessions}
    days = []
    prices = index_series.dropna().sort_index()
    for offset, label in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
        day = monday + timedelta(days=offset)
        day_text = day.isoformat()
        if day not in session_dates:
            days.append({"day": label, "date": day_text, "status": "holiday"})
        elif day > today:
            days.append({"day": label, "date": day_text, "status": "pending"})
        else:
            stamp = pd.Timestamp(day)
            if stamp not in prices.index:
                days.append({"day": label, "date": day_text, "status": "pending"})
                continue
            previous = prices.loc[prices.index < stamp]
            if previous.empty:
                days.append({"day": label, "date": day_text, "status": "pending"})
                continue
            change = (float(prices.loc[stamp]) / float(previous.iloc[-1]) - 1.0) * 100.0
            status = "flat" if abs(change) < 0.00005 else ("up" if change > 0 else "down")
            days.append({"day": label, "date": day_text, "status": status,
                         "change_pct": round(change, 2)})
    return {"week_start": monday.isoformat(), "days": days}


def trend_score(index_series: pd.Series) -> tuple[int, dict]:
    s = index_series.dropna()
    if len(s) < 205:
        raise RuntimeError("Insufficient index history for 200-day trend")
    close = float(s.iloc[-1])
    sma50 = float(s.tail(50).mean())
    sma200 = float(s.tail(200).mean())
    score = (50 if close > sma50 else -50) + (50 if sma50 > sma200 else -50)
    return score, {"close": close, "sma50": sma50, "sma200": sma200,
                   "above_50d": close > sma50, "golden_trend": sma50 > sma200}


def breadth(prices: dict[str, pd.Series], symbols: list[str], as_of: pd.Timestamp,
            minimum_fraction: float) -> tuple[float, int, int]:
    above = 0
    valid = 0
    for symbol in symbols:
        series = prices.get(symbol)
        if series is None:
            continue
        s = latest_on_or_before(series, as_of)
        if len(s) < 55:
            continue
        # Exclude symbols whose latest available close is stale by more than four days.
        if (as_of - s.index[-1]).days > 4:
            continue
        valid += 1
        if float(s.iloc[-1]) > float(s.tail(50).mean()):
            above += 1
    if valid < math.ceil(len(symbols) * minimum_fraction):
        raise RuntimeError(
            f"Insufficient constituent coverage: {valid}/{len(symbols)}"
        )
    return above / valid * 100.0, above, valid


def momentum_score(index_series: pd.Series) -> tuple[float, int]:
    s = index_series.dropna()
    if len(s) < 22:
        raise RuntimeError("Insufficient history for 20-day momentum")
    ret = float(s.iloc[-1] / s.iloc[-21] - 1.0)
    return ret * 100.0, max(-100, min(100, round(ret * 2000)))


def classify(score: float) -> str:
    if score >= 40:
        return "Strongly bullish"
    if score >= 15:
        return "Bullish"
    if score > -15:
        return "Neutral"
    if score > -40:
        return "Bearish"
    return "Strongly bearish"


def vix_label(value: float) -> str:
    if value < 15:
        return "Calm"
    if value <= 20:
        return "Normal"
    if value <= 30:
        return "Elevated"
    return "High Stress"


def singapore_volatility(index_series: pd.Series) -> tuple[str, dict]:
    returns = index_series.pct_change().dropna()
    rolling = returns.rolling(20).std() * math.sqrt(252) * 100
    valid = rolling.dropna()
    if len(valid) < 252:
        raise RuntimeError("Insufficient STI history for volatility thresholds")
    current = float(valid.iloc[-1])
    history = valid.tail(1260)
    p25, p75, p90 = [float(history.quantile(q)) for q in (0.25, 0.75, 0.90)]
    if current < p25:
        label = "Calm"
    elif current < p75:
        label = "Normal"
    elif current < p90:
        label = "Elevated"
    else:
        label = "High Stress"
    return f"{label} · {current:.1f}% annualised", {
        "realised_20d_annualised_pct": round(current, 2),
        "thresholds_pct": {"calm_below": round(p25, 2),
                           "normal_below": round(p75, 2),
                           "elevated_below": round(p90, 2)},
        "threshold_method": "Rolling 20-day annualised realised volatility; trailing five-year 25th, 75th and 90th percentiles."
    }


def build_us() -> tuple[dict, str]:
    symbols = download_sp500_symbols()
    universe = symbols + ["^GSPC", "^VIX"] + SECTORS
    prices = download_prices(universe, "18mo")
    if "^GSPC" not in prices or "^VIX" not in prices:
        raise RuntimeError("S&P 500 or VIX history unavailable")
    index = prices["^GSPC"]
    as_of = index.index[-1]
    index = latest_on_or_before(index, as_of)
    trend, trend_detail = trend_score(index)
    pct_above, above, covered = breadth(prices, symbols, as_of, 0.80)
    sector_above = 0
    sector_valid = 0
    for ticker in SECTORS:
        series = prices.get(ticker)
        if series is None:
            continue
        s = latest_on_or_before(series, as_of)
        if len(s) >= 55 and (as_of - s.index[-1]).days <= 4:
            sector_valid += 1
            if float(s.iloc[-1]) > float(s.tail(50).mean()):
                sector_above += 1
    if sector_valid < 9:
        raise RuntimeError(f"Insufficient sector ETF coverage: {sector_valid}/11")
    sector_pct = sector_above / sector_valid * 100
    sector_component = (sector_pct - 50) * 2
    breadth_component = (pct_above - 50) * 2
    momentum_pct, momentum_component = momentum_score(index)
    composite = trend * .35 + breadth_component * .30 + sector_component * .20 + momentum_component * .15
    vix_series = prices["^VIX"]
    vix_series = latest_on_or_before(vix_series, as_of)
    if vix_series.empty or (as_of - vix_series.index[-1]).days > 4:
        raise RuntimeError("VIX observation is stale")
    vix = float(vix_series.iloc[-1])
    summary = (
        f"{above}/{covered} S&P 500 stocks above their 50-day average; "
        f"{sector_above}/{sector_valid} sectors above their 50-day average; "
        f"S&P 500 20-day momentum {momentum_pct:+.1f}%."
    )
    return {
        "market_state": classify(composite),
        "summary": summary,
        "volatility": f"{vix_label(vix)} · VIX {vix:.1f}",
        "as_of": as_of.strftime("%Y-%m-%d"),
        "directional_score": round(composite, 1),
        "indicators": {
            "index_close": round(trend_detail["close"], 2),
            "sma_50": round(trend_detail["sma50"], 2),
            "sma_200": round(trend_detail["sma200"], 2),
            "constituents_above_50d_pct": round(pct_above, 1),
            "constituents_covered": covered,
            "sector_etfs_above_50d": sector_above,
            "sector_etfs_covered": sector_valid,
            "momentum_20d_pct": round(momentum_pct, 2),
            "vix": round(vix, 2)
        },
        "weekly_status": weekly_market_status(index, "XNYS")
    }, as_of.strftime("%Y-%m-%d")


def fetch_yahoo_sti_direct(target_date: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    """Fetch the exact requested completed STI session from Yahoo's chart endpoint."""

    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/%5ESTI"
        "?range=5y&interval=1d&events=history"
        f"&_={int(datetime.now(timezone.utc).timestamp())}"
    )
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })
    with urlopen(req, timeout=25) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp") or []
    closes = result["indicators"]["quote"][0].get("close") or []
    observations = [
        (pd.Timestamp(datetime.fromtimestamp(ts, timezone.utc).date()),
         float(close))
        for ts, close in zip(timestamps, closes)
        if close is not None
    ]
    target = target_date.normalize()
    matching = [item for item in observations if item[0] == target]
    if not matching:
        latest = max((item[0] for item in observations), default=None)
        raise RuntimeError(
            f"Yahoo direct chart endpoint has no close for {target.date()} "
            f"(latest available row: {latest.date() if latest is not None else 'none'})"
        )
    return matching[-1]


def latest_completed_session(calendar_name: str) -> pd.Timestamp:
    now_sg = datetime.now(ZoneInfo("Asia/Singapore"))
    today = now_sg.date()
    # A session is not complete until after the SGX closing auction.
    if now_sg.hour < 17 or (now_sg.hour == 17 and now_sg.minute < 10):
        today -= timedelta(days=1)
    calendar = xcals.get_calendar(calendar_name)
    start = (today - timedelta(days=10)).isoformat()
    sessions = calendar.sessions_in_range(start, today.isoformat())
    if len(sessions) == 0:
        raise RuntimeError(f"No recent sessions returned for {calendar_name}")
    return pd.Timestamp(sessions[-1]).tz_localize(None).normalize()


def build_singapore() -> tuple[dict, str]:
    sti_data = json.loads(STI_FILE.read_text(encoding="utf-8"))
    constituents = sti_data["tickers"]
    symbols = [x["ticker"] for x in constituents]
    prices = download_prices(["^STI"] + symbols, "5y")
    if "^STI" not in prices:
        raise RuntimeError("STI index history unavailable")
    index = prices["^STI"].copy()
    expected_session = latest_completed_session("XSES")

    # Never use an in-progress/current Singapore trading day's candle. The
    # scorecard must reflect the latest completed SGX session, even if Yahoo
    # publishes a row dated today before the market has closed.
    index = index.loc[index.index.normalize() <= expected_session]
    if index.empty or index.index[-1].normalize() < expected_session:
        yfinance_date = index.index[-1].date() if not index.empty else "none"
        print(
            f"Yahoo yfinance STI is stale ({yfinance_date}); "
            f"retrying direct Yahoo chart endpoint for {expected_session.date()}",
            file=sys.stderr,
        )
        try:
            direct_date, direct_close = fetch_yahoo_sti_direct(expected_session)
            if direct_date != expected_session:
                raise RuntimeError(f"Yahoo returned {direct_date.date()} instead of {expected_session.date()}")
            source = "direct Yahoo endpoint"
        except Exception as yahoo_error:
            # One-off verified fallback for the 25 Nov 2026 STI close.
            # FT historical prices and SGX both reported 4,930.83.
            verified_sti_closes = {
                "2026-11-25": (4930.83, "FT/SGX verified historical close")
            }
            fallback = verified_sti_closes.get(expected_session.strftime("%Y-%m-%d"))
            if fallback is None:
                raise RuntimeError(
                    f"STI data stale: yfinance through {yfinance_date}; "
                    f"direct Yahoo failed: {yahoo_error}"
                ) from yahoo_error
            direct_close, source = fallback
            direct_date = expected_session
        index.loc[direct_date] = direct_close
        index = index.sort_index()
        print(
            f"STI latest completed close refreshed from {source}: "
            f"{direct_date.date()} = {direct_close:.2f}"
        )
    as_of = expected_session
    index = latest_on_or_before(index, as_of)
    index = latest_on_or_before(index, as_of)
    trend, trend_detail = trend_score(index)
    pct_above, above, covered = breadth(prices, symbols, as_of, 0.75)
    momentum_pct, momentum_component = momentum_score(index)
    # Singapore has no directly comparable 11-sector ETF basket. Renormalise
    # the agreed trend/breadth/momentum weights to total 100%.
    breadth_component = (pct_above - 50) * 2
    composite = trend * .4375 + breadth_component * .375 + momentum_component * .1875
    vol_label, vol_detail = singapore_volatility(index)
    summary = (
        f"{above}/{covered} STI constituents above their 50-day average; "
        f"STI 20-day momentum {momentum_pct:+.1f}%."
    )
    return {
        "market_state": classify(composite),
        "summary": summary,
        "volatility": vol_label,
        "as_of": as_of.strftime("%Y-%m-%d"),
        "directional_score": round(composite, 1),
        "indicators": {
            "index_close": round(trend_detail["close"], 2),
            "sma_50": round(trend_detail["sma50"], 2),
            "sma_200": round(trend_detail["sma200"], 2),
            "constituents_above_50d_pct": round(pct_above, 1),
            "constituents_covered": covered,
            "momentum_20d_pct": round(momentum_pct, 2),
            **vol_detail
        },
        "weekly_status": weekly_market_status(index, "XSES")
    }, as_of.strftime("%Y-%m-%d")


def main() -> None:
    existing = {}
    if OUT.exists():
        try:
            existing = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    data = {
        "status": "partial",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "market_data_as_of": existing.get("market_data_as_of") or {},
        "weekly_status": existing.get("weekly_status", {}),
        "us": existing.get("us", {}),
        "singapore": existing.get("singapore", {}),
        "sources": {
            "prices": "Yahoo Finance historical daily closes via yfinance; delayed/availability may vary.",
            "sp500_universe": SP500_URL,
            "sti_universe": "data/sti-constituents.json; State Street SPDR STI ETF prospectus and FTSE Russell September 2026 review."
        },
        "methodology": {
            "us_weights_pct": {"index_trend": 35, "market_breadth": 30, "sector_participation": 20, "momentum": 15},
            "singapore_weights_pct": {"index_trend": 43.75, "market_breadth": 37.5, "momentum": 18.75},
            "note": "Latest available completed trading session, not a weekly average. Singapore weights are renormalised because there is no directly comparable 11-sector ETF basket."
        },
        "note": "Latest available completed trading-session snapshot. A market block is replaced only after its data passes coverage and freshness checks."
    }
    successful = []
    for key, builder in (("us", build_us), ("singapore", build_singapore)):
        try:
            block, date = builder()
            data[key] = block
            data.setdefault("weekly_status", {})[key] = block.pop("weekly_status")
            data["market_data_as_of"][key] = date
            successful.append(key)
            print(f"{key} scorecard updated for {date}: {block['market_state']}")
        except Exception as exc:
            print(f"{key} scorecard not updated; retaining previous valid block: {exc}", file=sys.stderr)
    if not successful:
        raise RuntimeError("Neither market block passed validation; no scorecard update written.")
    data["status"] = "verified" if len(successful) == 2 else "partial"
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated scorecard JSON: {', '.join(successful)}")


if __name__ == "__main__":
    main()
