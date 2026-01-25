#!/usr/bin/env python3
"""
economics.py

Fetch key economic indicators (FRED) and market prices (Yahoo Finance),
print a daily summary, optionally plot historical series from a start date,
and write simple artifacts for a Streamlit dashboard.

Artifacts written to --out-dir:
- latest.json    (latest snapshot rows)
- history.csv    (append-only history of snapshots)
- metrics_macro.png, metrics_markets.png  (if --plot)

Data sources:
- FRED API (macro indicators): https://fred.stlouisfed.org/
- Yahoo Finance via yfinance (market prices)
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Optional, List, Tuple

import requests
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; script still works if env vars are set another way
    pass


# ----------------------------
# Configuration
# ----------------------------

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# FRED series IDs:
# CPIAUCSL: CPI for All Urban Consumers (Index, monthly)
# UNRATE: Unemployment Rate (monthly)
# EFFR: Effective Federal Funds Rate (daily)
# DGS10: 10-Year Treasury Constant Maturity Rate (daily, percent)
FRED_SERIES = {
    "Unemployment Rate": "UNRATE",
    "Effective Fed Funds Rate": "EFFR",
    "10Y Treasury Yield": "DGS10",
}

# Market tickers via yfinance:
MARKET_TICKERS = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Dow Jones": "^DJI",
    "VIX": "^VIX",
}


# ----------------------------
# Data Structures
# ----------------------------

@dataclass
class MetricPoint:
    name: str
    value: Optional[float]
    date: Optional[str]
    units: str = ""


# ----------------------------
# Helpers
# ----------------------------

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s in ("", ".", "nan", "NaN", "None"):
            return None
        return float(s)
    except Exception:
        return None


def fred_get_latest_observation(
    series_id: str,
    api_key: str,
    *,
    observation_start: str = "1900-01-01",
) -> Tuple[Optional[float], Optional[str]]:
    """
    Fetch the latest non-missing observation for a FRED series.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 10,  # grab a few in case latest is "."
        "observation_start": observation_start,
    }
    r = requests.get(FRED_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    obs = data.get("observations", [])
    for row in obs:
        val = _safe_float(row.get("value"))
        if val is not None:
            return val, row.get("date")
    return None, None


def fred_get_series(
    series_id: str,
    api_key: str,
    *,
    observation_start: str,
) -> pd.Series:
    """
    Fetch a full FRED series from observation_start to latest, as a pandas Series.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
        "observation_start": observation_start,
        "limit": 100000,
    }
    r = requests.get(FRED_BASE, params=params, timeout=45)
    r.raise_for_status()
    data = r.json()

    obs = data.get("observations", [])
    dates, values = [], []
    for row in obs:
        v = _safe_float(row.get("value"))
        d = row.get("date")
        if v is not None and d:
            dates.append(pd.to_datetime(d))
            values.append(v)

    if not dates:
        return pd.Series(dtype="float64", name=series_id)

    s = pd.Series(values, index=pd.DatetimeIndex(dates)).sort_index()
    s.name = series_id
    return s


def compute_yoy_from_index(index_series: pd.Series, periods: int = 12) -> pd.Series:
    """
    YoY% = (x / x.shift(periods) - 1) * 100
    """
    if index_series is None or index_series.empty:
        return pd.Series(dtype="float64")
    yoy = (index_series / index_series.shift(periods) - 1.0) * 100.0
    return yoy.dropna()


def compute_cpi_yoy_from_fred(
    api_key: str,
    *,
    series_id: str = "CPIAUCSL",
    years_back: int = 8,
) -> MetricPoint:
    """
    Compute CPI YoY inflation from CPI index:
      YoY% = (CPI_t / CPI_{t-12} - 1) * 100

    Pulls only recent history to ensure we compute the most recent YoY.
    """
    start_year = date.today().year - years_back
    observation_start = f"{start_year}-01-01"

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",   # newest first
        "limit": 300,           # enough monthly points
        "observation_start": observation_start,
    }
    r = requests.get(FRED_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    obs = data.get("observations", [])
    rows = []
    for row in obs:
        v = _safe_float(row.get("value"))
        d = row.get("date")
        if v is not None and d:
            rows.append((d, v))

    if len(rows) < 13:
        return MetricPoint("US CPI YoY", None, None, units="%")

    # rows are newest->oldest; reverse for shift(12)
    rows.reverse()

    s = pd.Series(
        data=[v for _, v in rows],
        index=pd.to_datetime([d for d, _ in rows]),
    ).sort_index()

    yoy = (s / s.shift(12) - 1.0) * 100.0
    yoy = yoy.dropna()
    if yoy.empty:
        return MetricPoint("US CPI YoY", None, None, units="%")

    last_val = float(yoy.iloc[-1])
    last_date = yoy.index[-1].date().isoformat()
    return MetricPoint("US CPI YoY", last_val, last_date, units="%")


def yahoo_get_latest_close(ticker: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Fetch latest close price via yfinance.
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

    t = yf.Ticker(ticker)
    hist = t.history(period="7d", interval="1d")  # handles weekends/holidays
    if hist is None or hist.empty:
        return None, None

    last = hist.iloc[-1]
    close = _safe_float(last.get("Close"))
    d = hist.index[-1].date().isoformat()
    return close, d


def yahoo_get_series(
    ticker: str,
    *,
    start_date: str,
) -> pd.Series:
    """
    Fetch historical close prices from Yahoo Finance via yfinance.
    Always returns a pandas Series of Close prices.
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

    df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
    if df is None or len(df) == 0:
        return pd.Series(dtype="float64", name=ticker)

    # Sometimes columns are MultiIndex.
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        else:
            # fallback: pick first column under 'Close'
            s = df.xs("Close", axis=1, level=0).iloc[:, 0]
    else:
        s = df["Close"]

    s = pd.to_numeric(s, errors="coerce").dropna()
    s.name = ticker
    return s


def format_metric(mp: MetricPoint) -> str:
    if mp.value is None:
        return f"- {mp.name}: (no data)"
    if mp.units == "%":
        return f"- {mp.name}: {mp.value:.2f}{mp.units} (as of {mp.date})"
    return f"- {mp.name}: {mp.value:,.2f} {mp.units}".rstrip() + (f" (as of {mp.date})" if mp.date else "")

def plot_macro_with_cpi_rhs(
    *,
    cpi_yoy: pd.Series,
    left_series: list[tuple[str, pd.Series]],
    title: str,
    output_png: str = "",
):
    left_series = [(name, s) for name, s in left_series if s is not None and not s.empty]
    if cpi_yoy is None or cpi_yoy.empty:
        print("CPI YoY series is empty; cannot make RHS plot.")
        return
    if not left_series:
        print("No left-axis series to plot.")
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    cpi_color = colors[2]  # C2 (green) — distinct from blue & orange

    fig, ax_left = plt.subplots()

    # Left axis series (keep default colors)
    linestyles = ["-", "--", "-.", ":"]
    for i, (name, s) in enumerate(left_series):
        ax_left.plot(
            s.index, s.values,
            label=name,
            linestyle=linestyles[i % len(linestyles)],
        )

    ax_left.set_xlabel("Date")
    ax_left.set_ylabel("Rate (%)")

    # Right axis for CPI YoY (use a different default-cycle color than the first left line)
    ax_right = ax_left.twinx()

    cpi_color = colors[4] #if len(colors) > 1 else None  # "C1" orange in default matplotlib
    ax_right.plot(
        cpi_yoy.index,
        cpi_yoy.values,
        label="US CPI YoY (%)",
        linestyle="--",
        linewidth=2.0,
        color=cpi_color,
    )
    ax_right.set_ylabel("CPI YoY (%)", color=cpi_color)
    ax_right.tick_params(axis="y", colors=cpi_color)

    # One combined legend
    left_handles, left_labels = ax_left.get_legend_handles_labels()
    right_handles, right_labels = ax_right.get_legend_handles_labels()
    ax_left.legend(left_handles + right_handles, left_labels + right_labels, loc="best")

    plt.title(title)
    fig.tight_layout()

    # Left axis grid (neutral)
    ax_left.grid(
        True,
        which="major",
        axis="both",
        linestyle=":",
        linewidth=0.6,
        alpha=0.5,
    )

    # Right axis grid (CPI-colored, Y only)
    ax_right.grid(
        True,
        which="major",
        axis="y",
        color=cpi_color,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )

    ax_left.set_axisbelow(True)
    ax_right.set_axisbelow(True)

    if output_png:
        plt.savefig(output_png, dpi=150)
        print(f"Saved plot: {output_png}")
    else:
        plt.show()

def plot_series(series_list, *, title: str, output_png: str = ""):
    series_list = [(name, s) for name, s in series_list if s is not None and not s.empty]

    plt.figure()

    if not series_list:
        plt.title(title)
        plt.text(
            0.5, 0.5, "No data available",
            ha="center", va="center",
            transform=plt.gca().transAxes,
        )
        plt.axis("off")
        plt.tight_layout()
        if output_png:
            plt.savefig(output_png, dpi=150)
            print(f"Saved placeholder plot: {output_png}")
        else:
            plt.show()
        return

    for name, s in series_list:
        plt.plot(s.index, s.values, label=name)

    plt.title(title)
    plt.xlabel("Date")
    plt.legend()

    # --- Grid
    plt.grid(
        True,
        which="major",
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    plt.tight_layout()

    if output_png:
        plt.savefig(output_png, dpi=150)
        print(f"Saved plot: {output_png}")
    else:
        plt.show()


# ----------------------------
# Main logic
# ----------------------------

def build_metrics(fred_api_key: str) -> List[MetricPoint]:
    metrics: List[MetricPoint] = []

    # Inflation (computed YoY)
    metrics.append(compute_cpi_yoy_from_fred(fred_api_key))

    # Other FRED latest points
    for name, sid in FRED_SERIES.items():
        val, d = fred_get_latest_observation(sid, fred_api_key)
        units = "%"  # all of these are percentages
        metrics.append(MetricPoint(name, val, d, units=units))

    # Market latest close
    for name, ticker in MARKET_TICKERS.items():
        val, d = yahoo_get_latest_close(ticker)
        metrics.append(MetricPoint(name, val, d, units=""))

    return metrics


def metrics_to_dataframe(metrics: List[MetricPoint]) -> pd.DataFrame:
    return pd.DataFrame([{
        "name": m.name,
        "value": m.value,
        "date": m.date,
        "units": m.units,
    } for m in metrics])


def write_dashboard_artifacts(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Writes:
      - latest.json (snapshot)
      - history.csv (append-only)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    latest_path = out_dir / "latest.json"
    latest_path.write_text(df.to_json(orient="records", indent=2))
    print(f"Wrote {latest_path}")

    history_path = out_dir / "history.csv"
    if history_path.exists():
        old = pd.read_csv(history_path)
        new = pd.concat([old, df], ignore_index=True)
        new = new.drop_duplicates(subset=["retrieved_at_utc", "name"], keep="last")
        new.to_csv(history_path, index=False)
    else:
        df.to_csv(history_path, index=False)
    print(f"Updated {history_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch key economic metrics (FRED + Yahoo Finance).")
    parser.add_argument("--save-csv", type=str, default="", help="Path to save a CSV snapshot (optional).")
    parser.add_argument("--json", action="store_true", help="Print output as JSON instead of text.")
    parser.add_argument("--start-date", type=str, default="2015-01-01", help="Start date for historical series (YYYY-MM-DD).")
    parser.add_argument("--plot", action="store_true", help="Plot historical series from --start-date to latest.")
    parser.add_argument("--plot-out", type=str, default="metrics", help="Plot filename prefix.")
    parser.add_argument("--out-dir", type=str, default="dashboard_data", help="Directory to write dashboard output files.")
    args = parser.parse_args()

    # Silence noisy warnings (yfinance / pandas)
    warnings.filterwarnings("ignore", message=".*Timestamp\\.utcnow is deprecated.*")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    fred_api_key = os.getenv("FRED_API_KEY", "").strip()
    if not fred_api_key:
        print("ERROR: Missing FRED_API_KEY.", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    # ----------------------------
    # Snapshot metrics
    # ----------------------------
    try:
        metrics = build_metrics(fred_api_key)
    except Exception as e:
        print(f"ERROR: Failed to fetch metrics: {e}", file=sys.stderr)
        return 3

    df = metrics_to_dataframe(metrics)
    df.insert(0, "retrieved_at_utc", now_utc)

    # ----------------------------
    # Historical plots
    # ----------------------------
    if args.plot:
        start_date = args.start_date

        # --- Macro (CPI YoY on RHS)
        cpi_index = fred_get_series("CPIAUCSL", fred_api_key, observation_start=start_date)
        cpi_yoy = compute_yoy_from_index(cpi_index, periods=12)

        unrate = fred_get_series("UNRATE", fred_api_key, observation_start=start_date)
        effr   = fred_get_series("EFFR", fred_api_key, observation_start=start_date)
        dgs10  = fred_get_series("DGS10", fred_api_key, observation_start=start_date)

        macro_png = out_dir / f"{args.plot_out}_macro.png"

        plot_macro_with_cpi_rhs(
            cpi_yoy=cpi_yoy,
            left_series=[
                ("Unemployment Rate (%)", unrate),
                ("EFFR (%)", effr),
                ("10Y Treasury Yield (%)", dgs10),
            ],
            title=f"US Macro Indicators (CPI YoY on RHS, from {start_date})",
            output_png=str(macro_png),
        )

        # --- Markets (indices + VIX)
        market_series = []
        for name, ticker in MARKET_TICKERS.items():
            s = yahoo_get_series(ticker, start_date=start_date)

            # Normalize ^TNX if still present (safe guard)
            if ticker == "^TNX" and not s.empty:
                s = s.copy()
                s = s.apply(lambda v: v / 10.0 if v > 20 else v)

            market_series.append((name, s))

        markets_png = out_dir / f"{args.plot_out}_markets.png"

        plot_series(
            market_series,
            title=f"Market Indicators (from {start_date})",
            output_png=str(markets_png),
        )

    # ----------------------------
    # Write dashboard artifacts (ALWAYS)
    # ----------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latest_path = out_dir / "latest.json"
    tmp_latest = out_dir / "latest.json.tmp"
    tmp_latest.write_text(df.to_json(orient="records", indent=2))
    os.replace(tmp_latest, latest_path)  # atomic replace on macOS

    history_path = out_dir / "history.csv"
    if history_path.exists():
        df.to_csv(history_path, mode="a", header=False, index=False)
    else:
        df.to_csv(history_path, index=False)

    # ----------------------------
    # Output
    # ----------------------------
    if args.json:
        print(json.dumps(df.to_dict(orient="records"), indent=2))
    else:
        print(f"Economic Metrics Snapshot (retrieved {now_utc})")
        print("=" * 50)
        for m in metrics:
            print(format_metric(m))

    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        if not args.json:
            print(f"\nSaved CSV: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())