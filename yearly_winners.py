"""Buy last year's winners and hold them for one calendar year.

Rebalance at the last trading day's close. Top 1 is 100%.
Top 2 is 55% / 45%, with the larger weight on the bigger winner.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

FEE = 0.001


def year_end_dates(index: pd.DatetimeIndex) -> dict[int, pd.Timestamp]:
    idx = pd.DatetimeIndex(pd.to_datetime(index)).normalize().unique().sort_values()
    out = {}
    grouped = pd.Series(1, index=idx).groupby(idx.year)
    for year, group in grouped:
        out[int(year)] = pd.Timestamp(group.index[-1])
    return out


def price_on(series: pd.Series, day: pd.Timestamp) -> float | None:
    cut = series.dropna().loc[:day]
    if cut.empty:
        return None
    if day - cut.index[-1] > pd.Timedelta(days=10):
        return None
    value = float(cut.iloc[-1])
    if value <= 0:
        return None
    return value


def prior_returns(closes: dict[str, pd.Series], names: list[str], ends: dict[int, pd.Timestamp], year: int) -> dict[str, float]:
    start = ends.get(year - 1)
    finish = ends.get(year)
    if start is None or finish is None:
        return {}
    out = {}
    for ticker in names:
        series = closes.get(ticker)
        if series is None:
            continue
        a = price_on(series, start)
        b = price_on(series, finish)
        if a is None or b is None:
            continue
        out[ticker] = b / a - 1.0
    return out


def choose(returns: dict[str, float], top_n: int, weights: tuple[float, ...]) -> list[dict]:
    ranked = sorted(returns.items(), key=lambda item: item[1], reverse=True)[:top_n]
    if not ranked:
        return []
    use = list(weights[: len(ranked)])
    scale = sum(use)
    rows = []
    for (ticker, ret), weight in zip(ranked, use):
        rows.append({"ticker": ticker, "weight": weight / scale, "prior_return": float(ret)})
    return rows


def _metrics(equity: pd.Series) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return {"cagr": 0.0, "max_dd": 0.0, "total": 0.0, "years": 0.0}
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = (1.0 + total) ** (1.0 / years) - 1.0
    max_dd = float((equity / equity.cummax() - 1.0).min())
    return {"cagr": cagr, "max_dd": max_dd, "total": total, "years": years}


def run_decade(
    closes: dict[str, pd.Series],
    names: list[str],
    bench: str,
    top_n: int,
    weights: tuple[float, ...],
    first_hold_year: int,
    last_hold_year: int,
    allowed: dict[int, set[str]] | None = None,
) -> dict:
    bench_px = closes[bench].dropna().sort_index()
    bench_px.index = pd.DatetimeIndex(bench_px.index).normalize()
    ends = year_end_dates(bench_px.index)
    frame = pd.DataFrame({ticker: closes[ticker] for ticker in names if ticker in closes})
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    frame = frame.sort_index().ffill()
    daily = frame.pct_change().fillna(0.0)
    start = ends[first_hold_year - 1]
    finish = ends[last_hold_year]
    window = daily.loc[start:finish].index
    equity = 1.0
    curve = []
    rows = []
    held: list[dict] = []
    year_base = 1.0
    for day in window:
        year = int(day.year)
        if held:
            ret = 0.0
            for item in held:
                if item["ticker"] in daily.columns:
                    ret += item["weight"] * float(daily.at[day, item["ticker"]])
            equity *= 1.0 + ret
        curve.append(equity)
        if day == ends.get(year) and first_hold_year <= year <= last_hold_year and held:
            rows.append(
                {
                    "year": year,
                    "picks": [
                        {"ticker": item["ticker"], "weight": item["weight"], "prior_return": item["prior_return"]}
                        for item in held
                    ],
                    "return": equity / year_base - 1.0,
                }
            )
        if day == ends.get(year) and first_hold_year - 1 <= year <= last_hold_year - 1:
            pool = names
            if allowed is not None:
                pool = [ticker for ticker in names if ticker in allowed.get(year, set())]
            new = choose(prior_returns(closes, pool, ends, year), top_n, weights)
            old = {item["ticker"]: item["weight"] for item in held}
            new_map = {item["ticker"]: item["weight"] for item in new}
            turn = sum(abs(new_map.get(ticker, 0.0) - old.get(ticker, 0.0)) for ticker in set(old) | set(new_map))
            equity *= 1.0 - turn * FEE
            curve[-1] = equity
            held = new
            year_base = equity
    eq = pd.Series(curve, index=window)
    bh = bench_px.reindex(eq.index).ffill()
    bh = bh / bh.iloc[0]
    return {"equity": eq, "bench": bh, "years": rows, "metrics": _metrics(eq), "bench_metrics": _metrics(bh)}


def holding_year_return(closes: dict[str, pd.Series], picks: list[dict], ends: dict[int, pd.Timestamp], year: int) -> float | None:
    """Constant-mix return of an already chosen book over one calendar year.

    The year-end turnover fee is not included. The decade CAGR charges that fee separately.
    """
    start = ends.get(year - 1)
    finish = ends.get(year)
    if start is None or finish is None or not picks:
        return None
    if any(item["ticker"] not in closes for item in picks):
        return None
    frame = pd.DataFrame({item["ticker"]: closes[item["ticker"]] for item in picks})
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    window = frame.sort_index().loc[start:finish].ffill().dropna(how="any")
    if len(window) < 2:
        return None
    daily = window.pct_change().fillna(0.0)
    equity = 1.0
    for day in daily.index[1:]:
        ret = 0.0
        for item in picks:
            ret += item["weight"] * float(daily.at[day, item["ticker"]])
        equity *= 1.0 + ret
    return equity - 1.0


def current_hold(closes: dict[str, pd.Series], names: list[str], bench: str, top_n: int, weights: tuple[float, ...]) -> dict:
    bench_px = closes[bench].dropna().sort_index()
    bench_px.index = pd.DatetimeIndex(bench_px.index).normalize()
    ends = year_end_dates(bench_px.index)
    latest = bench_px.index[-1]
    completed = [year for year, day in ends.items() if day.month == 12 and day <= latest]
    signal_year = max(completed)
    picks = choose(prior_returns(closes, names, ends, signal_year), top_n, weights)
    next_year = signal_year + 1
    return {
        "signal_year": signal_year,
        "hold_until": f"{next_year}年最后一个交易日收盘",
        "picks": picks,
        "asof": str(latest.date()),
    }


def save_archive(path: Path, strategies: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in strategies:
        for year in item["years"]:
            names = " / ".join(f"{pick['ticker']} {pick['weight']*100:.0f}%" for pick in year["picks"])
            rows.append({"strategy": item["name"], "year": year["year"], "picks": names, "return": year["return"]})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
