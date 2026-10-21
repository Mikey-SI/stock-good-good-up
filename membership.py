"""Who was in QQQ or the S&P 500 for an entire calendar year.

Current names come from the caller. Additions and removals come from Wikipedia,
with the committed CSV used if Wikipedia is unavailable.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "archive" / "membership"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def _norm(ticker: str) -> str:
    return str(ticker).strip().upper().replace(".", "-").replace("\xa0", "")


def _flatten(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [" ".join(str(part) for part in col if str(part) not in {"nan", ""}).strip() for col in frame.columns]
    else:
        frame.columns = [str(col).strip() for col in frame.columns]
    return frame


def _changes_from_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for table in tables:
        frame = _flatten(table)
        cols = list(frame.columns)
        lower = [col.lower() for col in cols]
        if not any("date" in col for col in lower):
            continue
        if not any("added" in col for col in lower) or not any("removed" in col for col in lower):
            continue
        date_col = cols[next(i for i, col in enumerate(lower) if "date" in col)]
        added_cols = [col for col in cols if "added" in col.lower() and ("ticker" in col.lower() or "symbol" in col.lower())]
        removed_cols = [col for col in cols if "removed" in col.lower() and ("ticker" in col.lower() or "symbol" in col.lower())]
        if not added_cols:
            added_cols = [col for col in cols if "added" in col.lower()]
        if not removed_cols:
            removed_cols = [col for col in cols if "removed" in col.lower()]
        for _, record in frame.iterrows():
            day = pd.to_datetime(str(record[date_col]), errors="coerce")
            if pd.isna(day):
                continue
            added = _norm(record[added_cols[0]]) if pd.notna(record[added_cols[0]]) and str(record[added_cols[0]]).strip() else ""
            removed = _norm(record[removed_cols[0]]) if pd.notna(record[removed_cols[0]]) and str(record[removed_cols[0]]).strip() else ""
            if added or removed:
                rows.append({"date": day, "added": added, "removed": removed})
        if rows:
            break
    if not rows:
        raise RuntimeError("no membership changes")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_changes(kind: str) -> pd.DataFrame:
    url = {
        "QQQ": "https://en.wikipedia.org/wiki/Nasdaq-100",
        "SPY": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    }[kind]
    cache = CACHE / ("nasdaq100_changes.csv" if kind == "QQQ" else "sp500_changes.csv")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        html = urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "ignore")
        changes = _changes_from_tables(pd.read_html(io.StringIO(html)))
        cache.parent.mkdir(parents=True, exist_ok=True)
        changes.to_csv(cache, index=False)
        return changes
    except Exception:
        if not cache.exists():
            raise
        return pd.read_csv(cache, parse_dates=["date"])


class PITMembership:
    def __init__(self, current: list[str], changes: pd.DataFrame):
        self.current = {_norm(ticker) for ticker in current if ticker}
        ordered = changes.sort_values("date", ascending=False)
        self._records = [
            (pd.Timestamp(row["date"]), str(row["added"] or ""), str(row["removed"] or ""))
            for _, row in ordered.iterrows()
        ]

    def members_on(self, day: pd.Timestamp) -> set[str]:
        names = set(self.current)
        for when, added, removed in self._records:
            if when > day:
                if added:
                    names.discard(added)
                if removed:
                    names.add(removed)
            else:
                break
        return {ticker for ticker in names if ticker}


def full_year_members(current: list[str], changes: pd.DataFrame, trading_index, year: int) -> list[str]:
    days = pd.DatetimeIndex(pd.to_datetime(trading_index)).normalize().unique().sort_values()
    days = days[days.year == year]
    if len(days) == 0:
        return []
    first = pd.Timestamp(days[0])
    last = pd.Timestamp(days[-1])
    pit = PITMembership(current, changes)
    stayed = pit.members_on(first) & pit.members_on(last)
    changed = set()
    for when, added, removed in pit._records:
        if when <= first or when > last:
            continue
        if added:
            changed.add(added)
        if removed:
            changed.add(removed)
    return sorted(ticker for ticker in stayed if ticker and ticker not in changed)
