"""Rebuild the public monthly-picks page.

Same rules as the top of the local dashboard:
  monthly rebalance on a month-end that is itself a trading day,
  126-day momentum, top 5, benchmark above its 200-day average,
  otherwise the strongest of GLD / TLT / IEF / BIL.
"""
from __future__ import annotations

import html
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DOCS = HERE / "docs"
DOCS.mkdir(exist_ok=True)

LOOKBACK = 126
SKIP = 21
SMA = 200
TOP_N = 5
DEFENSIVE = ["GLD", "TLT", "IEF", "BIL"]
DEF_NAME = {"GLD": "黄金", "TLT": "长期国债", "IEF": "中期国债", "BIL": "超短债"}
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _get(url: str, referer: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": UA, "Referer": referer, "Accept": "application/json,text/html"},
    )
    with urllib.request.urlopen(req, timeout=40) as resp:
        return resp.read()


def wiki_table(url: str) -> list[pd.DataFrame]:
    raw = _get(url, "https://en.wikipedia.org/").decode("utf-8", "ignore")
    return pd.read_html(io.StringIO(raw))


def nasdaq100() -> list[str]:
    raw = _get(
        "https://api.nasdaq.com/api/quote/list-type/nasdaq100",
        "https://www.nasdaq.com/",
    )
    payload = json.loads(raw.decode("utf-8"))
    rows = ((payload.get("data") or {}).get("data") or {}).get("rows") or []
    symbols = [str(row.get("symbol", "")).strip().replace(".", "-") for row in rows]
    symbols = sorted(set(s for s in symbols if s and s.lower() != "nan"))
    if len(symbols) < 80:
        raise RuntimeError("nasdaq-100 list is short")
    return symbols


def sp500() -> list[str]:
    tables = wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    table = tables[0]
    col = "Symbol" if "Symbol" in table.columns else table.columns[0]
    symbols = table[col].astype(str).str.strip().str.replace(".", "-", regex=False)
    symbols = sorted(set(s for s in symbols if s and s.lower() != "nan"))
    if len(symbols) < 400:
        raise RuntimeError("sp500 table looks short")
    return symbols


def sina_symbol(ticker: str) -> str:
    return ticker.replace("-", ".")


def fetch_sina(ticker: str) -> pd.Series:
    symbol = sina_symbol(ticker)
    url = (
        "https://stock.finance.sina.com.cn/usstock/api/jsonp.php/"
        f"var%20_{symbol}=/US_MinKService.getDailyK?symbol={symbol}"
    )
    raw = _get(url, "https://stock.finance.sina.com.cn/").decode("utf-8", "ignore")
    start = raw.find("([")
    end = raw.rfind("])")
    if start < 0 or end < 0:
        raise RuntimeError("empty")
    rows = json.loads(raw[start + 1 : end + 1])
    if not rows:
        raise RuntimeError("empty")
    series = pd.Series({row["d"]: float(row["c"]) for row in rows}, dtype=float)
    series.index = pd.to_datetime(series.index).normalize()
    return series[~series.index.duplicated(keep="last")].sort_index()


def fetch_yf(tickers: list[str]) -> dict[str, pd.Series]:
    import yfinance as yf

    out: dict[str, pd.Series] = {}
    for start in range(0, len(tickers), 80):
        chunk = tickers[start : start + 80]
        raw = yf.download(
            chunk,
            period="2y",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            names = raw.columns.get_level_values(0)
            for ticker in chunk:
                if ticker not in names:
                    continue
                close = raw[ticker]["Close"] if "Close" in raw[ticker] else None
                if close is None:
                    continue
                series = pd.to_numeric(close, errors="coerce").dropna()
                series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
                series = series[~series.index.duplicated(keep="last")].sort_index()
                if len(series) > SMA + 5:
                    out[ticker] = series
        else:
            ticker = chunk[0]
            close = raw["Close"]
            series = pd.to_numeric(close, errors="coerce").dropna()
            series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
            series = series[~series.index.duplicated(keep="last")].sort_index()
            if len(series) > SMA + 5:
                out[ticker] = series
        if start == 0 and len(out) < 5:
            print("  yfinance looks blocked, using sina", flush=True)
            return out
        print(f"  yfinance {start + len(chunk)}/{len(tickers)} kept={len(out)}", flush=True)
    return out


def fetch_many_sina(tickers: list[str]) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}

    def one(ticker: str):
        last = None
        for attempt in range(3):
            try:
                series = fetch_sina(ticker)
                if len(series) > SMA + 5:
                    return ticker, series
            except Exception as exc:
                last = exc
                time.sleep(0.3 * (attempt + 1))
        return ticker, last

    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(one, ticker) for ticker in tickers]
        for fut in as_completed(futs):
            ticker, result = fut.result()
            done += 1
            if isinstance(result, pd.Series):
                out[ticker] = result
            if done % 40 == 0 or done == len(tickers):
                print(f"  sina {done}/{len(tickers)} kept={len(out)}", flush=True)
    return out


def load_prices(tickers: list[str]) -> dict[str, pd.Series]:
    got: dict[str, pd.Series] = {}
    try:
        got = fetch_yf(tickers)
    except Exception as exc:
        print(f"  yfinance failed: {exc}", flush=True)
    missing = [ticker for ticker in tickers if ticker not in got]
    if len(got) < max(20, len(tickers) // 5):
        missing = tickers
        got = {}
    if missing:
        print(f"  filling {len(missing)} names from sina", flush=True)
        got.update(fetch_many_sina(missing))
    return got


def month_ends(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    dummy = pd.Series(1.0, index=index)
    ends = pd.DatetimeIndex(dummy.resample("ME").last().index).normalize()
    have = set(pd.DatetimeIndex(index).normalize())
    return [pd.Timestamp(day) for day in ends if day in have]


def next_rebalance(last_done: pd.Timestamp) -> pd.Timestamp:
    cursor = last_done
    for _ in range(8):
        cursor = pd.Timestamp(cursor + pd.offsets.MonthEnd(1)).normalize()
        if cursor.weekday() < 5:
            return cursor
    return cursor


def signal_on(closes: dict[str, pd.Series], stocks: list[str], bench: str, day: pd.Timestamp) -> dict:
    history = closes[bench].loc[:day].dropna()
    used = history.index[-1]
    sma = float(history.rolling(SMA).mean().iloc[-1])
    price = float(history.iloc[-1])
    bull = bool(price > sma)
    frame = pd.DataFrame({ticker: closes[ticker] for ticker in stocks if ticker in closes})
    frame = frame.reindex(history.index)
    mom = frame.shift(SKIP).pct_change(max(LOOKBACK - SKIP, 1))
    defensive = pd.DataFrame({name: closes[name] for name in DEFENSIVE if name in closes}).reindex(history.index)
    dmom = defensive.pct_change(LOOKBACK)
    best = dmom.iloc[-1].dropna()
    best_name = str(best.idxmax()) if len(best) else "BIL"
    best_score = float(best.max()) if len(best) else 0.0
    if bull:
        scores = mom.iloc[-1].dropna()
        scores = scores[scores > 0]
        live = frame.iloc[-1]
        scores = scores[live.reindex(scores.index).notna()]
        scores = scores.sort_values(ascending=False).head(TOP_N)
        if len(scores):
            rows = [
                {"ticker": str(ticker), "name": "", "weight": 1.0 / len(scores), "momentum": float(score)}
                for ticker, score in scores.items()
            ]
            mode = "进攻"
        else:
            rows = [{"ticker": best_name, "name": DEF_NAME.get(best_name, ""), "weight": 1.0, "momentum": best_score}]
            mode = "防守"
    else:
        rows = [{"ticker": best_name, "name": DEF_NAME.get(best_name, ""), "weight": 1.0, "momentum": best_score}]
        mode = "防守"
    return {
        "mode": mode,
        "on": str(used.date()),
        "bull": bull,
        "bench": bench,
        "bench_px": round(price, 2),
        "sma": round(sma, 2),
        "rows": rows,
    }


def book(title: str, bench: str, names: list[str], closes: dict[str, pd.Series], held_day: pd.Timestamp, preview_day: pd.Timestamp) -> dict:
    stocks = [ticker for ticker in names if ticker in closes and ticker not in DEFENSIVE and ticker != bench and len(closes[ticker]) >= LOOKBACK + SKIP + 5]
    return {
        "title": title,
        "n": len(stocks),
        "held": signal_on(closes, stocks, bench, held_day),
        "preview": signal_on(closes, stocks, bench, preview_day),
    }


def cn_date(day: pd.Timestamp) -> str:
    return f"{int(day.year)}年{int(day.month)}月{int(day.day)}日"


def esc(value: str) -> str:
    return html.escape(str(value), quote=True)


def row_html(rows: list[dict]) -> str:
    body = []
    for row in rows:
        label = esc(row["ticker"])
        if row.get("name"):
            label += "  " + esc(row["name"])
        mom = row.get("momentum")
        mom_text = "-" if mom is None else f"{mom * 100:.1f}%"
        body.append(
            f"<tr><td>{label}</td><td>{round(row['weight'] * 100)}%</td><td>{mom_text}</td></tr>"
        )
    return "\n".join(body)


def note(held: dict) -> str:
    side = "在 200 日均线上方，这一档买股票" if held["bull"] else "在 200 日均线下方，这一档全部换成防守"
    return f"{held['on']} 收盘  {held['bench']} {held['bench_px']:.2f}，200日均线 {held['sma']:.2f}，{side}。"


def preview_line(preview: dict) -> str:
    names = "、".join(row["ticker"] + (row["name"] or "") for row in preview["rows"])
    return f"下一次预览（{preview['on']}，{preview['mode']}）：{names}。等下次调仓日收盘才换成这一组。"


def render(payload: dict) -> str:
    lines = "".join(f"<p class=\"lead\">{esc(line)}</p>" for line in payload["lines"])
    qqq = payload["qqq"]
    sp = payload["sp"]
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>本月选股</title>
  <style>
    :root {{ color-scheme: light; --bg:#f4f1ea; --ink:#1c1b19; --muted:#6f6a62; --line:#ddd6c8; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family: "Palatino Linotype", Palatino, "Songti SC", "SimSun", serif; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 28px 22px 64px; }}
    header {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-end; border-bottom:1px solid var(--ink); padding-bottom:14px; }}
    h1 {{ font-size:34px; font-weight:500; margin:0; letter-spacing:0.04em; }}
    .meta {{ color:var(--muted); font-family: Consolas, "Cascadia Mono", monospace; font-size:12px; text-align:right; }}
    .lead {{ font-size:17px; line-height:1.75; margin:0 0 8px; }}
    .cols {{ display:grid; grid-template-columns: 1fr 1fr; gap: 28px; margin-top: 8px; }}
    h2 {{ font-size:18px; font-weight:500; margin:18px 0 8px; }}
    .note {{ color:var(--muted); font-size:12px; margin:0 0 8px; line-height:1.6; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th {{ text-align:left; font-weight:500; color:var(--muted); border-bottom:1px solid var(--ink); padding:6px 4px; }}
    td {{ border-bottom:1px solid var(--line); padding:6px 4px; font-family: Consolas, "Cascadia Mono", monospace; }}
    @media (max-width: 860px) {{ .cols {{ grid-template-columns: 1fr; }} header {{ display:block; }} .meta {{ text-align:left; margin-top:8px; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>本月选股</h1>
      <div class="meta">行情 {esc(payload["asof"])}<br>每月 1 日自动更新</div>
    </header>
    <section>
      <h2>这个月怎么操作</h2>
      {lines}
      <div class="cols">
        <div>
          <h2>纳指100，看 QQQ</h2>
          <p class="note">{esc(note(qqq["held"]))}</p>
          <table>
            <thead><tr><th>现在应该拿着</th><th>仓位</th><th>动量</th></tr></thead>
            <tbody>{row_html(qqq["held"]["rows"])}</tbody>
          </table>
          <p class="note">{esc(preview_line(qqq["preview"]))}</p>
        </div>
        <div>
          <h2>标普500，看 SPY</h2>
          <p class="note">{esc(note(sp["held"]))}</p>
          <table>
            <thead><tr><th>现在应该拿着</th><th>仓位</th><th>动量</th></tr></thead>
            <tbody>{row_html(sp["held"]["rows"])}</tbody>
          </table>
          <p class="note">{esc(preview_line(sp["preview"]))}</p>
        </div>
      </div>
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    print("universes", flush=True)
    nq = nasdaq100()
    sp = sp500()
    tickers = list(dict.fromkeys(nq + sp + DEFENSIVE + ["QQQ", "SPY"]))
    print(f"  nasdaq={len(nq)} sp500={len(sp)} download={len(tickers)}", flush=True)
    closes = load_prices(tickers)
    if "QQQ" not in closes or "SPY" not in closes:
        raise RuntimeError("QQQ or SPY missing")
    for name in DEFENSIVE:
        if name not in closes:
            raise RuntimeError(f"missing {name}")
    latest = min(closes["QQQ"].index[-1], closes["SPY"].index[-1])
    ends = [day for day in month_ends(closes["QQQ"].index) if day <= latest]
    if not ends:
        raise RuntimeError("no month-end")
    held_day = ends[-1]
    nxt = next_rebalance(held_day)
    sleeve = held_day + pd.offsets.MonthBegin(1)
    preview_day = pd.Timestamp(latest).normalize()
    qqq = book("纳指100", "QQQ", nq, closes, held_day, preview_day)
    sp_book = book("标普500", "SPY", sp, closes, held_day, preview_day)
    sleeve_label = f"{int(sleeve.year)}年{int(sleeve.month)}月"
    lines = [
        "每个月只调一次仓：美股该月最后一个交易日收盘后换。月中保持不动。",
        f"现在拿的是{sleeve_label}这一档。它在 {cn_date(held_day)} 美股收盘选出，拿到 {cn_date(nxt)} 美股收盘。",
        f"下一次调仓是 {cn_date(nxt)} 美股收盘之后，北京时间大约是第二天早上。在那之前，按「现在应该拿着」买。",
        f"行情用到 {preview_day.date()}。下一次预览要等 {cn_date(nxt)} 的收盘价出来才生效。",
    ]
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    payload = {
        "asof": str(preview_day.date()),
        "generated_at": now,
        "held_on": str(held_day.date()),
        "next_on": str(nxt.date()),
        "sleeve": sleeve_label,
        "lines": lines,
        "qqq": qqq,
        "sp": sp_book,
    }
    (DOCS / "picks.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (DOCS / "index.html").write_text(render(payload), encoding="utf-8")
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    print(f"held {held_day.date()} next {nxt.date()} asof {preview_day.date()}", flush=True)
    for key in ("qqq", "sp"):
        rows = payload[key]["held"]["rows"]
        print(key, payload[key]["held"]["mode"], [row["ticker"] for row in rows], flush=True)
    print("wrote", DOCS / "index.html", flush=True)


if __name__ == "__main__":
    main()
