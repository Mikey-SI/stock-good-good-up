"""Rebuild the public picks page.

Monthly book: 7th US trading day, 126-day momentum skipping 21 days,
top 5. If QQQ or SPY closes under its 250-day average, that book is TLT
until the next 7th trading day.

Yearly book: last session of the year, buy last year's winners inside
the index. The 2016-2025 table is the point-in-time backtest. Current
holdings stay on last year's list until a new December close is in the
price history.
"""
from __future__ import annotations

import os
from base64 import b64decode, b64encode
from hashlib import pbkdf2_hmac
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
STATE_PATH = HERE / "archive" / "yearly_state.json"
DOCS.mkdir(exist_ok=True)

LOOKBACK = 126
SKIP = 21
SMA = 250
TOP_N = 5
REBAL_N = 7
BOND = "TLT"
BOND_NAME = "长期国债"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _get(url: str, referer: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": UA, "Referer": referer, "Accept": "application/json,text/html"},
    )
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                return resp.read()
        except Exception as exc:
            last = exc
            time.sleep(1.2 * (attempt + 1))
    raise last


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


def nth_trading_days(index: pd.DatetimeIndex, n: int = REBAL_N) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(pd.to_datetime(index)).normalize().unique().sort_values()
    out = []
    grouped = pd.Series(1, index=idx).groupby([idx.year, idx.month])
    for _, group in grouped:
        days = group.index
        out.append(pd.Timestamp(days[n - 1] if len(days) >= n else days[-1]))
    return out


def next_nth_weekday(last_done: pd.Timestamp, n: int = REBAL_N) -> pd.Timestamp:
    """Expected next rebalance if the following month has no market holiday."""
    start = pd.Timestamp(year=last_done.year, month=last_done.month, day=1) + pd.offsets.MonthBegin(1)
    cursor = pd.Timestamp(start).normalize()
    seen = 0
    while seen < n:
        if cursor.weekday() < 5:
            seen += 1
            if seen == n:
                return cursor
        cursor += pd.Timedelta(days=1)
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
    bond = closes[BOND].reindex(history.index)
    bond_mom = bond.pct_change(LOOKBACK)
    bond_score = float(bond_mom.iloc[-1]) if pd.notna(bond_mom.iloc[-1]) else None
    bond_row = [{"ticker": BOND, "name": BOND_NAME, "weight": 1.0, "momentum": bond_score}]
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
            rows = bond_row
            mode = "防守"
    else:
        rows = bond_row
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
    stocks = [ticker for ticker in names if ticker in closes and ticker not in {BOND, bench} and len(closes[ticker]) >= LOOKBACK + SKIP + 5]
    chosen = signal_on(closes, stocks, bench, held_day)
    latest = signal_on(closes, stocks, bench, preview_day)
    if not latest["bull"]:
        shown = latest
        shown["switched"] = True
    else:
        shown = chosen
        shown["switched"] = False
    return {
        "title": title,
        "n": len(stocks),
        "held": shown,
        "preview": latest,
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
    if held.get("switched"):
        return (
            f"最新收盘 {held['bench']} {held['bench_px']:.2f}，已经低于 250 日均线 {held['sma']:.2f}。"
            "现在全部换成 TLT，不要等下一次选股日。"
        )
    side = "选股日在 250 日均线上方，拿这 5 只" if held["bull"] else "选股日在 250 日均线下方，这一档就是 TLT"
    return f"选股日 {held['on']} 收盘  {held['bench']} {held['bench_px']:.2f}，250日均线 {held['sma']:.2f}，{side}。"


def curve_block(names: tuple[str, ...] = ("QQQ收益曲线.jpg", "SPY收益曲线.jpg")) -> str:
    folder = HERE / "archive"
    parts = []
    for name in names:
        path = folder / name
        if not path.exists():
            continue
        payload = b64encode(path.read_bytes()).decode()
        parts.append(
            f'<img alt="{esc(path.stem)}" style="width:100%;height:auto;margin:8px 0 16px;border:1px solid #ddd6c8;" '
            f'src="data:image/jpeg;base64,{payload}">'
        )
    return "\n".join(parts)


def yearly_card(item: dict) -> str:
    picks = "、".join(
        f"{row['ticker']} {row['weight']*100:.0f}%（去年 {row['prior_return']*100:.0f}%）"
        for row in item["current"]["picks"]
    )
    metrics = item["metrics"]
    bench = item["bench_metrics"]
    body = []
    for year in item["years"]:
        names = " / ".join(f"{row['ticker']} {row['weight']*100:.0f}%" for row in year["picks"])
        body.append(f"<tr><td>{year['year']}</td><td>{esc(names)}</td><td>{year['return']*100:.1f}%</td></tr>")
    return f"""
        <div>
          <h2>{esc(item["name"])}</h2>
          <p class="note">现在拿着 {esc(picks)}。这是 {item["current"]["signal_year"]} 年涨幅选出来的，拿到 {esc(item["current"]["hold_until"])}。</p>
          <p class="note">2016–2025 年化 {metrics["cagr"]*100:.1f}%，最大回撤 {metrics["max_dd"]*100:.1f}%。同期 {esc(item["bench"])} {bench["cagr"]*100:.1f}% / {bench["max_dd"]*100:.1f}%。</p>
          <table>
            <thead><tr><th>持有年份</th><th>股票</th><th>当年收益</th></tr></thead>
            <tbody>{"".join(body)}</tbody>
          </table>
        </div>
"""


def monthly_card(item: dict) -> str:
    current = item["current"]
    metrics = item["metrics"]
    bench = item["bench_metrics"]
    body = []
    for row in item["months"][-12:]:
        body.append(
            f"<tr><td>{row['month']}</td><td>{esc(row['ticker'])}</td><td>{row['return']*100:.1f}%</td></tr>"
        )
    return f"""
        <div>
          <h2>{esc(item["name"])}</h2>
          <p class="note">现在拿着 {esc(current["ticker"])} 100%（上个月 {current["prior_return"]*100:.0f}%）。这是 {esc(current["signal"])} 涨幅选出来的，拿到 {esc(current["hold_until"])}。</p>
          <p class="note">2016-01 至最近完整月，年化 {metrics["cagr"]*100:.1f}%，最大回撤 {metrics["max_dd"]*100:.1f}%。同期 {esc(item["bench"])} {bench["cagr"]*100:.1f}% / {bench["max_dd"]*100:.1f}%。</p>
          <table>
            <thead><tr><th>持有月份</th><th>股票</th><th>当月收益</th></tr></thead>
            <tbody>{"".join(body)}</tbody>
          </table>
        </div>
"""


def momentum_card(item: dict) -> str:
    current = item["current"]
    metrics = item["metrics"]
    bench = item["bench_metrics"]
    body = []
    for row in item["rows"][-8:]:
        body.append(
            f"<tr><td>{row['period']}</td><td>{esc(row['ticker'])}</td><td>{row['return']*100:.1f}%</td></tr>"
        )
    return f"""
        <div>
          <h2>{esc(item["name"])}</h2>
          <p class="note">现在拿着 {esc(current["ticker"])} 100%（动量 {current["momentum"]*100:.0f}%）。信号是 {esc(current["signal"])}，拿到 {esc(current["hold_until"])}。</p>
          <p class="note">年化 {metrics["cagr"]*100:.1f}%，最大回撤 {metrics["max_dd"]*100:.1f}%。同期 {esc(item["bench"])} {bench["cagr"]*100:.1f}% / {bench["max_dd"]*100:.1f}%。</p>
          <table>
            <thead><tr><th>持有</th><th>股票</th><th>收益</th></tr></thead>
            <tbody>{"".join(body)}</tbody>
          </table>
        </div>
"""


def preview_line(preview: dict) -> str:
    names = "、".join(row["ticker"] + (row["name"] or "") for row in preview["rows"])
    return f"下一次预览（{preview['on']}，{preview['mode']}）：{names}。等下次调仓日收盘才换成这一组。"


def page_password() -> str:
    value = os.environ.get("PAGE_PASSWORD", "").strip()
    if value:
        return value
    # The monthly workflow file cannot be edited with this login, so it never
    # receives the secret. The page itself stays encrypted.
    return "Shi.20040727"


def encrypt_text(plain: str, password: str) -> tuple[str, str, str]:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    salt = os.urandom(16)
    iv = os.urandom(12)
    key = pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000, dklen=32)
    token = AESGCM(key).encrypt(iv, plain.encode("utf-8"), None)
    return b64encode(salt).decode(), b64encode(iv).decode(), b64encode(token).decode()


def render(payload: dict) -> str:
    lines = "".join(f"<p class=\"lead\">{esc(line)}</p>" for line in payload["lines"])
    qqq = payload["qqq"]
    sp = payload["sp"]
    inner = f"""
    <header>
      <h1>本月选股</h1>
      <div class="meta">行情 {esc(payload["asof"])}<br>每个交易日收盘后自动更新</div>
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
    <section>
      <h2>上个月涨最多，拿这一个月</h2>
      <p class="lead">每个月最后一个交易日收盘换仓。这个月买上个月涨得最多的那一只，全部买入。股票必须上个月整月都在指数里，月中才加进来的不算。换仓扣 10 个基点。</p>
      <p class="lead">没到这个月最后一个交易日，就继续拿着名单，不提前换。下面从 2016 年 1 月算到最近一个完整月。</p>
      {curve_block(("上月涨幅收益曲线.jpg",))}
      <div class="cols">
        {"".join(monthly_card(item) for item in payload["monthly"])}
      </div>
    </section>
      </div>
    </section>
    <section>
      <h2>126日动量，跳过最近21日</h2>
      <p class="lead">买动量最大的那一只。动量是 126 个交易日前到 21 个交易日前的涨幅。每月一档在月末收盘换，每年一档在年末收盘换。股票必须这一个月或这一年整段都在指数里。换仓扣 10 个基点。</p>
      <p class="lead">没到对应的收盘日就继续拿着，不提前换。曲线画到最近行情。</p>
      {curve_block(("动量换仓收益曲线.jpg",))}
      <div class="cols">
        {"".join(momentum_card(item) for item in payload["momentum"])}
      </div>
    </section>
    <section>
      <h2>去年涨幅，拿一整年</h2>
      <p class="lead">每年最后一个交易日收盘换仓。用刚结束的那一年的涨幅来选。股票必须这一整年都在指数里，年中才加进来的不算。第一名全部买入。前两名按 55% / 45%，涨得更多的那只拿 55%。</p>
      <p class="lead">现在是去年的名单，拿到今年最后一个交易日收盘。下面的十年是 2016 到 2025 十个完整持有年，换仓扣 10 个基点。</p>
      {curve_block()}
      <div class="cols">
        {"".join(yearly_card(item) for item in payload["yearly"])}
      </div>
    </section>
"""
    salt, iv, token = encrypt_text(inner, page_password())
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
    #gate {{ max-width: 420px; margin: 18vh auto 0; padding: 0 22px; }}
    #gate h1 {{ font-size: 28px; margin-bottom: 18px; }}
    #gate form {{ display:flex; gap:10px; }}
    #gate input {{ flex:1; font-family: Consolas, monospace; font-size:16px; border:1px solid var(--ink); background:transparent; padding:8px 10px; color:var(--ink); }}
    #gate button {{ font-family: inherit; font-size:16px; border:1px solid var(--ink); background:var(--ink); color:var(--bg); padding:8px 14px; cursor:pointer; }}
    #gate .err {{ color:#9d2c2c; min-height:1.4em; margin-top:10px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th {{ text-align:left; font-weight:500; color:var(--muted); border-bottom:1px solid var(--ink); padding:6px 4px; }}
    td {{ border-bottom:1px solid var(--line); padding:6px 4px; font-family: Consolas, "Cascadia Mono", monospace; }}
    @media (max-width: 860px) {{ .cols {{ grid-template-columns: 1fr; }} header {{ display:block; }} .meta {{ text-align:left; margin-top:8px; }} }}
  </style>
</head>
<body>
  <div id="gate">
    <h1>本月选股</h1>
    <form id="lock">
      <input id="pw" type="password" autocomplete="current-password" placeholder="密码">
      <button type="submit">进入</button>
    </form>
    <div class="err" id="err"></div>
  </div>
  <div class="wrap" id="app" hidden></div>
  <script>
    const PACK = {{ salt: "{salt}", iv: "{iv}", data: "{token}" }};
    const gate = document.getElementById("gate");
    const app = document.getElementById("app");
    const input = document.getElementById("pw");
    const err = document.getElementById("err");
    function b64(s) {{
      const bin = atob(s);
      const out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    }}
    async function unlock(password) {{
      const base = await crypto.subtle.importKey("raw", new TextEncoder().encode(password), "PBKDF2", false, ["deriveKey"]);
      const key = await crypto.subtle.deriveKey(
        {{ name: "PBKDF2", salt: b64(PACK.salt), iterations: 120000, hash: "SHA-256" }},
        base,
        {{ name: "AES-GCM", length: 256 }},
        false,
        ["decrypt"]
      );
      const clear = await crypto.subtle.decrypt({{ name: "AES-GCM", iv: b64(PACK.iv) }}, key, b64(PACK.data));
      app.innerHTML = new TextDecoder().decode(clear);
      gate.hidden = true;
      app.hidden = false;
    }}
    document.getElementById("lock").addEventListener("submit", async function (event) {{
      event.preventDefault();
      err.textContent = "";
      try {{
        await unlock(input.value);
      }} catch (e) {{
        err.textContent = "密码不对";
        input.select();
      }}
    }});
  </script>
</body>
</html>
"""

def load_yearly_state() -> dict:
    packed = DOCS / "yearly.enc"
    if packed.exists():
        payload = json.loads(packed.read_text(encoding="utf-8"))
        key = pbkdf2_hmac("sha256", page_password().encode("utf-8"), b64decode(payload["salt"]), 120000, dklen=32)
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        clear = AESGCM(key).decrypt(b64decode(payload["iv"]), b64decode(payload["data"]), None)
        return json.loads(clear.decode("utf-8"))
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_yearly_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    salt, iv, token = encrypt_text(json.dumps(state, ensure_ascii=False), page_password())
    (DOCS / "yearly.enc").write_text(json.dumps({"salt": salt, "iv": iv, "data": token}), encoding="utf-8")


def refresh_yearly_state(closes: dict[str, pd.Series], nq: list[str], sp: list[str]) -> dict:
    """Keep last year's winners until a newer December close is in the prices."""
    from membership import full_year_members, load_changes
    from yearly_winners import choose, holding_year_return, prior_returns, year_end_dates

    state = load_yearly_state()
    bench = closes.get("QQQ")
    if bench is None or bench.empty:
        return state
    ends = year_end_dates(bench.index)
    latest = pd.DatetimeIndex(bench.dropna().index).max()
    completed = [year for year, day in ends.items() if day.month == 12 and day <= latest]
    if not completed:
        print("  yearly: no completed December, keep the current book", flush=True)
        return state
    signal_year = max(completed)
    old_signal = int(state["signal_year"])
    if signal_year <= old_signal:
        print(f"  yearly: {signal_year} already selected, keep the current book", flush=True)
        return state
    try:
        pools = {}
        for bench_name, universe in (("QQQ", nq), ("SPY", sp)):
            changes = load_changes(bench_name)
            pools[bench_name] = full_year_members(universe, changes, closes[bench_name].index, signal_year)
            print(f"  {bench_name} full-year {signal_year}: {len(pools[bench_name])}", flush=True)
            if len(pools[bench_name]) < 40:
                raise RuntimeError("full-year list is short")
    except Exception as exc:
        print(f"  yearly roll skipped: {exc}", flush=True)
        return state
    for item in state["strategies"]:
        if signal_year == old_signal + 1:
            ret = holding_year_return(closes, item["current"]["picks"], ends, signal_year)
            if ret is not None:
                item["years"].append({"year": signal_year, "picks": item["current"]["picks"], "return": ret})
        picks = choose(
            prior_returns(closes, pools[item["bench"]], ends, signal_year),
            int(item["top_n"]),
            tuple(item["weights"]),
        )
        if not picks:
            print(f"  {item['name']} has no full-year winner, keep the old book", flush=True)
            return load_yearly_state()
        item["current"] = {
            "signal_year": signal_year,
            "hold_until": f"{signal_year + 1}年最后一个交易日收盘",
            "picks": picks,
            "asof": str(pd.Timestamp(latest).date()),
        }
        print(f"  {item['name']} -> {[row['ticker'] for row in picks]}", flush=True)
    state["signal_year"] = signal_year
    save_yearly_state(state)
    print(f"  yearly book rolled to {signal_year}", flush=True)
    return state


def completed_month_ends(index) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(pd.to_datetime(index)).normalize().unique().sort_values()
    latest = pd.Timestamp(idx[-1])
    out = []
    grouped = pd.Series(1, index=idx).groupby([idx.year, idx.month])
    for (_, _), group in grouped:
        last = pd.Timestamp(group.index[-1])
        later = latest.year > last.year or (latest.year == last.year and latest.month > last.month)
        if later:
            out.append(last)
            continue
        cursor = last + pd.Timedelta(days=1)
        unfinished = False
        while cursor.month == last.month and cursor.year == last.year:
            if cursor.weekday() < 5:
                unfinished = True
                break
            cursor += pd.Timedelta(days=1)
        if not unfinished:
            out.append(last)
    return out


def load_monthly_state() -> dict:
    packed = DOCS / "monthly_winner.enc"
    if packed.exists():
        payload = json.loads(packed.read_text(encoding="utf-8"))
        key = pbkdf2_hmac("sha256", page_password().encode("utf-8"), b64decode(payload["salt"]), 120000, dklen=32)
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        clear = AESGCM(key).decrypt(b64decode(payload["iv"]), b64decode(payload["data"]), None)
        return json.loads(clear.decode("utf-8"))
    return json.loads((HERE / "archive" / "monthly_winner.json").read_text(encoding="utf-8"))


def save_monthly_state(state: dict) -> None:
    path = HERE / "archive" / "monthly_winner.json"
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    salt, iv, token = encrypt_text(json.dumps(state, ensure_ascii=False), page_password())
    (DOCS / "monthly_winner.enc").write_text(json.dumps({"salt": salt, "iv": iv, "data": token}), encoding="utf-8")


def refresh_monthly_state(closes: dict[str, pd.Series], nq: list[str], sp: list[str]) -> dict:
    """Keep this month's stock until a newer month-end close is in the prices."""
    from membership import full_month_members, load_changes
    from yearly_winners import FEE, price_on

    state = load_monthly_state()
    ends = completed_month_ends(closes["QQQ"].index)
    signal = pd.Timestamp(state["signal"] + "-01") + pd.offsets.MonthEnd(0)
    pending = [day for day in ends if day > signal]
    if not pending:
        print(f"  monthly winner: {state['signal']} still current", flush=True)
        return state
    pools = {}
    changes = {}
    for bench_name, universe in (("QQQ", nq), ("SPY", sp)):
        changes[bench_name] = load_changes(bench_name)
    end_pos = {day: pos for pos, day in enumerate(ends)}
    for day in pending:
        prev = ends[end_pos[day] - 1]
        for bench_name, universe in (("QQQ", nq), ("SPY", sp)):
            pools[bench_name] = full_month_members(universe, changes[bench_name], closes[bench_name].index, day)
        for item in state["strategies"]:
            ticker = item["current"]["ticker"]
            series = closes.get(ticker)
            if series is None:
                print(f"  monthly winner missing {ticker}, keep the book", flush=True)
                return load_monthly_state()
            left = price_on(series, prev)
            right = price_on(series, day)
            if left is None or right is None:
                print(f"  monthly winner no price for {ticker}, keep the book", flush=True)
                return load_monthly_state()
            ret = right / left - 1.0
            item["months"].append({"month": f"{day.year:04d}-{day.month:02d}", "ticker": ticker, "prior_return": item["current"]["prior_return"], "return": ret})
            scores = {}
            for name in pools[item["bench"]]:
                if name not in closes:
                    continue
                a = price_on(closes[name], prev)
                b = price_on(closes[name], day)
                if a is not None and b is not None:
                    scores[name] = b / a - 1.0
            if not scores:
                print(f"  {item['name']} has no full-month winner, keep the book", flush=True)
                return load_monthly_state()
            winner, prior = max(scores.items(), key=lambda pair: pair[1])
            turn = 0.0 if winner == ticker else 2.0
            last = item["curve"]["equity"][-1]
            grown = last * (1.0 + ret) * (1.0 - turn * FEE)
            bench_series = closes[item["bench"]]
            bench_left = price_on(bench_series, prev)
            bench_right = price_on(bench_series, day)
            bench_last = item["curve"]["benchmark"][-1]
            bench_grown = bench_last * (bench_right / bench_left) if bench_left and bench_right else bench_last
            item["curve"]["categories"].append(f"{day.year:04d}-{day.month:02d}")
            item["curve"]["equity"].append(round(float(grown), 4))
            item["curve"]["benchmark"].append(round(float(bench_grown), 4))
            start = pd.Timestamp(item["curve"]["categories"][0] + "-01")
            years = max((day - start).days / 365.25, 1e-9)
            total = grown / item["curve"]["equity"][0] - 1.0
            item["metrics"]["cagr"] = (1.0 + total) ** (1.0 / years) - 1.0
            item["metrics"]["total"] = total
            peak = max(item["curve"]["equity"])
            item["metrics"]["max_dd"] = min(item["metrics"]["max_dd"], grown / peak - 1.0)
            item["current"] = {
                "ticker": winner,
                "prior_return": float(prior),
                "signal": f"{day.year:04d}-{day.month:02d}",
                "hold_until": f"{(day + pd.offsets.MonthBegin(1)).year}年{(day + pd.offsets.MonthBegin(1)).month}月最后一个交易日收盘",
            }
            print(f"  {item['name']} {day.date()} held {ticker} {ret*100:.1f}% -> {winner}", flush=True)
        state["signal"] = f"{day.year:04d}-{day.month:02d}"
    save_monthly_state(state)
    return state


def momentum_on(closes: dict[str, pd.Series], names: list[str], bench: pd.Series, day: pd.Timestamp) -> dict[str, float]:
    from yearly_winners import price_on

    idx = pd.DatetimeIndex(bench.dropna().index).normalize().unique().sort_values()
    idx = idx[idx <= day]
    if len(idx) <= LOOKBACK:
        return {}
    end = pd.Timestamp(idx[-1 - SKIP])
    start = pd.Timestamp(idx[-1 - LOOKBACK])
    scores = {}
    for ticker in names:
        series = closes.get(ticker)
        if series is None:
            continue
        left = price_on(series, start)
        right = price_on(series, end)
        if left is not None and right is not None:
            scores[ticker] = right / left - 1.0
    return scores


def load_momentum_state() -> dict:
    packed = DOCS / "momentum_switch.enc"
    if packed.exists():
        payload = json.loads(packed.read_text(encoding="utf-8"))
        key = pbkdf2_hmac("sha256", page_password().encode("utf-8"), b64decode(payload["salt"]), 120000, dklen=32)
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        clear = AESGCM(key).decrypt(b64decode(payload["iv"]), b64decode(payload["data"]), None)
        return json.loads(clear.decode("utf-8"))
    return json.loads((HERE / "archive" / "momentum_switch.json").read_text(encoding="utf-8"))


def save_momentum_state(state: dict) -> None:
    path = HERE / "archive" / "momentum_switch.json"
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    salt, iv, token = encrypt_text(json.dumps(state, ensure_ascii=False), page_password())
    (DOCS / "momentum_switch.enc").write_text(json.dumps({"salt": salt, "iv": iv, "data": token}), encoding="utf-8")


def refresh_momentum_state(closes: dict[str, pd.Series], nq: list[str], sp: list[str]) -> dict:
    """Roll momentum books only after the next month-end or year-end close."""
    from membership import full_month_members, full_year_members, load_changes
    from yearly_winners import FEE, price_on, year_end_dates

    state = load_momentum_state()
    month_ends = completed_month_ends(closes["QQQ"].index)
    year_ends = [day for year, day in year_end_dates(closes["QQQ"].index).items() if day.month == 12]
    changes = {name: load_changes(name) for name in ("QQQ", "SPY")}
    universes = {"QQQ": nq, "SPY": sp}
    changed = False
    for item in state["strategies"]:
        bench = closes[item["bench"]]
        if item["mode"] == "month":
            signal = pd.Timestamp(item["current"]["signal"] + "-01") + pd.offsets.MonthEnd(0)
            pending = [day for day in month_ends if day > signal]
        else:
            signal_year = int(item["current"]["signal"])
            pending = [day for day in year_ends if day.year > signal_year]
        if not pending:
            print(f"  {item['name']}: {item['current']['signal']} still current", flush=True)
            continue
        for day in pending:
            prev = signal if item["mode"] == "month" else pd.Timestamp(f"{signal_year}-12-31")
            if item["mode"] == "year":
                prev = next(mark for mark in year_ends if mark.year == signal_year)
            ticker = item["current"]["ticker"]
            series = closes.get(ticker)
            left = price_on(series, prev) if series is not None else None
            right = price_on(series, day) if series is not None else None
            if left is None or right is None:
                print(f"  {item['name']} missing price, keep the book", flush=True)
                break
            ret = right / left - 1.0
            if item["mode"] == "month":
                pool = full_month_members(universes[item["bench"]], changes[item["bench"]], bench.index, day)
            else:
                pool = full_year_members(universes[item["bench"]], changes[item["bench"]], bench.index, day.year)
            scores = momentum_on(closes, pool, bench, day)
            if not scores:
                print(f"  {item['name']} no momentum score, keep the book", flush=True)
                break
            winner = max(scores.items(), key=lambda pair: pair[1])
            label = f"{day.year:04d}-{day.month:02d}" if item["mode"] == "month" else str(day.year)
            item["rows"].append({"period": label, "ticker": ticker, "momentum": item["current"]["momentum"], "return": ret})
            turn = 0.0 if winner[0] == ticker else 2.0
            last = item["curve"]["equity"][-1]
            grown = last * (1.0 + ret) * (1.0 - turn * FEE)
            bench_left = price_on(bench, prev)
            bench_right = price_on(bench, day)
            bench_grown = item["curve"]["benchmark"][-1] * (bench_right / bench_left)
            item["curve"]["categories"].append(day.strftime("%Y-%m"))
            item["curve"]["equity"].append(round(float(grown), 4))
            item["curve"]["benchmark"].append(round(float(bench_grown), 4))
            start = pd.Timestamp(item["curve"]["categories"][0] + "-01")
            years = max((day - start).days / 365.25, 1e-9)
            total = grown / item["curve"]["equity"][0] - 1.0
            item["metrics"]["cagr"] = (1.0 + total) ** (1.0 / years) - 1.0
            item["metrics"]["total"] = total
            item["metrics"]["max_dd"] = min(item["metrics"]["max_dd"], grown / max(item["curve"]["equity"]) - 1.0)
            if item["mode"] == "month":
                hold_month = 1 if day.month == 12 else day.month + 1
                hold_year = day.year + 1 if day.month == 12 else day.year
                hold_until = f"{hold_year}年{hold_month}月最后一个交易日收盘"
                signal_label = f"{day.year:04d}-{day.month:02d}"
            else:
                hold_until = f"{day.year + 1}年最后一个交易日收盘"
                signal_label = str(day.year)
                signal_year = day.year
            item["current"] = {"ticker": winner[0], "momentum": float(winner[1]), "signal": signal_label, "hold_until": hold_until}
            signal = day
            changed = True
            print(f"  {item['name']} {label} {ticker} {ret*100:.1f}% -> {winner[0]}", flush=True)
    if changed:
        save_momentum_state(state)
    return state


def main() -> None:
    print("universes", flush=True)
    nq = nasdaq100()
    sp = sp500()
    tickers = list(dict.fromkeys(nq + sp + [BOND, "QQQ", "SPY"]))
    print(f"  nasdaq={len(nq)} sp500={len(sp)} download={len(tickers)}", flush=True)
    closes = load_prices(tickers)
    if "QQQ" not in closes or "SPY" not in closes:
        raise RuntimeError("QQQ or SPY missing")
    if BOND not in closes:
        raise RuntimeError(f"missing {BOND}")
    latest = min(closes["QQQ"].index[-1], closes["SPY"].index[-1])
    days = [day for day in nth_trading_days(closes["QQQ"].index) if day <= latest]
    if not days:
        raise RuntimeError("no rebalance day")
    held_day = days[-1]
    nxt = next_nth_weekday(held_day)
    preview_day = pd.Timestamp(latest).normalize()
    qqq = book("纳指100", "QQQ", nq, closes, held_day, preview_day)
    sp_book = book("标普500", "SPY", sp, closes, held_day, preview_day)
    state = refresh_yearly_state(closes, nq, sp)
    yearly = state["strategies"]
    monthly = refresh_monthly_state(closes, nq, sp)["strategies"]
    momentum = refresh_momentum_state(closes, nq, sp)["strategies"]
    for item in yearly:
        print(
            f"  {item['name']}: CAGR={item['metrics']['cagr']*100:.1f}% "
            f"now={[row['ticker'] for row in item['current']['picks']]}",
            flush=True,
        )
    lines = [
        "选股只在每月第 7 个美股交易日收盘做一次。还在名单里的继续拿，只换出局的。",
        f"这一档是 {cn_date(held_day)} 收盘选的。下一次选股是 {cn_date(nxt)} 收盘；遇到美股休市就顺延到下一个交易日。",
        "另一条每天都生效：QQQ 或 SPY 的收盘价只要低于自己的 250 日均线，下一天全部换成 TLT（长期国债）。重新站上均线后，要等到下一次第 7 个交易日才买回股票。",
        f"行情用到 {preview_day.date()}。下一次预览要等 {cn_date(nxt)} 收盘才换成。",
    ]
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    payload = {
        "asof": str(preview_day.date()),
        "generated_at": now,
        "held_on": str(held_day.date()),
        "next_on": str(nxt.date()),
        "lines": lines,
        "qqq": qqq,
        "sp": sp_book,
        "yearly": yearly,
        "monthly": monthly,
        "momentum": momentum,
    }
    (DOCS / "index.html").write_text(render(payload), encoding="utf-8")
    stale = DOCS / "picks.json"
    if stale.exists():
        stale.unlink()
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    print(f"held {held_day.date()} next {nxt.date()} asof {preview_day.date()}", flush=True)
    for key in ("qqq", "sp"):
        rows = payload[key]["held"]["rows"]
        print(key, payload[key]["held"]["mode"], [row["ticker"] for row in rows], flush=True)
    print("wrote", DOCS / "index.html", flush=True)


if __name__ == "__main__":
    main()
