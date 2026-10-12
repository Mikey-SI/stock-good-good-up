# 本月选股

公开页只放本地操作页最上面的选股：纳指 100 和标普 500 这个月该拿的股票。不含券商账户，不会下单。

页面：https://mikey-si.github.io/stock-good-good-up/

每个美股交易日收盘后，GitHub Actions 会跑 `update_picks.py`，把页面写进 `docs/` 并推上去。澳门时间大约是第二天清晨 5:30。也可以在 Actions 里手动点 Run workflow。

选股日是每月第 7 个美股交易日收盘。基准收盘跌破 250 日均线时，页面会改成全部持有 TLT。
