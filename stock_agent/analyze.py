"""台股持股分析(研究輔助,非投資建議)。

用法: python3 stock_agent/analyze.py 6949 5289 ...
資料來源: FinMind 公開 API(免登入有額度限制,含重試與本機快取)。
"""
import json, os, sys, time, datetime as dt
import urllib.request, urllib.parse
import pandas as pd

API = "https://api.finmindtrade.com/api/v4/data"
CACHE = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(CACHE, exist_ok=True)
TODAY = dt.date.today()
# 股票分割事件: {代號: [(分割後首個交易日, 拆分倍數)]}
SPLITS = {"6949": [("2026-09-07", 20)]}


def fetch(dataset, sid, start):
    key = os.path.join(CACHE, f"{dataset}_{sid}_{TODAY}.json")
    if os.path.exists(key):
        return json.load(open(key))
    q = urllib.parse.urlencode({"dataset": dataset, "data_id": sid, "start_date": start})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=40) as r:
                j = json.load(r)
            if j.get("status") == 200:
                json.dump(j["data"], open(key, "w"))
                return j["data"]
        except Exception:
            pass
        time.sleep(5 * (attempt + 1))
    return None


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def analyze(sid):
    out = {"id": sid}
    start = str(TODAY - dt.timedelta(days=330))
    p = fetch("TaiwanStockPrice", sid, start)
    if not p:
        return {"id": sid, "error": "價格資料取得失敗"}
    df = pd.DataFrame(p)
    # 股票分割還原: 分割日之前的價格除以倍數、成交量乘以倍數
    for split_date, ratio in SPLITS.get(sid, []):
        m = df["date"] < split_date
        for col in ("open", "max", "min", "close"):
            df.loc[m, col] = df.loc[m, col] / ratio
        df.loc[m, "Trading_Volume"] = df.loc[m, "Trading_Volume"] * ratio
    c = df["close"]
    for n in (5, 20, 60, 120):
        df[f"ma{n}"] = c.rolling(n).mean()
    df["rsi"] = rsi(c)
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["sig"] = df["macd"].ewm(span=9).mean()
    df["vma20"] = df["Trading_Volume"].rolling(20).mean()
    last = df.iloc[-1]
    out.update(
        date=last["date"], close=last["close"],
        chg5=c.iloc[-1] / c.iloc[-6] - 1, chg20=c.iloc[-1] / c.iloc[-21] - 1,
        chg60=c.iloc[-1] / c.iloc[-61] - 1 if len(c) > 61 else None,
        ma5=last.ma5, ma20=last.ma20, ma60=last.ma60, ma120=last.ma120,
        rsi=last.rsi, macd=last.macd, sig=last.sig,
        volratio=last["Trading_Volume"] / last.vma20,
        hi120=df["max"].tail(120).max(), lo120=df["min"].tail(120).min(),
        hi20=df["max"].tail(20).max(), lo20=df["min"].tail(20).min(),
        hi250=df["max"].max(), lo250=df["min"].min(), n=len(df),
    )
    # 近 60 日成交量加權價格區(套牢區參考)
    t = df.tail(60)
    out["vwap60"] = (t["close"] * t["Trading_Volume"]).sum() / t["Trading_Volume"].sum()

    # 三大法人
    inst = fetch("TaiwanStockInstitutionalInvestorsBuySell", sid, str(TODAY - dt.timedelta(days=45)))
    if inst:
        d = pd.DataFrame(inst)
        d["net"] = d["buy"] - d["sell"]
        d["grp"] = d["name"].map(lambda x: "外資" if "Foreign" in x else ("投信" if "Investment_Trust" in x else "自營商"))
        g = d.groupby(["date", "grp"])["net"].sum().unstack(fill_value=0) / 1000  # 張
        out["inst5"] = g.tail(5).sum().round(0).to_dict()
        out["inst20"] = g.tail(20).sum().round(0).to_dict()
    # 融資融券
    m = fetch("TaiwanStockMarginPurchaseShortSale", sid, str(TODAY - dt.timedelta(days=45)))
    if m:
        d = pd.DataFrame(m)
        out["margin_now"] = int(d.iloc[-1]["MarginPurchaseTodayBalance"])
        out["margin_chg20"] = int(d.iloc[-1]["MarginPurchaseTodayBalance"] - d.iloc[max(0, len(d) - 21)]["MarginPurchaseTodayBalance"])
        out["short_now"] = int(d.iloc[-1]["ShortSaleTodayBalance"])
    # 月營收
    r = fetch("TaiwanStockMonthRevenue", sid, str(TODAY - dt.timedelta(days=500)))
    if r:
        d = pd.DataFrame(r)
        d["yoy"] = d["revenue"].pct_change(12)
        d["mom"] = d["revenue"].pct_change(1)
        out["rev"] = d.tail(4)[["revenue_year", "revenue_month", "revenue", "yoy", "mom"]].to_dict("records")
    # 本益比
    pe = fetch("TaiwanStockPER", sid, str(TODAY - dt.timedelta(days=400)))
    if pe:
        d = pd.DataFrame(pe)
        cur = d.iloc[-1]
        out["per"] = cur["PER"]; out["pbr"] = cur["PBR"]; out["dy"] = cur["dividend_yield"]
        pos = d["PER"][d["PER"] > 0]
        if len(pos) > 20 and cur["PER"] > 0:
            out["per_pct"] = float((pos < cur["PER"]).mean())
    return out


if __name__ == "__main__":
    ids = sys.argv[1:]
    res = []
    for i in ids:
        res.append(analyze(i))
        time.sleep(1)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
