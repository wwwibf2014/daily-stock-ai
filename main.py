import os
import json
import time
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import requests
from jinja2 import Template
from google import genai

# ===========================
# 🔧 使用者設定
# ===========================
TARGET_STOCKS = ["2330.TW", "2317.TW", "0050.TW", "NVDA", "AAPL"]
TZ = ZoneInfo("Asia/Taipei")
GEMINI_MODEL = "gemma-3-27b-it"
CHART_BARS = 120

# GitHub Pages 連結（可用環境變數覆蓋）
GITHUB_USER = os.getenv("GITHUB_USER", "wwwibf2014")
REPO_NAME = os.getenv("REPO_NAME", "daily-stock-ai")

STOCK_NAMES_ZH = {
    "2330.TW": "台積電",
    "2317.TW": "鴻海",
    "0050.TW": "元大台灣50",
    "NVDA": "輝達",
    "AAPL": "蘋果",
}

# ===========================
# 小工具
# ===========================
def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"缺少必要環境變數：{name}")
    return v

def safe_parse_json(text: str) -> dict:
    cleaned = (text or "").strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if not m:
            raise ValueError(f"AI 回傳不是 JSON：{cleaned[:200]}")
        return json.loads(m.group(0))

def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    把 yfinance 偶爾出現的 MultiIndex 欄位扁平化，確保 Open/High/Low/Close/Volume 是單層欄位。
    """
    if isinstance(df.columns, pd.MultiIndex):
        # 可能像 ('Close','AAPL') 或 ('AAPL','Close')，我們把能辨識的那層取出
        new_cols = []
        for col in df.columns:
            # col 是 tuple
            parts = [str(x) for x in col if str(x) != ""]  # 避免空字串
            # 常見情況：('Close', 'AAPL') -> Close
            # 或 ('AAPL','Close') -> Close
            if "Open" in parts: new_cols.append("Open")
            elif "High" in parts: new_cols.append("High")
            elif "Low" in parts: new_cols.append("Low")
            elif "Close" in parts: new_cols.append("Close")
            elif "Volume" in parts: new_cols.append("Volume")
            else:
                new_cols.append("_".join(parts))
        df.columns = new_cols
    return df

def nz(x, default=0.0) -> float:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    return float(x)

# ===========================
# 技術指標（繁體中文）
# ===========================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["20日均線"] = df["Close"].rolling(20).mean()
    df["60日均線"] = df["Close"].rolling(60).mean()

    # RSI(14)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    df["相對強弱指標RSI(14)"] = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["平滑異同移動平均線MACD"] = ema12 - ema26
    df["MACD訊號線"] = df["平滑異同移動平均線MACD"].ewm(span=9, adjust=False).mean()
    df["MACD柱狀體"] = df["平滑異同移動平均線MACD"] - df["MACD訊號線"]

    # 20日乖離率(%)  ✅ 這裡保證 Close/均線是 Series
    df["20日乖離率(%)"] = (df["Close"] / df["20日均線"] - 1) * 100

    # 20日均量 & 均量比
    if "Volume" in df.columns:
        df["20日均量"] = df["Volume"].rolling(20).mean()
        df["均量比(今日/20日)"] = df["Volume"] / df["20日均量"]

    return df

def fetch_history(symbol: str, period="1y", retries=3) -> pd.DataFrame:
    last_err = None
    for i in range(1, retries + 1):
        try:
            df = yf.download(symbol, period=period, progress=False, auto_adjust=False)
            if df is None or df.empty:
                raise RuntimeError("yfinance 回傳空資料")
            df = flatten_yf_columns(df)

            # 必要欄位檢查
            for col in ("Open", "High", "Low", "Close"):
                if col not in df.columns:
                    raise RuntimeError(f"缺少欄位 {col}，目前欄位：{list(df.columns)}")

            # 確保是數字
            for col in ("Open", "High", "Low", "Close", "Volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except Exception as e:
            last_err = e
            time.sleep(1.2 * i)
    raise RuntimeError(f"{symbol} 抓取最終失敗：{last_err}")

def analyze_stock(client: genai.Client, symbol: str):
    df = fetch_history(symbol, period="1y", retries=3)
    df = calculate_indicators(df)

    latest = df.iloc[-1]

    close = nz(latest.get("Close"), 0.0)
    open_ = nz(latest.get("Open"), 0.0)
    high = nz(latest.get("High"), 0.0)
    low = nz(latest.get("Low"), 0.0)

    rsi = nz(latest.get("相對強弱指標RSI(14)"), 50.0)
    macd = nz(latest.get("平滑異同移動平均線MACD"), 0.0)
    macd_sig = nz(latest.get("MACD訊號線"), 0.0)
    macd_hist = nz(latest.get("MACD柱狀體"), 0.0)

    ma20 = nz(latest.get("20日均線"), 0.0)
    ma60 = nz(latest.get("60日均線"), 0.0)
    bias20 = nz(latest.get("20日乖離率(%)"), 0.0)

    vol = nz(latest.get("Volume"), 0.0)
    vol_ma20 = nz(latest.get("20日均量"), 0.0)
    vr = nz(latest.get("均量比(今日/20日)"), 0.0)

    prompt = f"""
你是「給完全新手看的股市老師」，請用非常白話的繁體中文解釋，不要給買賣建議。
只能用「可能」「傾向」「需要觀察」，不要說保證會漲跌。

股票：{symbol}（{STOCK_NAMES_ZH.get(symbol,"")}）
今日K線：開 {open_:.2f} / 高 {high:.2f} / 低 {low:.2f} / 收 {close:.2f}
20日均線：{ma20:.2f}
60日均線：{ma60:.2f}
相對強弱指標 RSI(14)：{rsi:.2f}
MACD：{macd:.4f}
MACD訊號線：{macd_sig:.4f}
MACD柱狀體：{macd_hist:.4f}
成交量：{vol:.0f}
20日均量：{vol_ma20:.0f}
均量比(今日/20日)：{vr:.2f}
20日乖離率(%)：{bias20:.2f}

請只回傳 JSON：
{{
  "signal": "偏多" 或 "偏空" 或 "觀望",
  "reason": "60字內白話解釋（一定要提到：均線 + 成交量 + RSI或MACD其中一個）",
  "tips": ["新手重點1(20字內)","新手重點2(20字內)","新手重點3(20字內)"]
}}
""".strip()

    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    data = safe_parse_json(resp.text)

    signal = data.get("signal", "觀望")
    if signal not in ("偏多", "偏空", "觀望"):
        signal = "觀望"

    # 近 120 根資料做圖
    tail = df.tail(CHART_BARS)

    # K線資料（candlestick 需要 o/h/l/c）
    ohlc = []
    for o, h, l, c in zip(tail["Open"], tail["High"], tail["Low"], tail["Close"]):
        if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
            ohlc.append(None)
        else:
            ohlc.append({"o": float(o), "h": float(h), "l": float(l), "c": float(c)})

    chart_data = {
        "labels": [d.strftime("%Y-%m-%d") for d in tail.index],
        "ohlc": ohlc,
        "volume": [0 if pd.isna(v) else float(v) for v in tail.get("Volume", pd.Series([0]*len(tail)))],
        "ma20": [None if pd.isna(x) else float(x) for x in tail["20日均線"]],
        "ma60": [None if pd.isna(x) else float(x) for x in tail["60日均線"]],
        "vol_ma20": [None if pd.isna(x) else float(x) for x in tail.get("20日均量", pd.Series([None]*len(tail)))],
        "rsi": [None if pd.isna(x) else float(x) for x in tail["相對強弱指標RSI(14)"]],
        "macd": [None if pd.isna(x) else float(x) for x in tail["平滑異同移動平均線MACD"]],
        "macd_sig": [None if pd.isna(x) else float(x) for x in tail["MACD訊號線"]],
        "macd_hist": [None if pd.isna(x) else float(x) for x in tail["MACD柱狀體"]],
    }

    tips = data.get("tips", [])
    if not isinstance(tips, list):
        tips = []

    return {
        "symbol": symbol,
        "name_zh": STOCK_NAMES_ZH.get(symbol, ""),
        "signal": signal,
        "comment": str(data.get("reason", "")).strip(),
        "tips": [str(x).strip() for x in tips[:3]],

        "open": round(open_, 2),
        "high": round(high, 2),
        "low": round(low, 2),
        "price": round(close, 2),

        "ma20": round(ma20, 2),
        "ma60": round(ma60, 2),
        "rsi": round(rsi, 2),
        "macd": round(macd, 4),
        "macd_sig": round(macd_sig, 4),
        "macd_hist": round(macd_hist, 4),

        "volume": int(vol),
        "vol_ma20": int(vol_ma20),
        "vr": round(vr, 2),
        "bias20": round(bias20, 2),

        "chart_data": json.dumps(chart_data, ensure_ascii=False),
    }

def render_html(results, errors):
    html_template = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 每日股市戰報（教學版）</title>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/luxon@3"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial@0.2.1"></script>

<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f3f5f7; padding: 18px; max-width: 1060px; margin: 0 auto; }
  h1 { text-align: center; margin: 10px 0 6px; }
  .sub { text-align: center; color: #777; margin-bottom: 14px; }
  .panel { background: #fff; border-radius: 16px; padding: 16px; box-shadow: 0 6px 14px rgba(0,0,0,0.06); margin-bottom: 14px; }
  .warn { background: #fff3cd; border: 1px solid #ffeeba; color: #856404; border-radius: 12px; padding: 12px; margin-bottom: 14px; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.9em; white-space: pre-wrap; }

  .card { background: #fff; border-radius: 18px; padding: 16px; box-shadow: 0 6px 14px rgba(0,0,0,0.06); margin-bottom: 14px; }
  .top { display:flex; justify-content: space-between; align-items: center; gap: 10px; }
  .title { font-size: 1.25em; font-weight: 900; }
  .badge { padding: 6px 12px; border-radius: 16px; color: #fff; font-weight: 900; font-size: 0.95em; }
  .badge.偏多 { background: #ff4d4d; }
  .badge.偏空 { background: #00b66a; }
  .badge.觀望 { background: #888; }

  .kline { color:#333; margin-top: 6px; }
  .meta { display:flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }
  .chip { background:#f7f7f7; padding: 6px 10px; border-radius: 12px; color:#333; }
  .chip b { font-weight: 900; }

  /* ✅ tooltip */
  .tt { position: relative; display: inline-block; cursor: help; font-weight: 900; text-decoration: underline dotted; text-underline-offset: 3px; }
  .tt .tip {
    position: absolute;
    left: 0;
    bottom: 130%;
    width: min(360px, 80vw);
    background: rgba(20,20,20,0.95);
    color: #fff;
    padding: 10px 12px;
    border-radius: 12px;
    box-shadow: 0 10px 24px rgba(0,0,0,0.25);
    font-weight: 600;
    font-size: 0.92em;
    line-height: 1.5;
    opacity: 0;
    transform: translateY(6px);
    pointer-events: none;
    transition: opacity 0.15s ease, transform 0.15s ease;
    z-index: 50;
  }
  .tt .tip b { color: #ffd966; }
  .tt:hover .tip, .tt:focus-within .tip { opacity: 1; transform: translateY(0); }
  .tt .tip:after{
    content:"";
    position:absolute;
    left: 14px;
    top: 100%;
    border-width: 8px;
    border-style: solid;
    border-color: rgba(20,20,20,0.95) transparent transparent transparent;
  }

  .teachbox { margin-top: 12px; background:#f8f9fa; border-radius: 14px; padding: 12px; border-left: 5px solid #ddd; }
  .teachbox.偏多 { border-left-color: #ff4d4d; }
  .teachbox.偏空 { border-left-color: #00b66a; }
  .teachbox.觀望 { border-left-color: #888; }
  .teach-title { font-weight: 900; margin-bottom: 6px; }
  .points { margin: 8px 0 0; padding-left: 18px; color:#444; }
  .points li { margin: 4px 0; }

  .charts { margin-top: 12px; background:#fbfbfb; border-radius: 14px; padding: 12px; }
  .grid { display:grid; grid-template-columns: 1fr; gap: 12px; }
  @media (min-width: 980px){ .grid { grid-template-columns: 1fr 1fr; } }
  .explain { margin-top: 10px; color:#555; line-height: 1.6; }
  .explain b { color:#222; }

  .footer { text-align:center; color:#999; margin: 18px 0 10px; font-size: 0.9em; }
</style>
</head>
<body>

<h1>📈 AI 每日股市戰報（教學版）</h1>
<div class="sub">{{ date }} · {{ model }}</div>

<div class="panel">
  <div style="font-weight:900; font-size:1.05em;">新手快速讀法（超白話）</div>
  <div class="explain">
    ① 先看 <b>K線＋均線</b>：收盤在均線上方通常偏強；跌破均線可能偏弱。<br>
    ② 再看 <b>成交量</b>：量像「力氣」。價漲＋量增 → 比較有底氣；價漲＋量縮 → 可能續航不足。<br>
    ③ 用 <b>RSI</b> 看「熱度」：&gt;70 可能偏熱，&lt;30 可能偏冷（不是一定反轉）。<br>
    ④ 用 <b>MACD</b> 看「動能」：MACD 上穿訊號線、柱狀體轉正 → 動能偏強；反之偏弱。
  </div>
</div>

{% if errors and results|length == 0 %}
  <div class="warn"><b>本次抓取全部失敗</b><div class="mono">{{ errors|join("\n") }}</div></div>
{% elif errors %}
  <div class="warn"><b>部分股票抓取失敗</b><div class="mono">{{ errors|join("\n") }}</div></div>
{% endif %}

{% for r in results %}
<div class="card">
  <div class="top">
    <div>
      <div class="title">{{ r.symbol }}{% if r.name_zh %}（{{ r.name_zh }}）{% endif %}</div>
      <div class="kline">
        今日K線：開 <b>{{ r.open }}</b>｜高 <b>{{ r.high }}</b>｜低 <b>{{ r.low }}</b>｜收 <b>{{ r.price }}</b>
        <span class="tt" tabindex="0">K線是什麼？
          <span class="tip"><b>K線=一天的價格故事</b><br>開=開始、收=結束、高/低=最高/最低。新手先看收盤在不在均線上方。</span>
        </span>
      </div>
    </div>
    <div class="badge {{ r.signal }}">{{ r.signal }}</div>
  </div>

  <div class="meta">
    <div class="chip"><span class="tt" tabindex="0">20日均線<span class="tip"><b>近20天平均價</b><br>收盤在上方：常被解讀偏強；跌破：可能轉弱或整理。</span></span>：<b>{{ r.ma20 }}</b></div>
    <div class="chip"><span class="tt" tabindex="0">60日均線<span class="tip"><b>中期趨勢參考</b><br>20日看短，60日看中；兩者一起看更清楚。</span></span>：<b>{{ r.ma60 }}</b></div>
    <div class="chip"><span class="tt" tabindex="0">RSI(14)<span class="tip"><b>熱度(0~100)</b><br>&gt;70偏熱、&lt;30偏冷（不代表立刻反轉）。</span></span>：<b>{{ r.rsi }}</b></div>
    <div class="chip"><span class="tt" tabindex="0">MACD<span class="tip"><b>動能指標</b><br>MACD&gt;訊號線：動能偏強；反之偏弱。</span></span>：<b>{{ r.macd }}</b></div>
    <div class="chip"><span class="tt" tabindex="0">MACD柱狀體<span class="tip"><b>MACD-訊號線</b><br>轉正：動能變強跡象；轉負：動能轉弱跡象。</span></span>：<b>{{ r.macd_hist }}</b></div>

    <div class="chip"><span class="tt" tabindex="0">成交量<span class="tip"><b>量=力氣</b><br>價漲＋量增更有底氣；價漲＋量縮可能續航不足。</span></span>：<b>{{ r.volume }}</b></div>
    <div class="chip"><span class="tt" tabindex="0">20日均量<span class="tip"><b>近20天平均成交量</b><br>用來比今天量是大還小。</span></span>：<b>{{ r.vol_ma20 }}</b></div>
    <div class="chip"><span class="tt" tabindex="0">均量比<span class="tip"><b>今日量 / 20日均量</b><br>1.0=差不多；>1較熱；<1較冷。</span></span>：<b>{{ r.vr }}</b></div>
    <div class="chip"><span class="tt" tabindex="0">20日乖離率<span class="tip"><b>跟20日均線差多遠(%)</b><br>太大容易震盪加大（不等於一定回檔）。</span></span>：<b>{{ r.bias20 }}</b></div>
  </div>

  <div class="teachbox {{ r.signal }}">
    <div class="teach-title">🤖 白話解釋（給完全新手）</div>
    <div>{{ r.comment }}</div>
    {% if r.tips and r.tips|length > 0 %}
    <ul class="points">{% for p in r.tips %}<li>{{ p }}</li>{% endfor %}</ul>
    {% endif %}
  </div>

  <div class="charts">
    <div class="grid">
      <div><div style="font-weight:900;margin:4px 0 8px;">① K線＋均線（看趨勢）</div><canvas id="k{{ loop.index }}"></canvas></div>
      <div><div style="font-weight:900;margin:4px 0 8px;">② 成交量＋均量（看力氣）</div><canvas id="v{{ loop.index }}"></canvas></div>
      <div><div style="font-weight:900;margin:4px 0 8px;">③ RSI（看熱度）</div><canvas id="rsi{{ loop.index }}"></canvas></div>
      <div><div style="font-weight:900;margin:4px 0 8px;">④ MACD（看動能）</div><canvas id="macd{{ loop.index }}"></canvas></div>
    </div>
    <div class="explain"><b>小抄：</b>新手先看「均線＋成交量」，再用 RSI/MACD 做確認。</div>
  </div>

  <script>
    (function(){
      const data = {{ r.chart_data | safe }};
      const labels = data.labels;

      new Chart(document.getElementById("k{{ loop.index }}"), {
        type: "candlestick",
        data: { labels: labels, datasets: [
          { label: "K線（開高低收）", data: data.ohlc },
          { label: "20日均線", type: "line", data: data.ma20, spanGaps: true },
          { label: "60日均線", type: "line", data: data.ma60, spanGaps: true }
        ]},
        options: { plugins: { legend: { display: true } }, scales: { x: { display:false } } }
      });

      new Chart(document.getElementById("v{{ loop.index }}"), {
        data: { labels: labels, datasets: [
          { type:"bar", label:"成交量", data: data.volume },
          { type:"line", label:"20日均量", data: data.vol_ma20, spanGaps:true }
        ]},
        options: { plugins: { legend: { display: true } }, scales: { x: { display:false } } }
      });

      new Chart(document.getElementById("rsi{{ loop.index }}"), {
        type: "line",
        data: { labels: labels, datasets: [
          { label:"相對強弱指標 RSI(14)", data: data.rsi, spanGaps:true }
        ]},
        options: { plugins: { legend: { display: true } }, scales: { x: { display:false } } }
      });

      new Chart(document.getElementById("macd{{ loop.index }}"), {
        data: { labels: labels, datasets: [
          { type:"bar", label:"MACD柱狀體", data: data.macd_hist },
          { type:"line", label:"MACD", data: data.macd, spanGaps:true },
          { type:"line", label:"MACD訊號線", data: data.macd_sig, spanGaps:true }
        ]},
        options: { plugins: { legend: { display: true } }, scales: { x: { display:false } } }
      });
    })();
  </script>
</div>
{% endfor %}

<div class="footer">教學提醒：指標是工具，不是保證答案。越多指標同方向，通常越「像」有趨勢，但仍要注意風險。</div>
</body>
</html>
"""
    return Template(html_template).render(
        results=results,
        errors=errors,
        date=datetime.now(TZ).strftime("%Y-%m-%d"),
        model=GEMINI_MODEL,
    )

def line_push(line_token: str, to_id: str, msg: str):
    r = requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={"Authorization": f"Bearer {line_token}", "Content-Type": "application/json"},
        json={"to": to_id, "messages": [{"type": "text", "text": msg}]},
        timeout=20,
    )
    if r.status_code >= 300:
        raise RuntimeError(f"LINE 推播失敗 {r.status_code}: {r.text[:200]}")

def main():
    client = genai.Client(api_key=require_env("GEMINI_API_KEY"))
    line_token = require_env("LINE_TOKEN")
    line_to = require_env("LINE_TO")

    results = []
    errors = []

    for s in TARGET_STOCKS:
        try:
            print(f"🔍 正在分析 {s} ...")
            results.append(analyze_stock(client, s))
            time.sleep(1.0)
        except Exception as e:
            errors.append(f"{s}: {e}")
            print(f"❌ {s} 失敗：{e}")

    html = render_html(results, errors)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    page_url = f"https://{GITHUB_USER}.github.io/{REPO_NAME}/"
    bull = sum(1 for x in results if x["signal"] == "偏多")
    bear = sum(1 for x in results if x["signal"] == "偏空")
    watch = sum(1 for x in results if x["signal"] == "觀望")

    msg = (
        f"📚 教學版股市戰報（{datetime.now(TZ).strftime('%m/%d')}）\n"
        f"偏多：{bull}｜觀望：{watch}｜偏空：{bear}\n"
        f"抓取失敗：{len(errors)}\n\n"
        f"👉 查看K線/成交量/RSI/MACD（含白話泡泡解釋）：\n{page_url}"
    )

    try:
        line_push(line_token, line_to, msg)
        print("✅ LINE 推播成功")
    except Exception as e:
        print(f"⚠️ LINE 推播失敗（不影響網頁生成）：{e}")

if __name__ == "__main__":
    main()
