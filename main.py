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

GITHUB_USER = os.getenv("GITHUB_USER", "wwwibf2014")
REPO_NAME = os.getenv("REPO_NAME", "daily-stock-ai")

# 大盤（教學用）
MARKET_INDICES = [
    {"symbol": "^TWII", "name_zh": "台股加權指數", "market": "TW"},
    {"symbol": "^GSPC", "name_zh": "標普500（S&P 500）", "market": "US"},
    {"symbol": "^IXIC", "name_zh": "那斯達克（NASDAQ）", "market": "US"},
]

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
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            parts = [str(x) for x in col if str(x) != ""]
            if "Open" in parts: new_cols.append("Open")
            elif "High" in parts: new_cols.append("High")
            elif "Low" in parts: new_cols.append("Low")
            elif "Close" in parts: new_cols.append("Close")
            elif "Volume" in parts: new_cols.append("Volume")
            else: new_cols.append("_".join(parts))
        df.columns = new_cols
    return df

def nz(x, default=0.0) -> float:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    return float(x)

def clip_text(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"

# ===========================
# 技術指標（繁體中文）
# ===========================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["20日均線"] = df["Close"].rolling(20).mean()
    df["60日均線"] = df["Close"].rolling(60).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    df["相對強弱指標RSI(14)"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["平滑異同移動平均線MACD"] = ema12 - ema26
    df["MACD訊號線"] = df["平滑異同移動平均線MACD"].ewm(span=9, adjust=False).mean()
    df["MACD柱狀體"] = df["平滑異同移動平均線MACD"] - df["MACD訊號線"]

    df["20日乖離率(%)"] = (df["Close"] / df["20日均線"] - 1) * 100

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

            for col in ("Open", "High", "Low", "Close"):
                if col not in df.columns:
                    raise RuntimeError(f"缺少欄位 {col}，目前欄位：{list(df.columns)}")

            for col in ("Open", "High", "Low", "Close", "Volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            return df
        except Exception as e:
            last_err = e
            time.sleep(1.2 * i)

    raise RuntimeError(f"{symbol} 抓取最終失敗：{last_err}")

def build_chart_data(df: pd.DataFrame) -> dict:
    tail = df.tail(CHART_BARS).copy()
    labels = [d.strftime("%Y-%m-%d") for d in tail.index]
    return {
        "labels": labels,
        "open": [None if pd.isna(x) else float(x) for x in tail["Open"]],
        "high": [None if pd.isna(x) else float(x) for x in tail["High"]],
        "low":  [None if pd.isna(x) else float(x) for x in tail["Low"]],
        "close":[None if pd.isna(x) else float(x) for x in tail["Close"]],
        "volume": [0 if pd.isna(v) else float(v) for v in tail.get("Volume", pd.Series([0]*len(tail)))],
        "ma20": [None if pd.isna(x) else float(x) for x in tail["20日均線"]],
        "ma60": [None if pd.isna(x) else float(x) for x in tail["60日均線"]],
        "vol_ma20": [None if pd.isna(x) else float(x) for x in tail.get("20日均量", pd.Series([None]*len(tail)))],
        "rsi": [None if pd.isna(x) else float(x) for x in tail["相對強弱指標RSI(14)"]],
        "macd": [None if pd.isna(x) else float(x) for x in tail["平滑異同移動平均線MACD"]],
        "macd_sig": [None if pd.isna(x) else float(x) for x in tail["MACD訊號線"]],
        "macd_hist": [None if pd.isna(x) else float(x) for x in tail["MACD柱狀體"]],
    }

# ===========================
# 市場環境：抓指數 + AI白話總結（不用出買賣建議）
# ===========================
def analyze_market_index(client: genai.Client, symbol: str, name_zh: str):
    df = calculate_indicators(fetch_history(symbol, period="1y", retries=3))
    latest = df.iloc[-1]

    close = nz(latest.get("Close"), 0.0)
    open_ = nz(latest.get("Open"), 0.0)
    high = nz(latest.get("High"), 0.0)
    low = nz(latest.get("Low"), 0.0)

    ma20 = nz(latest.get("20日均線"), 0.0)
    ma60 = nz(latest.get("60日均線"), 0.0)
    rsi = nz(latest.get("相對強弱指標RSI(14)"), 50.0)

    macd = nz(latest.get("平滑異同移動平均線MACD"), 0.0)
    macd_sig = nz(latest.get("MACD訊號線"), 0.0)
    macd_hist = nz(latest.get("MACD柱狀體"), 0.0)

    vol = nz(latest.get("Volume"), 0.0)
    vol_ma20 = nz(latest.get("20日均量"), 0.0)
    vr = nz(latest.get("均量比(今日/20日)"), 0.0)

    prompt = f"""
你是「給完全新手看的市場老師」。請用繁體中文、非常白話。
這是「大盤指數」的環境說明：不要講買賣建議，只講氣氛/順風逆風。

指數：{symbol}（{name_zh}）
今日K線：開 {open_:.2f} / 高 {high:.2f} / 低 {low:.2f} / 收 {close:.2f}
20日均線：{ma20:.2f}
60日均線：{ma60:.2f}
RSI(14)：{rsi:.2f}
MACD：{macd:.4f}
MACD訊號線：{macd_sig:.4f}
MACD柱狀體：{macd_hist:.4f}
成交量：{vol:.0f}
20日均量：{vol_ma20:.0f}
均量比：{vr:.2f}

請只回 JSON：
{{
  "mood": "偏多" 或 "偏空" 或 "整理",
  "summary": "60字內白話（一定提到：均線 + 成交量 + RSI或MACD其中一個）",
  "teach": ["新手提示1(20字內)","新手提示2(20字內)"]
}}
""".strip()

    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    data = safe_parse_json(resp.text)

    mood = data.get("mood", "整理")
    if mood not in ("偏多", "偏空", "整理"):
        mood = "整理"

    teach = data.get("teach", [])
    if not isinstance(teach, list):
        teach = []

    return {
        "symbol": symbol,
        "name_zh": name_zh,
        "mood": mood,
        "summary": str(data.get("summary", "")).strip(),
        "teach": [str(x).strip() for x in teach[:2]],

        "open_now": round(open_, 2),
        "high_now": round(high, 2),
        "low_now": round(low, 2),
        "close_now": round(close, 2),

        "ma20_now": round(ma20, 2),
        "ma60_now": round(ma60, 2),
        "rsi_now": round(rsi, 2),
        "macd_now": round(macd, 4),
        "macd_hist_now": round(macd_hist, 4),

        "volume_now": int(vol),
        "vol_ma20_now": int(vol_ma20),
        "vr_now": round(vr, 2),

        "chart_data": json.dumps(build_chart_data(df), ensure_ascii=False),
    }

# ===========================
# 個股分析（含：與大盤關係教學文字）
# ===========================
def analyze_stock(client: genai.Client, symbol: str, market_context: dict):
    df = calculate_indicators(fetch_history(symbol, period="1y", retries=3))
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

    # 取市場環境（縮短避免 prompt 太長）
    tw = market_context.get("TWII", {})
    us_sp = market_context.get("GSPC", {})
    us_nq = market_context.get("IXIC", {})

    prompt = f"""
你是「給完全新手看的股市老師」，請用非常白話的繁體中文解釋，不要給買賣建議。
只能用「可能」「傾向」「需要觀察」，不要說保證會漲跌。

個股：{symbol}（{STOCK_NAMES_ZH.get(symbol,"")}）
今日K線：開 {open_:.2f} / 高 {high:.2f} / 低 {low:.2f} / 收 {close:.2f}
20日均線：{ma20:.2f}
60日均線：{ma60:.2f}
RSI(14)：{rsi:.2f}
MACD：{macd:.4f}
MACD訊號線：{macd_sig:.4f}
MACD柱狀體：{macd_hist:.4f}
成交量：{vol:.0f}
20日均量：{vol_ma20:.0f}
均量比：{vr:.2f}
20日乖離率(%)：{bias20:.2f}

市場環境（大盤）摘要：
- 台股加權指數：{tw.get("mood","")}，{tw.get("summary","")}
- 美股S&P500：{us_sp.get("mood","")}，{us_sp.get("summary","")}
- 美股NASDAQ：{us_nq.get("mood","")}，{us_nq.get("summary","")}

請只回傳 JSON：
{{
  "signal": "偏多" 或 "偏空" 或 "觀望",
  "reason": "60字內白話解釋（一定要提到：均線 + 成交量 + RSI或MACD其中一個）",
  "tips": ["新手重點1(20字內)","新手重點2(20字內)","新手重點3(20字內)"],
  "market_link": "用白話解釋：大盤與個股可能的關係（60字內，像順風/逆風的比喻）"
}}
""".strip()

    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    data = safe_parse_json(resp.text)

    signal = data.get("signal", "觀望")
    if signal not in ("偏多", "偏空", "觀望"):
        signal = "觀望"

    tips = data.get("tips", [])
    if not isinstance(tips, list):
        tips = []

    market_link = str(data.get("market_link", "")).strip()

    return {
        "symbol": symbol,
        "name_zh": STOCK_NAMES_ZH.get(symbol, ""),
        "signal": signal,
        "comment": str(data.get("reason", "")).strip(),
        "tips": [str(x).strip() for x in tips[:3]],
        "market_link": market_link,

        "open_now": round(open_, 2),
        "high_now": round(high, 2),
        "low_now": round(low, 2),
        "price": round(close, 2),

        "ma20_now": round(ma20, 2),
        "ma60_now": round(ma60, 2),
        "rsi_now": round(rsi, 2),
        "macd_now": round(macd, 4),
        "macd_hist_now": round(macd_hist, 4),

        "volume_now": int(vol),
        "vol_ma20_now": int(vol_ma20),
        "vr_now": round(vr, 2),
        "bias20_now": round(bias20, 2),

        "chart_data": json.dumps(build_chart_data(df), ensure_ascii=False),
    }

# ===========================
# HTML（固定 Chart.js 版本 + 自繪K線 + 市場環境區塊）
# ===========================
def render_html(market_results, stock_results, errors):
    html_template = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 每日股市戰報（教學版）</title>

<!-- ✅ 固定 Chart.js 版本 -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>

<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f3f5f7; padding: 18px; max-width: 1100px; margin: 0 auto; }
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
  .badge.整理 { background: #6c757d; }

  .kline { color:#333; margin-top: 6px; }
  .meta { display:flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }
  .chip { background:#f7f7f7; padding: 6px 10px; border-radius: 12px; color:#333; }
  .chip b { font-weight: 900; }

  .teachbox { margin-top: 12px; background:#f8f9fa; border-radius: 14px; padding: 12px; border-left: 5px solid #ddd; }
  .teachbox.偏多 { border-left-color: #ff4d4d; }
  .teachbox.偏空 { border-left-color: #00b66a; }
  .teachbox.觀望 { border-left-color: #888; }
  .teachbox.整理 { border-left-color: #6c757d; }
  .teach-title { font-weight: 900; margin-bottom: 6px; }
  .points { margin: 8px 0 0; padding-left: 18px; color:#444; }
  .points li { margin: 4px 0; }

  .charts { margin-top: 12px; background:#fbfbfb; border-radius: 14px; padding: 12px; }
  .grid { display:grid; grid-template-columns: 1fr; gap: 12px; }
  @media (min-width: 980px){ .grid { grid-template-columns: 1fr 1fr; } }
  .footer { text-align:center; color:#999; margin: 18px 0 10px; font-size: 0.9em; }

  .market-grid { display:grid; grid-template-columns: 1fr; gap: 12px; }
  @media (min-width: 980px){ .market-grid { grid-template-columns: 1fr 1fr; } }

  .hint { color:#555; line-height: 1.6; }
</style>
</head>
<body>

<h1>📈 AI 每日股市戰報（教學版）</h1>
<div class="sub">{{ date }} · {{ model }}</div>

<div class="panel">
  <div style="font-weight:900; font-size:1.05em;">新手快速讀法（超白話）</div>
  <div class="hint">
    ① 先看 <b>市場環境（大盤）</b>：像海流，順風逆風會影響多數個股。<br>
    ② 再看 <b>K線＋均線</b>：收盤在均線上方通常偏強；跌破均線可能偏弱。<br>
    ③ 看 <b>成交量</b>：量像力氣；價漲＋量增更有底氣。<br>
    ④ 用 <b>RSI / MACD</b> 做確認：熱度與動能是否一致。
  </div>
</div>

{% if errors %}
  <div class="warn"><b>本次有錯誤</b><div class="mono">{{ errors|join("\n") }}</div></div>
{% endif %}

<!-- ========== 市場環境（大盤） ========== -->
<div class="panel">
  <div style="font-weight:900; font-size:1.1em; margin-bottom:8px;">🌏 今日市場環境（大盤）</div>
  <div class="market-grid">
    {% for m in market_results %}
    <div class="card" style="margin-bottom:0;">
      <div class="top">
        <div>
          <div class="title">{{ m.name_zh }}（{{ m.symbol }}）</div>
          <div class="kline">今日：開 <b>{{ m.open_now }}</b>｜高 <b>{{ m.high_now }}</b>｜低 <b>{{ m.low_now }}</b>｜收 <b>{{ m.close_now }}</b></div>
        </div>
        <div class="badge {{ m.mood }}">{{ m.mood }}</div>
      </div>

      <div class="meta">
        <div class="chip">20日均線：<b>{{ m.ma20_now }}</b></div>
        <div class="chip">60日均線：<b>{{ m.ma60_now }}</b></div>
        <div class="chip">RSI(14)：<b>{{ m.rsi_now }}</b></div>
        <div class="chip">MACD：<b>{{ m.macd_now }}</b></div>
        <div class="chip">MACD柱狀體：<b>{{ m.macd_hist_now }}</b></div>
        <div class="chip">成交量：<b>{{ m.volume_now }}</b></div>
        <div class="chip">均量比：<b>{{ m.vr_now }}</b></div>
      </div>

      <div class="teachbox {{ m.mood }}">
        <div class="teach-title">📌 白話環境說明</div>
        <div>{{ m.summary }}</div>
        {% if m.teach %}
        <ul class="points">{% for p in m.teach %}<li>{{ p }}</li>{% endfor %}</ul>
        {% endif %}
      </div>

      <div class="charts">
        <div class="grid">
          <div><div style="font-weight:900;margin:4px 0 8px;">① K線＋均線</div><canvas id="mk{{ loop.index }}"></canvas></div>
          <div><div style="font-weight:900;margin:4px 0 8px;">② 成交量＋均量</div><canvas id="mv{{ loop.index }}"></canvas></div>
          <div><div style="font-weight:900;margin:4px 0 8px;">③ RSI</div><canvas id="mrsi{{ loop.index }}"></canvas></div>
          <div><div style="font-weight:900;margin:4px 0 8px;">④ MACD</div><canvas id="mmacd{{ loop.index }}"></canvas></div>
        </div>
      </div>

      <script>
        (function(){
          const data = {{ m.chart_data | safe }};
          const candlePlugin = {
            id: 'candlePlugin',
            afterDatasetsDraw(chart) {
              const {ctx, scales: {x, y}} = chart;
              ctx.save();
              ctx.lineWidth = 1;
              ctx.globalAlpha = 0.9;

              for (let i=0; i<data.labels.length; i++){
                const lab = data.labels[i];
                const o = data.open[i], h = data.high[i], l = data.low[i], c = data.close[i];
                if (o==null || h==null || l==null || c==null) continue;

                const xPos = x.getPixelForValue(lab);
                const yHi = y.getPixelForValue(h);
                const yLo = y.getPixelForValue(l);

                ctx.beginPath();
                ctx.moveTo(xPos, yHi);
                ctx.lineTo(xPos, yLo);
                ctx.stroke();
              }

              const barW = Math.max(3, Math.min(8, chart.chartArea.width / data.labels.length * 0.6));
              for (let i=0; i<data.labels.length; i++){
                const lab = data.labels[i];
                const o = data.open[i], c = data.close[i];
                if (o==null || c==null) continue;

                const xPos = x.getPixelForValue(lab);
                const yO = y.getPixelForValue(o);
                const yC = y.getPixelForValue(c);

                const top = Math.min(yO, yC);
                const height = Math.max(1, Math.abs(yC - yO));
                ctx.fillRect(xPos - barW/2, top, barW, height);
              }

              ctx.restore();
            }
          };

          new Chart(document.getElementById("mk{{ loop.index }}"), {
            type: "line",
            data: { labels: data.labels, datasets: [
              { label:"20日均線", data:data.ma20, spanGaps:true },
              { label:"60日均線", data:data.ma60, spanGaps:true },
            ]},
            options: { plugins:{legend:{display:true}}, scales:{x:{display:false}} },
            plugins: [candlePlugin]
          });

          new Chart(document.getElementById("mv{{ loop.index }}"), {
            data: { labels:data.labels, datasets:[
              { type:"bar", label:"成交量", data:data.volume },
              { type:"line", label:"20日均量", data:data.vol_ma20, spanGaps:true },
            ]},
            options: { plugins:{legend:{display:true}}, scales:{x:{display:false}} }
          });

          new Chart(document.getElementById("mrsi{{ loop.index }}"), {
            type:"line",
            data:{ labels:data.labels, datasets:[{ label:"RSI(14)", data:data.rsi, spanGaps:true }]},
            options:{ plugins:{legend:{display:true}}, scales:{x:{display:false}} }
          });

          new Chart(document.getElementById("mmacd{{ loop.index }}"), {
            data:{ labels:data.labels, datasets:[
              { type:"bar", label:"MACD柱狀體", data:data.macd_hist },
              { type:"line", label:"MACD", data:data.macd, spanGaps:true },
              { type:"line", label:"MACD訊號線", data:data.macd_sig, spanGaps:true },
            ]},
            options:{ plugins:{legend:{display:true}}, scales:{x:{display:false}} }
          });
        })();
      </script>
    </div>
    {% endfor %}
  </div>
  <div class="hint" style="margin-top:10px;">
    <b>教學重點：</b>大盤像海流，若大盤偏空，很多個股就算短線反彈也可能走得辛苦；若大盤偏多，個股更容易順風上行。
  </div>
</div>

<!-- ========== 個股 ========== -->
{% for r in stock_results %}
<div class="card">
  <div class="top">
    <div>
      <div class="title">{{ r.symbol }}{% if r.name_zh %}（{{ r.name_zh }}）{% endif %}</div>
      <div class="kline">今日：開 <b>{{ r.open_now }}</b>｜高 <b>{{ r.high_now }}</b>｜低 <b>{{ r.low_now }}</b>｜收 <b>{{ r.price }}</b></div>
    </div>
    <div class="badge {{ r.signal }}">{{ r.signal }}</div>
  </div>

  <div class="meta">
    <div class="chip">20日均線：<b>{{ r.ma20_now }}</b></div>
    <div class="chip">60日均線：<b>{{ r.ma60_now }}</b></div>
    <div class="chip">RSI(14)：<b>{{ r.rsi_now }}</b></div>
    <div class="chip">MACD：<b>{{ r.macd_now }}</b></div>
    <div class="chip">MACD柱狀體：<b>{{ r.macd_hist_now }}</b></div>
    <div class="chip">成交量：<b>{{ r.volume_now }}</b></div>
    <div class="chip">20日均量：<b>{{ r.vol_ma20_now }}</b></div>
    <div class="chip">均量比：<b>{{ r.vr_now }}</b></div>
    <div class="chip">20日乖離率：<b>{{ r.bias20_now }}</b></div>
  </div>

  <div class="teachbox {{ r.signal }}">
    <div class="teach-title">🤖 白話解釋（給完全新手）</div>
    <div>{{ r.comment }}</div>
    {% if r.tips %}<ul class="points">{% for p in r.tips %}<li>{{ p }}</li>{% endfor %}</ul>{% endif %}
  </div>

  <div class="teachbox" style="border-left-color:#4d7cff;">
    <div class="teach-title">🌊 大盤 × 個股：可能的關係（教學版）</div>
    <div>{{ r.market_link }}</div>
  </div>

  <div class="charts">
    <div class="grid">
      <div><div style="font-weight:900;margin:4px 0 8px;">① K線＋均線</div><canvas id="k{{ loop.index }}"></canvas></div>
      <div><div style="font-weight:900;margin:4px 0 8px;">② 成交量＋均量</div><canvas id="v{{ loop.index }}"></canvas></div>
      <div><div style="font-weight:900;margin:4px 0 8px;">③ RSI</div><canvas id="rsi{{ loop.index }}"></canvas></div>
      <div><div style="font-weight:900;margin:4px 0 8px;">④ MACD</div><canvas id="macd{{ loop.index }}"></canvas></div>
    </div>
  </div>

  <script>
    (function(){
      const data = {{ r.chart_data | safe }};
      const candlePlugin = {
        id: 'candlePlugin',
        afterDatasetsDraw(chart) {
          const {ctx, scales: {x, y}} = chart;
          ctx.save();
          ctx.lineWidth = 1;
          ctx.globalAlpha = 0.9;

          for (let i=0; i<data.labels.length; i++){
            const lab = data.labels[i];
            const o = data.open[i], h = data.high[i], l = data.low[i], c = data.close[i];
            if (o==null || h==null || l==null || c==null) continue;

            const xPos = x.getPixelForValue(lab);
            const yHi = y.getPixelForValue(h);
            const yLo = y.getPixelForValue(l);

            ctx.beginPath();
            ctx.moveTo(xPos, yHi);
            ctx.lineTo(xPos, yLo);
            ctx.stroke();
          }

          const barW = Math.max(3, Math.min(8, chart.chartArea.width / data.labels.length * 0.6));
          for (let i=0; i<data.labels.length; i++){
            const lab = data.labels[i];
            const o = data.open[i], c = data.close[i];
            if (o==null || c==null) continue;

            const xPos = x.getPixelForValue(lab);
            const yO = y.getPixelForValue(o);
            const yC = y.getPixelForValue(c);

            const top = Math.min(yO, yC);
            const height = Math.max(1, Math.abs(yC - yO));
            ctx.fillRect(xPos - barW/2, top, barW, height);
          }

          ctx.restore();
        }
      };

      new Chart(document.getElementById("k{{ loop.index }}"), {
        type: "line",
        data: { labels: data.labels, datasets: [
          { label:"20日均線", data:data.ma20, spanGaps:true },
          { label:"60日均線", data:data.ma60, spanGaps:true },
        ]},
        options: { plugins:{legend:{display:true}}, scales:{x:{display:false}} },
        plugins: [candlePlugin]
      });

      new Chart(document.getElementById("v{{ loop.index }}"), {
        data: { labels:data.labels, datasets:[
          { type:"bar", label:"成交量", data:data.volume },
          { type:"line", label:"20日均量", data:data.vol_ma20, spanGaps:true },
        ]},
        options: { plugins:{legend:{display:true}}, scales:{x:{display:false}} }
      });

      new Chart(document.getElementById("rsi{{ loop.index }}"), {
        type:"line",
        data:{ labels:data.labels, datasets:[{ label:"RSI(14)", data:data.rsi, spanGaps:true }]},
        options:{ plugins:{legend:{display:true}}, scales:{x:{display:false}} }
      });

      new Chart(document.getElementById("macd{{ loop.index }}"), {
        data:{ labels:data.labels, datasets:[
          { type:"bar", label:"MACD柱狀體", data:data.macd_hist },
          { type:"line", label:"MACD", data:data.macd, spanGaps:true },
          { type:"line", label:"MACD訊號線", data:data.macd_sig, spanGaps:true },
        ]},
        options:{ plugins:{legend:{display:true}}, scales:{x:{display:false}} }
      });
    })();
  </script>
</div>
{% endfor %}

<div class="footer">教學提醒：指標是工具，不是保證答案。建議用「大盤→趨勢→量→RSI/MACD」的順序閱讀。</div>
</body>
</html>
"""
    return Template(html_template).render(
        market_results=market_results,
        stock_results=stock_results,
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

    errors = []

    # 先做市場環境（大盤）
    market_results = []
    market_context = {}
    for idx in MARKET_INDICES:
        try:
            print(f"🌏 分析大盤 {idx['symbol']} ...")
            r = analyze_market_index(client, idx["symbol"], idx["name_zh"])
            market_results.append(r)
            # 做成 prompt 用的 context（縮短存摘要即可）
            key = idx["symbol"].replace("^", "")
            market_context[key] = {"mood": r["mood"], "summary": clip_text(r["summary"], 80)}
            time.sleep(0.8)
        except Exception as e:
            errors.append(f"{idx['symbol']}: {e}")
            print(f"❌ 大盤 {idx['symbol']} 失敗：{e}")

    # 個股
    stock_results = []
    for s in TARGET_STOCKS:
        try:
            print(f"🔍 正在分析 {s} ...")
            stock_results.append(analyze_stock(client, s, market_context))
            time.sleep(1.0)
        except Exception as e:
            errors.append(f"{s}: {e}")
            print(f"❌ {s} 失敗：{e}")

    html = render_html(market_results, stock_results, errors)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    page_url = f"https://{GITHUB_USER}.github.io/{REPO_NAME}/"
    bull = sum(1 for x in stock_results if x["signal"] == "偏多")
    bear = sum(1 for x in stock_results if x["signal"] == "偏空")
    watch = sum(1 for x in stock_results if x["signal"] == "觀望")

    msg = (
        f"📚 教學版股市戰報（{datetime.now(TZ).strftime('%m/%d')}）\n"
        f"個股：偏多{bull}｜觀望{watch}｜偏空{bear}\n"
        f"大盤：TWII/GSPC/IXIC 已更新\n"
        f"錯誤：{len(errors)}\n\n"
        f"👉 查看網頁：\n{page_url}"
    )

    try:
        line_push(line_token, line_to, msg)
        print("✅ LINE 推播成功")
    except Exception as e:
        print(f"⚠️ LINE 推播失敗（不影響網頁生成）：{e}")

if __name__ == "__main__":
    main()
