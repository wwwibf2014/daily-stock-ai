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
DEFAULT_REPO_NAME = "daily-stock-ai"
TZ = ZoneInfo("Asia/Taipei")

# 使用者指定模型
GEMINI_MODEL = "gemma-3-27b-it"
# ===========================


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"缺少必要環境變數：{name}")
    return v


def safe_parse_json(text: str) -> dict:
    cleaned = text.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(cleaned)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if not m:
            raise ValueError("AI 回傳非 JSON")
        return json.loads(m.group(0))


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
    ret = df["Close"].pct_change()
    df["年化波動率(20日)"] = ret.rolling(20).std() * (252 ** 0.5) * 100

    if "Volume" in df.columns:
        df["20日均量"] = df["Volume"].rolling(20).mean()
        df["均量比(今日/20日)"] = df["Volume"] / df["20日均量"]

    return df


def fetch_history(symbol: str, period="1y", retries=3) -> pd.DataFrame:
    last_err = None
    for i in range(1, retries + 1):
        try:
            df = yf.Ticker(symbol).history(period=period)
            if df is None or df.empty:
                df = yf.download(symbol, period=period, progress=False)
            if df is None or df.empty:
                raise RuntimeError("yfinance 回傳空資料")
            return df
        except Exception as e:
            last_err = e
            time.sleep(1.2 * i)
    raise RuntimeError(f"{symbol} 抓取失敗：{last_err}")


def analyze_stock(client: genai.Client, symbol: str):
    try:
        df = fetch_history(symbol)
        df = calculate_indicators(df)
        latest = df.iloc[-1]

        rsi = latest.get("相對強弱指標RSI(14)", 50)
        macd = latest.get("平滑異同移動平均線MACD", 0)
        macd_sig = latest.get("MACD訊號線", 0)
        macd_hist = latest.get("MACD柱狀體", 0)
        ma20 = latest.get("20日均線", 0)
        ma60 = latest.get("60日均線", 0)
        bias20 = latest.get("20日乖離率(%)", 0)
        vol20 = latest.get("年化波動率(20日)", 0)
        vr = latest.get("均量比(今日/20日)", 0)

        prompt = f"""
請以繁體中文進行技術分析（僅技術面）：

股票代號：{symbol}
收盤價：{latest['Close']:.2f}
相對強弱指標 RSI(14)：{rsi:.2f}
平滑異同移動平均線 MACD：{macd:.4f}
MACD 訊號線：{macd_sig:.4f}
MACD 柱狀體：{macd_hist:.4f}
20日均線：{ma20:.2f}
60日均線：{ma60:.2f}
20日乖離率(%)：{bias20:.2f}
年化波動率(20日)(%)：{vol20:.2f}
均量比(今日/20日)：{vr:.2f}

請只回傳 JSON：
{{
  "signal": "偏多" 或 "偏空" 或 "觀望",
  "reason": "30字以內，需提到至少兩個技術指標"
}}
""".strip()

        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        data = safe_parse_json(resp.text)

        tail = df.tail(60)
        chart_data = {
            "labels": [d.strftime("%m/%d") for d in tail.index],
            "close": tail["Close"].fillna("").tolist(),
            "ma20": tail["20日均線"].fillna("").tolist(),
            "ma60": tail["60日均線"].fillna("").tolist(),
            "rsi": tail["相對強弱指標RSI(14)"].fillna("").tolist(),
            "macd": tail["平滑異同移動平均線MACD"].fillna("").tolist(),
            "macd_sig": tail["MACD訊號線"].fillna("").tolist(),
            "macd_hist": tail["MACD柱狀體"].fillna("").tolist(),
        }

        return {
            "symbol": symbol,
            "price": round(float(latest["Close"]), 2),
            "rsi": round(float(rsi), 2),
            "signal": data.get("signal", "觀望"),
            "comment": data.get("reason", ""),
            "ma20": round(float(ma20), 2),
            "ma60": round(float(ma60), 2),
            "bias20": round(float(bias20), 2),
            "macd": round(float(macd), 4),
            "macd_hist": round(float(macd_hist), 4),
            "vol20": round(float(vol20), 2),
            "vr": round(float(vr), 2),
            "chart_data": json.dumps(chart_data, ensure_ascii=False),
        }, None

    except Exception as e:
        return None, f"{symbol}: {e}"


# ===========================
# HTML（含圖表解釋）
# ===========================
def render_html(results, errors):
    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>AI 每日股市戰報</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto;background:#f0f2f5;padding:20px}
.card{background:#fff;border-radius:16px;padding:18px;margin-bottom:18px}
.badge{padding:6px 12px;border-radius:18px;color:#fff;font-weight:700}
.badge.偏多{background:#ff4d4d}
.badge.偏空{background:#00cc66}
.badge.觀望{background:#888}
</style>
</head>
<body>
<h1>📈 AI 每日股市戰報</h1>
<p>{{ date }} · Generated by GitHub Actions & Gemini</p>

{% for r in results %}
<div class="card">
<h2>{{ r.symbol }} <span class="badge {{ r.signal }}">{{ r.signal }}</span></h2>
<p>收盤 {{ r.price }}｜RSI {{ r.rsi }}｜20日乖離率 {{ r.bias20 }}%</p>
<p>🤖 {{ r.comment }}</p>
<canvas id="c{{ loop.index }}"></canvas>
<script>
new Chart(document.getElementById("c{{ loop.index }}"),{
 type:"line",
 data:{labels:{{ r.chart_data | safe }}.labels,
 datasets:[
 {label:"收盤價",data:{{ r.chart_data | safe }}.close},
 {label:"20日均線",data:{{ r.chart_data | safe }}.ma20},
 {label:"60日均線",data:{{ r.chart_data | safe }}.ma60}
 ]}
});
</script>
</div>
{% endfor %}
</body>
</html>"""
    return Template(html).render(
        results=results,
        errors=errors,
        date=datetime.now(TZ).strftime("%Y-%m-%d"),
    )


def line_push(line_token, to_id, msg):
    requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={"Authorization": f"Bearer {line_token}"},
        json={"to": to_id, "messages": [{"type": "text", "text": msg}]},
        timeout=20,
    )


def main():
    client = genai.Client(api_key=require_env("GEMINI_API_KEY"))
    line_token = require_env("LINE_TOKEN")
    to_id = require_env("LINE_TO")

    results, errors = [], []
    for s in TARGET_STOCKS:
        r, e = analyze_stock(client, s)
        if r: results.append(r)
        if e: errors.append(e)
        time.sleep(1)

    html = render_html(results, errors)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    msg = f"📊 今日股市戰報\n看多/偏多：{sum(1 for x in results if x['signal']=='偏多')}\n"
    msg += f"看空/偏空：{sum(1 for x in results if x['signal']=='偏空')}\n"
    msg += f"觀望：{sum(1 for x in results if x['signal']=='觀望')}\n"
    msg += f"失敗：{len(errors)}"

    line_push(line_token, to_id, msg)


if __name__ == "__main__":
    main()
