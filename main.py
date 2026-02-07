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
# 🔧 使用者設定區
# ===========================
TARGET_STOCKS = ["2330.TW", "2317.TW", "0050.TW", "NVDA", "AAPL"]
DEFAULT_REPO_NAME = "daily-stock-ai"
TZ = ZoneInfo("Asia/Taipei")
# ===========================


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"缺少必要環境變數：{name}")
    return v


# === 技術指標（你原本的邏輯，略微加強容錯） ===
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_60"] = df["Close"].rolling(window=60).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, pd.NA)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_12_26_9"] = exp1 - exp2

    return df


def safe_parse_json(text: str) -> dict:
    if not text:
        raise ValueError("空回應")

    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", cleaned)
    if not m:
        raise ValueError(f"找不到 JSON：{cleaned[:200]}")
    return json.loads(m.group(0))


def analyze_stock(client: genai.Client, symbol: str) -> dict | None:
    print(f"🔍 正在分析 {symbol}...")
    try:
        df = yf.Ticker(symbol).history(period="6mo", auto_adjust=False)
        if df is None or df.empty:
            return None

        df = calculate_indicators(df)
        latest = df.iloc[-1]

        close = latest.get("Close")
        if close is None or pd.isna(close):
            return None

        rsi = latest.get("RSI_14", 50)
        macd = latest.get("MACD_12_26_9", 0)
        ma20 = latest.get("SMA_20", 0)
        ma60 = latest.get("SMA_60", 0)

        if pd.isna(rsi): rsi = 50
        if pd.isna(macd): macd = 0
        if pd.isna(ma20): ma20 = 0
        if pd.isna(ma60): ma60 = 0

        prompt = f"""
你是一位嚴謹的交易員。請根據 {symbol} 的今日技術數據進行分析：
收盤價: {float(close):.2f}
RSI (14): {float(rsi):.2f}
MACD: {float(macd):.2f}
月線 (20MA): {float(ma20):.2f}
季線 (60MA): {float(ma60):.2f}

請只回傳 JSON，不要加任何多餘文字：
{{
  "signal": "看多" 或 "看空" 或 "觀望",
  "reason": "20字以內的繁體中文短評"
}}
""".strip()

        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
        )

        ai_text = (resp.text or "").strip()
        data = safe_parse_json(ai_text)

        signal = data.get("signal", "觀望")
        reason = data.get("reason", "AI 無法分析")

        if signal not in ("看多", "看空", "觀望"):
            signal = "觀望"

        if len(reason) > 50:
            reason = reason[:50] + "…"

        return {
            "symbol": symbol,
            "price": round(float(close), 2),
            "rsi": round(float(rsi), 2),
            "signal": signal,
            "comment": reason,
            "date": datetime.now(TZ).strftime("%Y-%m-%d"),
        }

    except Exception as e:
        print(f"❌ Error {symbol}: {e}")
        return None


def render_html(results: list[dict]) -> str:
    html_template = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>股市 AI 戰情室</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f0f2f5; padding: 20px; max-width: 800px; margin: 0 auto; }
h1 { text-align: center; color: #333; margin-bottom: 30px; }
.card { background: white; border-radius: 15px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
.symbol { font-size: 1.4em; font-weight: bold; }
.badge { padding: 6px 12px; border-radius: 20px; color: white; font-weight: bold; }
.badge.看多 { background: #ff4d4d; }
.badge.看空 { background: #00cc66; }
.badge.觀望 { background: #888; }
.comment-box { background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #ddd; }
.comment-box.看多 { border-left-color: #ff4d4d; }
.comment-box.看空 { border-left-color: #00cc66; }
.footer { text-align: center; color: #aaa; margin-top: 30px; font-size: 0.8em; }
</style>
</head>
<body>
<h1>📈 AI 每日股市戰報<br><span style="font-size:0.5em">{{ date }}</span></h1>

{% for r in results %}
<div class="card">
  <div class="header">
    <span class="symbol">{{ r.symbol }}</span>
    <div class="badge {{ r.signal }}">{{ r.signal }}</div>
  </div>
  <div>收盤：{{ r.price }}　RSI：{{ r.rsi }}</div>
  <div class="comment-box {{ r.signal }}">🤖 {{ r.comment }}</div>
</div>
{% endfor %}

<div class="footer">Generated by GitHub Actions & Gemini</div>
</body>
</html>
"""
    template = Template(html_template)
    return template.render(results=results, date=datetime.now(TZ).strftime("%Y-%m-%d"))


def line_push_message(line_token: str, to_id: str, message: str):
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Authorization": f"Bearer {line_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "to": to_id,
        "messages": [{"type": "text", "text": message}],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"LINE 推播失敗 {r.status_code}: {r.text}")


def main():
    gemini_key = require_env("GEMINI_API_KEY")
    line_token = require_env("LINE_TOKEN")

    # 推播目標（請在 workflow env 或 repo secret 設定 LINE_TO）
    to_id = os.getenv("LINE_TO")
    if not to_id:
        raise RuntimeError("缺少 LINE_TO（userId 或 groupId）")

    github_user = os.getenv("GITHUB_USER", "wwwibf2014")
    repo_name = os.getenv("REPO_NAME", DEFAULT_REPO_NAME)
    page_url = f"https://{github_user}.github.io/{repo_name}/"

    client = genai.Client(api_key=gemini_key)

    results = []
    for stock in TARGET_STOCKS:
        r = analyze_stock(client, stock)
        if r:
            results.append(r)
        time.sleep(1.5)

    html = render_html(results)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    bull = sum(1 for x in results if x["signal"] == "看多")
    bear = sum(1 for x in results if x["signal"] == "看空")
    watch = len(results) - bull - bear

    msg = (
        f"\n📊 {datetime.now(TZ).strftime('%m/%d')} 股市戰報已生成！\n"
        f"🔴 看多：{bull} 檔\n"
        f"🟢 看空：{bear} 檔\n"
        f"⚪ 觀望：{watch} 檔\n\n"
        f"👉 查看完整報表：\n{page_url}"
    )

    try:
        line_push_message(line_token, to_id, msg)
        print("✅ LINE Messaging API 推播成功")
    except Exception as e:
        print(f"⚠️ LINE 推播失敗（不影響部署）：{e}")


if __name__ == "__main__":
    main()
