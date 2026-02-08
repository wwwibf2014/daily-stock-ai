# ===========================
# AI 股市戰報（教學版）
# 一次整合版 main.py
# ===========================

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

STOCK_NAMES_ZH = {
    "2330.TW": "台積電",
    "2317.TW": "鴻海",
    "0050.TW": "元大台灣50",
    "NVDA": "輝達",
    "AAPL": "蘋果",
}

# ===========================
# 工具函式
# ===========================
def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"缺少必要環境變數：{name}")
    return v

def safe_json(text: str) -> dict:
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise ValueError("AI 回傳不是 JSON")
        return json.loads(m.group(0))

# ===========================
# 技術指標（繁體中文）
# ===========================
def indicators(df):
    df = df.copy()
    df["20日均線"] = df["Close"].rolling(20).mean()
    df["60日均線"] = df["Close"].rolling(60).mean()

    d = df["Close"].diff()
    gain = d.where(d > 0, 0).rolling(14).mean()
    loss = (-d.where(d < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    df["RSI"] = 100 - (100 / (1 + rs))

    e12 = df["Close"].ewm(span=12, adjust=False).mean()
    e26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = e12 - e26
    df["MACD訊號線"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD柱狀體"] = df["MACD"] - df["MACD訊號線"]

    df["20日乖離率"] = (df["Close"] / df["20日均線"] - 1) * 100
    df["20日均量"] = df["Volume"].rolling(20).mean()
    df["均量比"] = df["Volume"] / df["20日均量"]
    return df

def fetch(symbol):
    df = yf.download(symbol, period="1y", progress=False)
    if df is None or df.empty:
        raise RuntimeError("股價抓取失敗")
    return df

# ===========================
# 分析單一股票
# ===========================
def analyze(client, symbol):
    df = indicators(fetch(symbol))
    last = df.iloc[-1]

    prompt = f"""
你是給完全新手看的股市老師，用非常白話的繁體中文解釋，不要給買賣建議。

股票：{symbol}（{STOCK_NAMES_ZH.get(symbol,"")}）
收盤價：{last.Close:.2f}
20日均線：{last['20日均線']:.2f}
60日均線：{last['60日均線']:.2f}
RSI：{last.RSI:.1f}
MACD：{last.MACD:.4f}
MACD訊號線：{last['MACD訊號線']:.4f}
成交量：{last.Volume:.0f}
20日均量：{last['20日均量']:.0f}
均量比：{last['均量比']:.2f}

只回傳 JSON：
{{
 "signal": "偏多" 或 "偏空" 或 "觀望",
 "reason": "50字內白話說明",
 "tips": ["一個新手重點","一個新手重點","一個新手重點"]
}}
"""

    res = safe_json(client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    ).text)

    tail = df.tail(CHART_BARS)
    chart = {
        "labels": [d.strftime("%Y-%m-%d") for d in tail.index],
        "ohlc": [{"o":o,"h":h,"l":l,"c":c} for o,h,l,c in zip(tail.Open,tail.High,tail.Low,tail.Close)],
        "volume": tail.Volume.tolist(),
        "ma20": tail["20日均線"].tolist(),
        "ma60": tail["60日均線"].tolist(),
        "rsi": tail.RSI.tolist(),
        "macd": tail.MACD.tolist(),
        "macd_sig": tail["MACD訊號線"].tolist(),
        "macd_hist": tail["MACD柱狀體"].tolist()
    }

    return {
        "symbol": symbol,
        "name": STOCK_NAMES_ZH.get(symbol,""),
        "signal": res["signal"],
        "reason": res["reason"],
        "tips": res["tips"],
        "chart": json.dumps(chart, ensure_ascii=False)
    }

# ===========================
# 主程式
# ===========================
def main():
    client = genai.Client(api_key=require_env("GEMINI_API_KEY"))
    line_token = require_env("LINE_TOKEN")
    line_to = require_env("LINE_TO")

    results = []
    for s in TARGET_STOCKS:
        try:
            results.append(analyze(client, s))
            time.sleep(1)
        except Exception as e:
            print("失敗：", s, e)

    html = Template(open("template.html","w")).render(results=results)
    open("index.html","w",encoding="utf-8").write(html)

    msg = "📘 教學版股市戰報完成\n"
    for r in results:
        msg += f"{r['symbol']} {r['signal']}\n"

    requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={"Authorization": f"Bearer {line_token}"},
        json={"to": line_to, "messages":[{"type":"text","text":msg}]}
    )

if __name__ == "__main__":
    main()
