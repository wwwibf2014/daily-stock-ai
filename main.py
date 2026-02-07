import yfinance as yf
import pandas as pd
import google.generativeai as genai
import requests
import os
import time
import json
from datetime import datetime
from jinja2 import Template

# ===========================
# 🔧 使用者設定區 (請修改這裡)
# ===========================
# 您的 GitHub 帳號 (用於生成網頁連結)
GITHUB_USER = "wwwibf2014" 
REPO_NAME = "daily-stock-ai"

# 追蹤股票清單 (台股請加 .TW, 美股直接打代號)
TARGET_STOCKS = ["2330.TW", "2317.TW", "0050.TW", "NVDA", "AAPL"] 

# ===========================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LINE_TOKEN = os.getenv("LINE_TOKEN")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemma-3-27b-it')

results = []

# === 🛠️ 新增：自己計算技術指標的函式 (取代 pandas_ta) ===
def calculate_indicators(df):
    # 1. 計算移動平均線 (MA)
    df['SMA_20'] = df['Close'].rolling(window=20).mean() # 月線
    df['SMA_60'] = df['Close'].rolling(window=60).mean() # 季線
    
    # 2. 計算 RSI (14天)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 3. 計算 MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = exp1 - exp2
    df['MACD_SIGNAL'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    
    return df

def analyze_stock(symbol):
    print(f"🔍 正在分析 {symbol}...")
    try:
        # 1. 抓取資料 (過去半年)
        df = yf.Ticker(symbol).history(period="6mo")
        if df.empty: return None
        
        # 2. 呼叫我們自己寫的函式計算指標
        df = calculate_indicators(df)
        
        # 取得最新一筆數據 (iloc[-1])
        latest = df.iloc[-1]
        
        # 3. 準備給 AI 的數據
        # 使用 .get() 避免剛上市股票數據不足導致報錯
        rsi = latest.get('RSI_14', 50) 
        macd = latest.get('MACD_12_26_9', 0)
        ma20 = latest.get('SMA_20', 0)
        ma60 = latest.get('SMA_60', 0)

        # 處理 NaN (若數據不足)
        if pd.isna(rsi): rsi = 50
        if pd.isna(macd): macd = 0
        if pd.isna(ma20): ma20 = 0

        prompt = f"""
        你是一位嚴謹的華爾街交易員。請根據 {symbol} 的今日技術數據進行分析：
        收盤價: {latest['Close']:.2f}
        RSI (14): {rsi:.2f}
        MACD: {macd:.2f}
        月線 (20MA): {ma20:.2f}
        季線 (60MA): {ma60:.2f}
        
        請依照以下 JSON 格式回傳，不要有其他廢話：
        {{
            "signal": "看多" 或 "看空" 或 "觀望",
            "reason": "20字以內的繁體中文短評，例如：RSI過熱且跌破月線，建議獲利了結。"
        }}
        """
        
        response = model.generate_content(prompt)
        ai_text = response.text.strip()
        
        # 清洗 AI 回傳的格式
        ai_text = ai_text.replace("```json", "").replace("```", "")
        analysis = json.loads(ai_text)
        
        return {
            "symbol": symbol,
            "price": round(latest['Close'], 2),
            "rsi": round(rsi, 2),
            "signal": analysis.get("signal", "觀望"),
            "comment": analysis.get("reason", "AI 無法分析"),
            "date": datetime.now().strftime('%Y-%m-%d')
        }
    except Exception as e:
        print(f"❌ Error {symbol}: {e}")
        return None

# === 主程式迴圈 ===
for stock in TARGET_STOCKS:
    res = analyze_stock(stock)
    if res:
        results.append(res)
    time.sleep(2) # 休息一下避免 API 限制

# === 生成 HTML 報表 ===
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
    .card { background: white; border-radius: 15px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: flex; flex-direction: column; }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
    .symbol { font-size: 1.4em; font-weight: bold; color: #1a1a1a; }
    .badge { padding: 6px 12px; border-radius: 20px; color: white; font-weight: bold; font-size: 0.9em; }
    .badge.看多 { background: #ff4d4d; }
    .badge.看空 { background: #00cc66; }
    .badge.觀望 { background: #888; }
    .data-row { display: flex; gap: 15px; margin-bottom: 15px; color: #666; font-size: 0.9em; }
    .comment-box { background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #ddd; }
    .comment-box.看多 { border-left-color: #ff4d4d; }
    .comment-box.看空 { border-left-color: #00cc66; }
    .footer { text-align: center; color: #aaa; margin-top: 30px; font-size: 0.8em; }
</style>
</head>
<body>
    <h1>📈 AI 每日股市戰報 <br><span style="font-size:0.5em; color:#888">{{ date }}</span></h1>
    
    {% for r in results %}
    <div class="card">
        <div class="header">
            <span class="symbol">{{ r.symbol }}</span>
            <div class="badge {{ r.signal }}">{{ r.signal }}</div>
        </div>
        <div class="data-row">
            <span>收盤: <b>{{ r.price }}</b></span>
            <span>RSI: <b>{{ r.rsi }}</b></span>
        </div>
        <div class="comment-box {{ r.signal }}">
            🤖 <b>AI：</b>{{ r.comment }}
        </div>
    </div>
    {% endfor %}
    <div class="footer">Generated by GitHub Actions & Gemini</div>
</body>
</html>
"""

template = Template(html_template)
html_output = template.render(results=results, date=datetime.now().strftime('%Y-%m-%d'))

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_output)

# === 發送 Line 通知 ===
bull_count = len([x for x in results if x['signal']=='看多'])
bear_count = len([x for x in results if x['signal']=='看空'])
page_url = f"https://{GITHUB_USER}.github.io/{REPO_NAME}/"

msg = f"\n📊 {datetime.now().strftime('%m/%d')} 股市戰報已生成！\n"
msg += f"🔴 看多：{bull_count} 檔\n"
msg += f"🟢 看空：{bear_count} 檔\n"
msg += f"⚪ 觀望：{len(results) - bull_count - bear_count} 檔\n\n"
msg += f"👉 點擊查看完整圖表：\n{page_url}"

requests.post("https://notify-api.line.me/api/notify", 
              headers={"Authorization": f"Bearer {LINE_TOKEN}"}, 
              data={"message": msg})
