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


# === 技術指標 ===
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


# ✅ 關鍵：用 requests.Session + User-Agent + 重試，避免 Yahoo 擋/暫時失敗
def fetch_history(symbol: str, period: str = "1y", retries: int = 3) -> pd.DataFrame:
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    })

    last_err = None
    for i in range(1, retries + 1):
        try:
            t = yf.Ticker(symbol, session=sess)
            df = t.history(period=period, auto_adjust=False)
            if df is None or df.empty:
                raise RuntimeError("yfinance 回傳空資料（df.empty）")
            if "Close" not in df.columns:
                raise RuntimeError(f"yfinance 欄位異常：{list(df.columns)}")
            return df
        except Exception as e:
            last_err = e
            wait = 1.5 * i
            print(f"⚠️ 抓取失敗 {symbol}（第 {i}/{retries} 次）：{e}，{wait:.1f}s 後重試")
            time.sleep(wait)

    raise RuntimeError(f"{symbol} 抓取最終失敗：{last_err}")


def analyze_stock(client: genai.Client, symbol: str) -> tuple[dict | None, str | None]:
    """
    回傳 (result, error_message)
    """
    print(f"🔍 正在分析 {symbol}...")
    try:
        df = fetch_history(symbol, period="1y", retries=3)
        df = calculate_indicators(df)
        latest = df.iloc[-1]

        close = latest.get("Close")
        if close is None or pd.isna(close):
            return None, "Close 欄位為空/NaN"

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
        data = safe_parse_json((resp.text or "").strip())

        signal = data.get("signal", "觀望")
        reason = str(data.get("reason", "AI 無法分析")).replace("\n", " ").strip()

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
        }, None

    except Exception as e:
        err = f"{symbol}: {e}"
        print(f"❌ {err}")
        return None, err


def render_html(results: list[dict], errors: list[str]) -> str:
    html_template = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>股市 AI 戰情室</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f0f2f5; padding: 20px; max-width: 900px; margin: 0 auto; }
h1 { text-align: center; color: #333; margin-bottom: 10px; }
.sub { text-align:center; color:#888; margin-bottom: 25px; }
.card { background: white; border-radius: 15px; padding: 20px; margin-bottom: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.08); }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.symbol { font-size: 1.25em; font-weight: 700; }
.badge { padding: 6px 12px; border-radius: 18px; color: white; font-weight: 700; }
.badge.看多 { background: #ff4d4d; }
.badge.看空 { background: #00cc66; }
.badge.觀望 { background: #888; }
.comment-box { background: #f8f9fa; padding: 12px; border-radius: 10px; border-left: 4px solid #ddd; }
.comment-box.看多 { border-left-color: #ff4d4d; }
.comment-box.看空 { border-left-color: #00cc66; }
.footer { text-align: center; color: #aaa; margin-top: 22px; font-size: 0.85em; }
.warn { background: #fff3cd; border: 1px solid #ffeeba; color: #856404; border-radius: 12px; padding: 14px; margin-bottom: 16px; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.9em; white-space: pre-wrap; }
</style>
</head>
<body>
  <h1>📈 AI 每日股市戰報</h1>
  <div class="sub">{{ date }} · Generated by GitHub Actions & Gemini</div>

  {% if errors and results|length == 0 %}
    <div class="warn">
      <b>本次抓取全部失敗</b>（所以頁面看起來是空的）。<br>
      下面是錯誤原因（可直接貼回給我，我可以精準判斷是哪個環節被擋）：
      <div class="mono">{{ errors|join("\\n") }}</div>
    </div>
  {% elif errors %}
    <div class="warn">
      <b>部分股票抓取失敗</b>（其餘已正常顯示）。<br>
      <div class="mono">{{ errors|join("\\n") }}</div>
    </div>
  {% endif %}

  {% for r in results %}
  <div class="card">
    <div class="header">
      <span class="symbol">{{ r.symbol }}</span>
      <div class="badge {{ r.signal }}">{{ r.signal }}</div>
    </div>
    <div>收盤：<b>{{ r.price }}</b>　RSI：<b>{{ r.rsi }}</b></div>
    <div class="comment-box {{ r.signal }}" style="margin-top:10px;">🤖 {{ r.comment }}</div>
  </div>
  {% endfor %}

  <div class="footer">Tip：週末手動跑也應該抓得到最後一個交易日的收盤；若全空，通常是 Yahoo 連線被擋或暫時失敗。</div>
</body>
</html>
"""
    template = Template(html_template)
    return template.render(
        results=results,
        errors=errors,
        date=datetime.now(TZ).strftime("%Y-%m-%d"),
    )


def line_push_message(line_token: str, to_id: str, message: str):
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Authorization": f"Bearer {line_token}",
        "Content-Type": "application/json",
    }
    payload = {"to": to_id, "messages": [{"type": "text", "text": message}]}
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"LINE 推播失敗 {r.status_code}: {r.text[:300]}")


def main():
    gemini_key = require_env("GEMINI_API_KEY")
    line_token = require_env("LINE_TOKEN")
    to_id = require_env("LINE_TO")

    github_user = os.getenv("GITHUB_USER", "wwwibf2014")
    repo_name = os.getenv("REPO_NAME", DEFAULT_REPO_NAME)
    page_url = f"https://{github_user}.github.io/{repo_name}/"

    client = genai.Client(api_key=gemini_key)

    results = []
    errors = []

    for stock in TARGET_STOCKS:
        r, err = analyze_stock(client, stock)
        if r:
            results.append(r)
        if err:
            errors.append(err)
        time.sleep(1.2)

    # 產出 HTML（就算全失敗也會顯示錯誤原因）
    html = render_html(results, errors)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    bull = sum(1 for x in results if x["signal"] == "看多")
    bear = sum(1 for x in results if x["signal"] == "看空")
    watch = len(results) - bull - bear

    msg = (
        f"\n📊 {datetime.now(TZ).strftime('%m/%d')} 股市戰報已生成！\n"
        f"🔴 看多：{bull} 檔\n"
        f"🟢 看空：{bear} 檔\n"
        f"⚪ 觀望：{watch} 檔\n"
        f"❗抓取失敗：{len(errors)} 檔\n\n"
        f"👉 查看完整報表：\n{page_url}"
    )

    try:
        line_push_message(line_token, to_id, msg)
        print("✅ LINE 推播成功")
    except Exception as e:
        print(f"⚠️ LINE 推播失敗（不影響部署）：{e}")


if __name__ == "__main__":
    main()
