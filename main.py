import os
import json
import time
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from jinja2 import Template
from google import genai

# ===========================
# 🔧 使用者設定
# ===========================
TZ = ZoneInfo("Asia/Taipei")

TARGET_STOCKS = ["2330.TW", "2317.TW", "0050.TW", "NVDA", "AAPL"]
STOCK_NAMES_ZH = {
    "2330.TW": "台積電",
    "2317.TW": "鴻海",
    "0050.TW": "元大台灣50",
    "NVDA": "輝達",
    "AAPL": "蘋果",
}

GEMINI_MODEL = "gemma-3-27b-it"

# 顯示最近幾根K線（交易日）
CHART_BARS = 120

# 預測：顯示在圖上 10 天；文字給 5/10/30
PRED_DAYS_ON_CHART = 10
PRED_HORIZONS = [5, 10, 30]
PRED_LOOKBACK_DAYS = 60

# 風險：計算窗口
RISK_WINDOW = 120

# 轉折：分數條件
TURN_SCORE_WINDOW_SLOPE = 10

# 歷史資料範圍（同時給：勝率、beta/corr、風險…）
HIST_PERIOD = "2y"

# GitHub Pages URL
GITHUB_USER = os.getenv("GITHUB_USER", "wwwibf2014")
REPO_NAME = os.getenv("REPO_NAME", "daily-stock-ai")

# 大盤
MARKET_INDICES = [
    {"symbol": "^TWII", "name_zh": "台股加權指數", "market": "TW"},
    {"symbol": "^GSPC", "name_zh": "標普500（S&P 500）", "market": "US"},
    {"symbol": "^IXIC", "name_zh": "那斯達克（NASDAQ）", "market": "US"},
]
BENCHMARK_FOR = {
    "TW": {"symbol": "^TWII", "name_zh": "台股加權指數"},
    "US": {"symbol": "^GSPC", "name_zh": "標普500（S&P 500）"},
}

# ===========================
# 工具
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

def market_of_symbol(symbol: str) -> str:
    return "TW" if symbol.upper().endswith(".TW") else "US"

def fmt_prob(p):
    if p is None:
        return "資料不足"
    return f"{int(round(p*100))}%"

# ===========================
# 資料抓取
# ===========================
def fetch_history(symbol: str, period=HIST_PERIOD, retries=3) -> pd.DataFrame:
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

            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            return df
        except Exception as e:
            last_err = e
            time.sleep(1.2 * i)
    raise RuntimeError(f"{symbol} 抓取最終失敗：{last_err}")

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

    # 乖離率
    df["20日乖離率(%)"] = (df["Close"] / df["20日均線"] - 1) * 100

    # 量
    if "Volume" in df.columns:
        df["20日均量"] = df["Volume"].rolling(20).mean()
        df["均量比(今日/20日)"] = df["Volume"] / df["20日均量"]

    return df

# ===========================
# ① 風險預測：ATR% / VaR / MDD / 紀律線
# ===========================
def calc_atr(df: pd.DataFrame, n=14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    return atr

def calc_mdd(close: pd.Series) -> float:
    # 最大回落（負值）
    peak = close.cummax()
    dd = close / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0

def calc_var95(returns: pd.Series) -> float:
    # 95% VaR：用 5%分位數（通常是負值）
    if returns is None or len(returns.dropna()) < 30:
        return np.nan
    return float(np.nanquantile(returns.dropna().values, 0.05))

def risk_level(atr_pct, var95, mdd):
    """
    atr_pct: 正值（%）
    var95: 通常負值（如 -0.02 = -2%）
    mdd: 負值（如 -0.15 = -15%）
    """
    score = 0
    # ATR%
    if atr_pct >= 4.0: score += 2
    elif atr_pct >= 2.0: score += 1

    # VaR（越負越危險）
    if var95 <= -0.03: score += 2
    elif var95 <= -0.02: score += 1

    # MDD（越負越危險）
    if mdd <= -0.25: score += 2
    elif mdd <= -0.15: score += 1

    if score >= 4:
        return "🔴 高"
    elif score >= 2:
        return "🟡 中"
    else:
        return "🟢 低"

def calc_trailing_stop(df: pd.DataFrame, atr: pd.Series):
    """
    紀律線（教學版）：近20日最低 - 0.5*ATR
    """
    low20 = df["Low"].rolling(20).min()
    stop = low20 - 0.5 * atr
    return stop

# ===========================
# ② 轉折點：轉折分數（0~100）+ 訊號
# ===========================
def turning_score(df: pd.DataFrame):
    df = df.copy().dropna()
    if len(df) < 80:
        return None, "資料不足"

    last = df.iloc[-1]

    score = 0
    reasons = []

    # 1) 收盤在 20MA 上
    if pd.notna(last["20日均線"]) and last["Close"] > last["20日均線"]:
        score += 20; reasons.append("收盤在20日均線上（偏強）")
    else:
        reasons.append("收盤在20日均線下（偏弱）")

    # 2) 20MA 在 60MA 上
    if pd.notna(last["60日均線"]) and pd.notna(last["20日均線"]) and last["20日均線"] > last["60日均線"]:
        score += 20; reasons.append("20MA在60MA上（趨勢偏多）")
    else:
        reasons.append("20MA不在60MA上（趨勢未偏多）")

    # 3) 20MA 斜率（近10日）
    ma20 = df["20日均線"].dropna()
    if len(ma20) >= TURN_SCORE_WINDOW_SLOPE + 1:
        slope = ma20.iloc[-1] - ma20.iloc[-(TURN_SCORE_WINDOW_SLOPE+1)]
        if slope > 0:
            score += 20; reasons.append("20MA上揚（趨勢升溫）")
        else:
            reasons.append("20MA走平/下彎（趨勢保守）")
    else:
        reasons.append("20MA資料不足（斜率略過）")

    # 4) MACD柱狀體：是否轉正或走升
    hist = df["MACD柱狀體"].dropna()
    if len(hist) >= 2:
        if hist.iloc[-1] > 0:
            score += 20; reasons.append("MACD柱狀體為正（動能偏多）")
        elif hist.iloc[-1] > hist.iloc[-2]:
            score += 10; reasons.append("MACD柱狀體回升（動能改善）")
        else:
            reasons.append("MACD柱狀體偏弱（動能不足）")
    else:
        reasons.append("MACD資料不足")

    # 5) 量能配合：均量比 > 1
    if "均量比(今日/20日)" in df.columns and pd.notna(last["均量比(今日/20日)"]):
        vr = float(last["均量比(今日/20日)"])
        if vr > 1.0:
            score += 20; reasons.append("量能大於均量（有力氣）")
        else:
            reasons.append("量能偏小（力氣不足）")
    else:
        reasons.append("量能資料不足")

    # 分類
    if score >= 70:
        label = "偏多轉折（可觀察）"
    elif score <= 40:
        label = "轉弱風險（需留意）"
    else:
        label = "整理觀察"

    return score, label

# ===========================
# ③ 持有期優化（教學版）
# ===========================
def holding_plan(turn_score, risk_lv):
    """
    用 轉折分數 + 風險燈號，給新手一個「持有框架」
    """
    if turn_score is None:
        return {"range": "資料不足", "hold": "資料不足", "warn": "資料不足", "rule": "資料不足"}

    is_red = str(risk_lv).startswith("🔴")
    if turn_score >= 70 and not is_red:
        plan = {
            "range": "10~30 個交易日",
            "hold": "偏向波段持有（趨勢較完整）",
            "warn": "若跌破20MA或紀律線，代表轉弱需提高警覺",
            "rule": "續抱條件：收盤維持在20MA上方"
        }
    elif 40 < turn_score < 70 and not is_red:
        plan = {
            "range": "5~10 個交易日",
            "hold": "續抱觀察（等待趨勢更明確）",
            "warn": "若量增跌破20MA，代表轉弱訊號更明顯",
            "rule": "觀察重點：量能是否配合、MACD是否持續改善"
        }
    else:
        plan = {
            "range": "保守（先保護資本）",
            "hold": "以風險控管為優先（先觀察再說）",
            "warn": "風險偏高時，避免硬抱；用紀律線保護資本",
            "rule": "風險優先：跌破紀律線 → 風險升高（教學警示）"
        }
    return plan

# ===========================
# 相關係數（20日）+ Beta（60日）
# ===========================
def compute_corr_beta(stock_df: pd.DataFrame, bench_df: pd.DataFrame, corr_window=20, beta_window=60):
    s = stock_df[["Close"]].rename(columns={"Close": "stock"})
    b = bench_df[["Close"]].rename(columns={"Close": "bench"})

    merged = s.join(b, how="inner").dropna()
    if len(merged) < max(corr_window, beta_window) + 5:
        return None, None

    ret = merged.pct_change().dropna()
    if len(ret) < max(corr_window, beta_window):
        return None, None

    corr20 = ret["stock"].tail(corr_window).corr(ret["bench"].tail(corr_window))

    tail_beta = ret.tail(beta_window)
    var_b = tail_beta["bench"].var()
    if var_b == 0 or pd.isna(var_b):
        beta60 = None
    else:
        beta60 = tail_beta["stock"].cov(tail_beta["bench"]) / var_b

    if pd.isna(corr20):
        corr20 = None
    if beta60 is not None and pd.isna(beta60):
        beta60 = None

    return corr20, beta60

# ===========================
# 預測線 + 區間（統計型情境預測）
# ===========================
def make_forecast(close_series: pd.Series, horizons=PRED_HORIZONS, lookback=PRED_LOOKBACK_DAYS):
    s = close_series.dropna()
    if len(s) < lookback + 5:
        return None

    last = float(s.iloc[-1])
    r = np.log(s / s.shift(1)).dropna().tail(lookback)
    mu = float(r.mean())
    sigma = float(r.std(ddof=1)) if len(r) > 2 else 0.0

    z = 1.0  # 約 68% 區間
    out = {"last": last, "mu": mu, "sigma": sigma, "points": {}}

    for h in horizons:
        mid = last * float(np.exp(mu * h))
        upper = last * float(np.exp(mu * h + z * sigma * np.sqrt(h)))
        lower = last * float(np.exp(mu * h - z * sigma * np.sqrt(h)))
        out["points"][h] = {"mid": round(mid, 2), "upper": round(upper, 2), "lower": round(lower, 2)}
    return out

# ===========================
# 上漲機率：條件統計勝率（保留原本）
# ===========================
def conditional_up_probability(df: pd.DataFrame, horizon_days: int):
    df = df.copy().dropna(subset=["Close", "20日均線", "相對強弱指標RSI(14)", "MACD柱狀體"])
    if "均量比(今日/20日)" not in df.columns:
        return None, 0
    if len(df) < 120:
        return None, 0

    cur = df.iloc[-1]
    cond_close_ma = bool(cur["Close"] > cur["20日均線"])
    cond_macd = bool(cur["MACD柱狀體"] > 0)

    rsi = float(cur["相對強弱指標RSI(14)"])
    if rsi < 40: rsi_bin = "low"
    elif rsi <= 60: rsi_bin = "mid"
    else: rsi_bin = "high"

    vr = float(cur["均量比(今日/20日)"])
    cond_vr = bool(vr > 1.0)

    hist = df.iloc[:-horizon_days].copy()
    if len(hist) < 120:
        return None, 0

    hist["rsi_bin"] = np.where(hist["相對強弱指標RSI(14)"] < 40, "low",
                        np.where(hist["相對強弱指標RSI(14)"] <= 60, "mid", "high"))
    hist["cond_close_ma"] = hist["Close"] > hist["20日均線"]
    hist["cond_macd"] = hist["MACD柱狀體"] > 0
    hist["cond_vr"] = hist["均量比(今日/20日)"] > 1.0

    mask = (
        (hist["cond_close_ma"] == cond_close_ma) &
        (hist["cond_macd"] == cond_macd) &
        (hist["rsi_bin"] == rsi_bin) &
        (hist["cond_vr"] == cond_vr)
    )
    sample = hist[mask]
    n = int(len(sample))
    if n < 25:
        return None, n

    future_close = df["Close"].shift(-horizon_days)
    ret_h = (future_close / df["Close"] - 1.0)
    wins = (ret_h.loc[sample.index] > 0).sum()
    prob = float(wins / n) if n else None
    return prob, n

# ===========================
# 箭頭訊號：均線/MACD/RSI/量
# ===========================
def detect_markers(df: pd.DataFrame):
    df = df.dropna().copy()
    if len(df) < 80:
        return []

    markers = []
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # MA 交叉
    if pd.notna(prev["20日均線"]) and pd.notna(prev["60日均線"]) and pd.notna(last["20日均線"]) and pd.notna(last["60日均線"]):
        if prev["20日均線"] <= prev["60日均線"] and last["20日均線"] > last["60日均線"]:
            markers.append({"type": "up", "text": "均線黃金交叉（偏多）"})
        elif prev["20日均線"] >= prev["60日均線"] and last["20日均線"] < last["60日均線"]:
            markers.append({"type": "down", "text": "均線死亡交叉（轉弱）"})

    # MACD 交叉
    if pd.notna(prev["平滑異同移動平均線MACD"]) and pd.notna(prev["MACD訊號線"]) and pd.notna(last["平滑異同移動平均線MACD"]) and pd.notna(last["MACD訊號線"]):
        if prev["平滑異同移動平均線MACD"] <= prev["MACD訊號線"] and last["平滑異同移動平均線MACD"] > last["MACD訊號線"]:
            markers.append({"type": "up", "text": "MACD翻多（動能轉強）"})
        elif prev["平滑異同移動平均線MACD"] >= prev["MACD訊號線"] and last["平滑異同移動平均線MACD"] < last["MACD訊號線"]:
            markers.append({"type": "down", "text": "MACD翻空（動能轉弱）"})

    # RSI 警告
    rsi = float(last["相對強弱指標RSI(14)"])
    if rsi >= 70:
        markers.append({"type": "warn", "text": "RSI偏熱（注意追高風險）"})
    elif rsi <= 30:
        markers.append({"type": "warn", "text": "RSI偏冷（可能超賣）"})

    # 量增靠近前高
    if "均量比(今日/20日)" in df.columns and pd.notna(last["均量比(今日/20日)"]):
        vr = float(last["均量比(今日/20日)"])
        hi20 = df["High"].tail(20).max()
        if vr > 1.5 and float(last["Close"]) >= 0.98 * float(hi20):
            markers.append({"type": "up", "text": "量增靠近前高（有力氣）"})

    markers = markers[:4]
    label = df.index[-1].strftime("%Y-%m-%d")
    for m in markers:
        m["at_label"] = label
    return markers

# ===========================
# 圖表資料（含預測線/區間/紀律線/箭頭）
# ===========================
def build_chart_data(df_ind: pd.DataFrame, forecast: dict | None, markers: list, trailing_stop: pd.Series | None):
    tail = df_ind.tail(CHART_BARS).copy()
    hist_labels = [d.strftime("%Y-%m-%d") for d in tail.index]

    data = {
        "labels": hist_labels,
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
        "markers": markers or [],
    }

    # 紀律線
    if trailing_stop is not None:
        ts = trailing_stop.reindex(tail.index)
        data["trail_stop"] = [None if pd.isna(x) else float(x) for x in ts]
    else:
        data["trail_stop"] = [None] * len(hist_labels)

    # 預測（10天）
    if forecast is not None and "last" in forecast:
        last_date = tail.index[-1]
        future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=PRED_DAYS_ON_CHART)
        future_labels = [d.strftime("%Y-%m-%d") for d in future_dates]

        mu = float(forecast["mu"])
        sigma = float(forecast["sigma"])
        last_price = float(forecast["last"])
        z = 1.0

        pred_mid = []
        pred_upper = []
        pred_lower = []
        for i in range(1, PRED_DAYS_ON_CHART + 1):
            mid = last_price * float(np.exp(mu * i))
            upper = last_price * float(np.exp(mu * i + z * sigma * np.sqrt(i)))
            lower = last_price * float(np.exp(mu * i - z * sigma * np.sqrt(i)))
            pred_mid.append(round(mid, 2))
            pred_upper.append(round(upper, 2))
            pred_lower.append(round(lower, 2))

        full_labels = hist_labels + future_labels
        none_hist = [None] * len(hist_labels)

        data["labels"] = full_labels
        data["pred_mid"] = none_hist + pred_mid
        data["pred_upper"] = none_hist + pred_upper
        data["pred_lower"] = none_hist + pred_lower

        # 預測區間的「起點連接」：讓線看起來更連續（可選）
        # 這裡不做硬連接，保持清楚：未來才開始畫
    else:
        data["pred_mid"] = [None] * len(hist_labels)
        data["pred_upper"] = [None] * len(hist_labels)
        data["pred_lower"] = [None] * len(hist_labels)

    return data

# ===========================
# 大盤分析（AI只說環境，不講買賣）
# ===========================
def analyze_market_index(client: genai.Client, symbol: str, name_zh: str):
    df_raw = fetch_history(symbol, period=HIST_PERIOD, retries=3)
    df = calculate_indicators(df_raw)

    # 風險（同樣可算）
    atr = calc_atr(df, 14)
    atr_pct = (atr / df["Close"] * 100).iloc[-1] if len(atr.dropna()) else np.nan
    ret = df["Close"].pct_change().dropna().tail(RISK_WINDOW)
    var95 = calc_var95(ret)
    mdd = calc_mdd(df["Close"].tail(RISK_WINDOW))
    risk_lv = risk_level(float(atr_pct) if not pd.isna(atr_pct) else 0.0, float(var95) if not pd.isna(var95) else 0.0, float(mdd))

    # 轉折
    tscore, tlabel = turning_score(df)

    # 紀律線
    trail = calc_trailing_stop(df, atr)

    latest = df.iloc[-1]
    open_ = nz(latest.get("Open"), 0.0)
    high = nz(latest.get("High"), 0.0)
    low = nz(latest.get("Low"), 0.0)
    close = nz(latest.get("Close"), 0.0)
    ma20 = nz(latest.get("20日均線"), 0.0)
    ma60 = nz(latest.get("60日均線"), 0.0)
    rsi = nz(latest.get("相對強弱指標RSI(14)"), 50.0)
    macd_hist = nz(latest.get("MACD柱狀體"), 0.0)
    vol = nz(latest.get("Volume"), 0.0)
    vr = nz(latest.get("均量比(今日/20日)"), 0.0)

    prompt = f"""
你是「給完全新手看的市場老師」。請用繁體中文、非常白話。
只講氣氛/順風逆風，不要講買賣建議，不要用保證。

指數：{symbol}（{name_zh}）
今日K線：開 {open_:.2f} / 高 {high:.2f} / 低 {low:.2f} / 收 {close:.2f}
20日均線：{ma20:.2f}
60日均線：{ma60:.2f}
RSI(14)：{rsi:.2f}
MACD柱狀體：{macd_hist:.4f}
成交量：{vol:.0f}
均量比：{vr:.2f}

風險：ATR%(14) {float(atr_pct):.2f}｜VaR95 {float(var95)*100:.2f}%｜MDD {float(mdd)*100:.2f}%｜風險燈號 {risk_lv}
轉折分數：{tscore if tscore is not None else "資料不足"}（{tlabel}）

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

    forecast = make_forecast(df["Close"])
    markers = detect_markers(df)
    chart_data = build_chart_data(df, forecast, markers, trail)

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
        "macd_hist_now": round(macd_hist, 4),

        "volume_now": int(vol),
        "vr_now": round(vr, 2),

        "risk_lv": risk_lv,
        "atr_pct": None if pd.isna(atr_pct) else round(float(atr_pct), 2),
        "var95_pct": None if pd.isna(var95) else round(float(var95) * 100, 2),
        "mdd_pct": round(float(mdd) * 100, 2),
        "turn_score": tscore,
        "turn_label": tlabel,

        "forecast_points": forecast["points"] if forecast else None,
        "chart_data": json.dumps(chart_data, ensure_ascii=False),
    }

# ===========================
# 個股分析（AI整合：風險/轉折/持有）
# ===========================
def analyze_stock(client: genai.Client, symbol: str, market_context: dict, benchmark_df: pd.DataFrame, benchmark_name_zh: str):
    stock_df_raw = fetch_history(symbol, period=HIST_PERIOD, retries=3)
    df = calculate_indicators(stock_df_raw)
    latest = df.iloc[-1]

    # 指標
    open_ = nz(latest.get("Open"), 0.0)
    high = nz(latest.get("High"), 0.0)
    low = nz(latest.get("Low"), 0.0)
    close = nz(latest.get("Close"), 0.0)

    rsi = nz(latest.get("相對強弱指標RSI(14)"), 50.0)
    ma20 = nz(latest.get("20日均線"), 0.0)
    ma60 = nz(latest.get("60日均線"), 0.0)

    macd = nz(latest.get("平滑異同移動平均線MACD"), 0.0)
    macd_sig = nz(latest.get("MACD訊號線"), 0.0)
    macd_hist = nz(latest.get("MACD柱狀體"), 0.0)

    vol = nz(latest.get("Volume"), 0.0)
    vol_ma20 = nz(latest.get("20日均量"), 0.0)
    vr = nz(latest.get("均量比(今日/20日)"), 0.0)
    bias20 = nz(latest.get("20日乖離率(%)"), 0.0)

    # 相關/ Beta
    corr20, beta60 = compute_corr_beta(stock_df_raw, benchmark_df, corr_window=20, beta_window=60)

    # 預測（統計型）
    forecast = make_forecast(df["Close"])
    pred_points = forecast["points"] if forecast else None

    # 機率（保留）
    prob3, n3 = conditional_up_probability(df, 3)
    prob5, n5 = conditional_up_probability(df, 5)
    prob10, n10 = conditional_up_probability(df, 10)

    # 風險（ATR%、VaR、MDD、紀律線）
    atr = calc_atr(df, 14)
    atr_last = atr.iloc[-1] if len(atr.dropna()) else np.nan
    atr_pct = (atr_last / close * 100) if (not pd.isna(atr_last) and close != 0) else np.nan

    ret = df["Close"].pct_change().dropna().tail(RISK_WINDOW)
    var95 = calc_var95(ret)  # daily return quantile
    mdd = calc_mdd(df["Close"].tail(RISK_WINDOW))
    risk_lv = risk_level(float(atr_pct) if not pd.isna(atr_pct) else 0.0, float(var95) if not pd.isna(var95) else 0.0, float(mdd))

    trail = calc_trailing_stop(df, atr)
    trail_now = trail.iloc[-1] if len(trail.dropna()) else np.nan

    # 轉折
    tscore, tlabel = turning_score(df)

    # 持有計畫
    plan = holding_plan(tscore, risk_lv)

    # 箭頭
    markers = detect_markers(df)

    # 市場摘要（縮短）
    tw = market_context.get("TWII", {})
    us_sp = market_context.get("GSPC", {})
    us_nq = market_context.get("IXIC", {})

    # 給 AI 做「白話總結」
    pred_txt = "資料不足"
    if pred_points:
        p5 = pred_points.get(5); p10 = pred_points.get(10); p30 = pred_points.get(30)
        pred_txt = f"5天：{p5['mid']}（{p5['lower']}~{p5['upper']}）｜10天：{p10['mid']}（{p10['lower']}~{p10['upper']}）｜30天：{p30['mid']}（{p30['lower']}~{p30['upper']}）"

    corr_txt = "資料不足" if corr20 is None else f"{corr20:.2f}"
    beta_txt = "資料不足" if beta60 is None else f"{beta60:.2f}"

    prompt = f"""
你是「給完全新手＋長期投資者」看的股市老師。請用非常白話的繁體中文。
重要：不要用「保證」「一定」「建議買賣」，只能用「可能」「傾向」「需要觀察」。
請把內容分成三段：風險（保命）→ 轉折（抓波段）→ 持有（安心紀律）。

個股：{symbol}（{STOCK_NAMES_ZH.get(symbol,"")}）
今日K線：開 {open_:.2f} / 高 {high:.2f} / 低 {low:.2f} / 收 {close:.2f}
20日均線：{ma20:.2f}｜60日均線：{ma60:.2f}
RSI(14)：{rsi:.2f}
MACD：{macd:.4f}｜訊號線：{macd_sig:.4f}｜柱狀體：{macd_hist:.4f}
成交量：{vol:.0f}｜20日均量：{vol_ma20:.0f}｜均量比：{vr:.2f}
20日乖離率(%)：{bias20:.2f}

風險指標（教學）：
ATR%(14)：{(float(atr_pct) if not pd.isna(atr_pct) else 0.0):.2f}%
VaR95（最糟一天可能跌幅，歷史統計）：{(float(var95)*100 if not pd.isna(var95) else 0.0):.2f}%
MDD（近120日最慘回落）：{float(mdd)*100:.2f}%
風險燈號：{risk_lv}
紀律線（追蹤停損）：{(float(trail_now) if not pd.isna(trail_now) else 0.0):.2f}

轉折分數（0~100）：{tscore if tscore is not None else "資料不足"}（{tlabel}）

持有計畫（請照這個模板寫成白話）：
持有期範圍：{plan["range"]}
續抱說法：{plan["hold"]}
觀察警示：{plan["warn"]}
紀律規則：{plan["rule"]}

和大盤的關聯（用日報酬率算）：
20日相關係數：{corr_txt}
60日Beta：{beta_txt}
基準大盤：{benchmark_name_zh}

條件統計上漲機率（不是保證，只是過去相似比例）：
3天：{fmt_prob(prob3)}（樣本 {n3}）
5天：{fmt_prob(prob5)}（樣本 {n5}）
10天：{fmt_prob(prob10)}（樣本 {n10}）

統計型情境預測（目標＋區間）：
{pred_txt}

市場環境摘要：
台股加權：{tw.get("mood","")}，{tw.get("summary","")}
美股S&P500：{us_sp.get("mood","")}，{us_sp.get("summary","")}
美股NASDAQ：{us_nq.get("mood","")}，{us_nq.get("summary","")}

請只回 JSON：
{{
  "signal": "偏多" 或 "偏空" 或 "觀望",
  "risk_text": "40字內：風險一句話（一定提到ATR或VaR其中一個）",
  "turn_text": "40字內：轉折一句話（一定提到分數+原因一個）",
  "hold_text": "50字內：持有期與紀律一句話（一定提到20MA或紀律線）",
  "tips": ["新手提示1(20字內)","新手提示2(20字內)","新手提示3(20字內)"],
  "market_link": "60字內：把相關係數或Beta講成白話（大盤影響程度）"
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

    # 圖表資料
    chart_data = build_chart_data(df, forecast, markers, trail)

    return {
        "symbol": symbol,
        "name_zh": STOCK_NAMES_ZH.get(symbol, ""),
        "signal": signal,

        "risk_lv": risk_lv,
        "atr_pct": None if pd.isna(atr_pct) else round(float(atr_pct), 2),
        "var95_pct": None if pd.isna(var95) else round(float(var95) * 100, 2),
        "mdd_pct": round(float(mdd) * 100, 2),
        "trail_now": None if pd.isna(trail_now) else round(float(trail_now), 2),

        "turn_score": tscore,
        "turn_label": tlabel,
        "plan_range": plan["range"],
        "plan_hold": plan["hold"],
        "plan_warn": plan["warn"],
        "plan_rule": plan["rule"],

        "risk_text": str(data.get("risk_text", "")).strip(),
        "turn_text": str(data.get("turn_text", "")).strip(),
        "hold_text": str(data.get("hold_text", "")).strip(),
        "tips": [str(x).strip() for x in tips[:3]],
        "market_link": str(data.get("market_link", "")).strip(),

        "benchmark_name_zh": benchmark_name_zh,
        "corr20": None if corr20 is None else round(float(corr20), 2),
        "beta60": None if beta60 is None else round(float(beta60), 2),

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

        "prob3": None if prob3 is None else round(float(prob3), 3),
        "prob5": None if prob5 is None else round(float(prob5), 3),
        "prob10": None if prob10 is None else round(float(prob10), 3),
        "n3": n3, "n5": n5, "n10": n10,

        "forecast_points": pred_points,
        "chart_data": json.dumps(chart_data, ensure_ascii=False),
    }

# ===========================
# LINE 推播
# ===========================
def line_push(line_token: str, to_id: str, msg: str):
    r = requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={"Authorization": f"Bearer {line_token}", "Content-Type": "application/json"},
        json={"to": to_id, "messages": [{"type": "text", "text": msg}]},
        timeout=20,
    )
    if r.status_code >= 300:
        raise RuntimeError(f"LINE 推播失敗 {r.status_code}: {r.text[:200]}")

def build_line_report(stock_results, page_url):
    lines = []
    lines.append(f"🔔 長期三件套戰報（{datetime.now(TZ).strftime('%m/%d')}）")
    lines.append("風險→轉折→持有（教學版，不保證）")

    bull = sum(1 for x in stock_results if x["signal"] == "偏多")
    bear = sum(1 for x in stock_results if x["signal"] == "偏空")
    watch = sum(1 for x in stock_results if x["signal"] == "觀望")
    lines.append(f"個股：偏多{bull}｜觀望{watch}｜偏空{bear}")

    for r in stock_results[:5]:
        name = f"{r['symbol']}{('（'+r['name_zh']+'）') if r.get('name_zh') else ''}"
        p5 = "NA" if r["prob5"] is None else f"{int(round(r['prob5']*100))}%"
        risk = r.get("risk_lv", "")
        tscore = r.get("turn_score", None)
        tscore_txt = "NA" if tscore is None else str(tscore)

        # 10天目標
        tgt10 = ""
        if r.get("forecast_points") and 10 in r["forecast_points"]:
            pt = r["forecast_points"][10]
            tgt10 = f"｜10天目標 {pt['mid']}（{pt['lower']}~{pt['upper']}）"

        lines.append(f"\n{name}")
        lines.append(f"信號：{r['signal']}｜風險：{risk}｜轉折分數：{tscore_txt}｜5天機率：{p5}{tgt10}")
        # 持有期一句
        lines.append(f"持有期：{r.get('plan_range','')}｜重點：{r.get('plan_rule','')}")

    lines.append(f"\n👉 網頁：{page_url}")
    return "\n".join(lines)

# ===========================
# HTML（Chart.js 固定版 + canvas plugin K線）
# ===========================
def render_html(market_results, stock_results, errors):
    html_template = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>AI 股市戰報（長期三件套：風險→轉折→持有）</title>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>

<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f3f5f7; padding: 18px; max-width: 1100px; margin: 0 auto; }
  h1 { text-align:center; margin:10px 0 6px; }
  .sub { text-align:center; color:#777; margin-bottom:14px; }

  .panel { background:#fff; border-radius:16px; padding:16px; box-shadow:0 6px 14px rgba(0,0,0,0.06); margin-bottom:14px; }
  .warn { background:#fff3cd; border:1px solid #ffeeba; color:#856404; border-radius:12px; padding:12px; margin-bottom:14px; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size:0.9em; white-space: pre-wrap; }

  .card { background:#fff; border-radius:18px; padding:16px; box-shadow:0 6px 14px rgba(0,0,0,0.06); margin-bottom:14px; }
  .top { display:flex; justify-content:space-between; align-items:center; gap:10px; }
  .title { font-size:1.25em; font-weight:900; }
  .badge { padding:6px 12px; border-radius:16px; color:#fff; font-weight:900; font-size:0.95em; }
  .badge.偏多 { background:#ff4d4d; }
  .badge.偏空 { background:#00b66a; }
  .badge.觀望 { background:#888; }
  .badge.整理 { background:#6c757d; }

  .kline { color:#333; margin-top:6px; }
  .meta { display:flex; flex-wrap:wrap; gap:10px; margin-top:10px; }
  .chip { background:#f7f7f7; padding:6px 10px; border-radius:12px; color:#333; }
  .chip b { font-weight:900; }

  .teachbox { margin-top:12px; background:#f8f9fa; border-radius:14px; padding:12px; border-left:5px solid #ddd; }
  .teach-title { font-weight:900; margin-bottom:6px; }
  .points { margin:8px 0 0; padding-left:18px; color:#444; }
  .points li { margin:4px 0; }

  .grid { display:grid; grid-template-columns:1fr; gap:12px; }
  @media (min-width: 980px){ .grid { grid-template-columns: 1fr 1fr; } }

  .charts { margin-top:12px; background:#fbfbfb; border-radius:14px; padding:12px; }
  .footer { text-align:center; color:#999; margin:18px 0 10px; font-size:0.9em; }

  .hint { color:#555; line-height:1.6; }
</style>
</head>
<body>

<h1>📈 AI 股市戰報（長期三件套：風險→轉折→持有）</h1>
<div class="sub">{{ date }} · {{ model }}</div>

<div class="panel">
  <div style="font-weight:900; font-size:1.05em;">新手超白話（長期投資者）</div>
  <div class="hint">
    ① 先看 <b>風險（保命）</b>：ATR%、VaR、MDD → 風險燈號🟢🟡🔴。<br>
    ② 再看 <b>轉折（抓波段）</b>：轉折分數0~100，分數越高代表趨勢/動能/量能越完整。<br>
    ③ 最後看 <b>持有（安心）</b>：用「20MA + 紀律線」做續抱/警示，讓你有規則可照做。<br>
    ④ 預測線/機率都是「教學工具」：不是保證，只是給你範圍與歷史相似比例。
  </div>
</div>

{% if errors %}
<div class="warn"><b>本次有錯誤</b><div class="mono">{{ errors|join("\n") }}</div></div>
{% endif %}

<!-- 大盤 -->
<div class="panel">
  <div style="font-weight:900; font-size:1.1em; margin-bottom:8px;">🌏 今日市場環境（大盤）</div>
  <div class="grid">
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
        <div class="chip">風險：<b>{{ m.risk_lv }}</b></div>
        <div class="chip">ATR%：<b>{% if m.atr_pct is not none %}{{ m.atr_pct }}{% else %}NA{% endif %}</b></div>
        <div class="chip">VaR95：<b>{% if m.var95_pct is not none %}{{ m.var95_pct }}%{% else %}NA{% endif %}</b></div>
        <div class="chip">MDD：<b>{{ m.mdd_pct }}%</b></div>
        <div class="chip">轉折分數：<b>{% if m.turn_score is not none %}{{ m.turn_score }}{% else %}NA{% endif %}</b></div>
        <div class="chip">轉折：<b>{{ m.turn_label }}</b></div>
      </div>

      <div class="teachbox">
        <div class="teach-title">📌 白話環境說明</div>
        <div>{{ m.summary }}</div>
        {% if m.teach %}
        <ul class="points">{% for p in m.teach %}<li>{{ p }}</li>{% endfor %}</ul>
        {% endif %}
      </div>

      <div class="charts">
        <div class="grid">
          <div><div style="font-weight:900;margin:4px 0 8px;">① K線＋均線（含預測/紀律線/箭頭）</div><canvas id="mk{{ loop.index }}"></canvas></div>
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

              // 影線
              for (let i=0; i<data.labels.length; i++){
                const lab = data.labels[i];
                const o = data.open[i], h = data.high[i], l = data.low[i], c = data.close[i];
                if (o==null || h==null || l==null || c==null) continue;
                const xPos = x.getPixelForValue(lab);
                ctx.beginPath();
                ctx.moveTo(xPos, y.getPixelForValue(h));
                ctx.lineTo(xPos, y.getPixelForValue(l));
                ctx.stroke();
              }

              // 實體
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

          // markers
          const upPts = [], downPts = [], warnPts = [];
          const markers = data.markers || [];
          for (const m of markers){
            const xlab = m.at_label;
            const idx = data.labels.indexOf(xlab);
            const yv = (idx >=0 && data.close[idx]!=null) ? data.close[idx] : null;
            if (yv==null) continue;
            if (m.type==="up") upPts.push({x:xlab, y:yv, t:m.text});
            if (m.type==="down") downPts.push({x:xlab, y:yv, t:m.text});
            if (m.type==="warn") warnPts.push({x:xlab, y:yv, t:m.text});
          }

          new Chart(document.getElementById("mk{{ loop.index }}"), {
            type:"line",
            data:{
              labels:data.labels,
              datasets:[
                { label:"20日均線", data:data.ma20, spanGaps:true },
                { label:"60日均線", data:data.ma60, spanGaps:true },

                { label:"紀律線（追蹤停損）", data:data.trail_stop, spanGaps:true, borderDash:[2,6], pointRadius:0 },

                { label:"預測上界（10天）", data:data.pred_upper, spanGaps:true, borderDash:[6,6], pointRadius:0, borderColor:"rgba(80,80,80,0.35)" },
                { label:"預測下界（10天）", data:data.pred_lower, spanGaps:true, borderDash:[6,6], pointRadius:0, borderColor:"rgba(80,80,80,0.35)" },
                { label:"預測中線（10天）", data:data.pred_mid, spanGaps:true, borderDash:[6,6], pointRadius:0, borderColor:"rgba(40,40,40,0.7)" },

                { type:"scatter", label:"偏多訊號", data:upPts, parsing:false, pointStyle:"triangle", pointRotation:0, pointRadius:7,
                  backgroundColor:"rgba(0,182,106,0.85)", borderColor:"rgba(0,182,106,1)" },
                { type:"scatter", label:"轉弱訊號", data:downPts, parsing:false, pointStyle:"triangle", pointRotation:180, pointRadius:7,
                  backgroundColor:"rgba(255,77,77,0.85)", borderColor:"rgba(255,77,77,1)" },
                { type:"scatter", label:"警告", data:warnPts, parsing:false, pointStyle:"rectRot", pointRadius:7,
                  backgroundColor:"rgba(255,193,7,0.85)", borderColor:"rgba(255,193,7,1)" },
              ]
            },
            options:{
              plugins:{
                legend:{display:true},
                tooltip:{callbacks:{label:(ctx)=>{ const raw=ctx.raw||{}; if(raw.t) return raw.t; return `${ctx.dataset.label}: ${ctx.formattedValue}`; }}}
              },
              scales:{x:{display:false}}
            },
            plugins:[candlePlugin]
          });

          new Chart(document.getElementById("mv{{ loop.index }}"), {
            data:{labels:data.labels, datasets:[
              { type:"bar", label:"成交量", data:data.volume },
              { type:"line", label:"20日均量", data:data.vol_ma20, spanGaps:true },
            ]},
            options:{plugins:{legend:{display:true}}, scales:{x:{display:false}}}
          });

          new Chart(document.getElementById("mrsi{{ loop.index }}"), {
            type:"line",
            data:{labels:data.labels, datasets:[{ label:"RSI(14)", data:data.rsi, spanGaps:true }]},
            options:{plugins:{legend:{display:true}}, scales:{x:{display:false}}}
          });

          new Chart(document.getElementById("mmacd{{ loop.index }}"), {
            data:{labels:data.labels, datasets:[
              { type:"bar", label:"MACD柱狀體", data:data.macd_hist },
              { type:"line", label:"MACD", data:data.macd, spanGaps:true },
              { type:"line", label:"MACD訊號線", data:data.macd_sig, spanGaps:true },
            ]},
            options:{plugins:{legend:{display:true}}, scales:{x:{display:false}}}
          });

        })();
      </script>
    </div>
    {% endfor %}
  </div>
</div>

<!-- 個股 -->
{% for r in stock_results %}
<div class="card">
  <div class="top">
    <div>
      <div class="title">{{ r.symbol }}{% if r.name_zh %}（{{ r.name_zh }}）{% endif %}</div>
      <div class="kline">今日：開 <b>{{ r.open_now }}</b>｜高 <b>{{ r.high_now }}</b>｜低 <b>{{ r.low_now }}</b>｜收 <b>{{ r.price }}</b></div>
      <div class="hint" style="margin-top:6px;">基準大盤：<b>{{ r.benchmark_name_zh }}</b></div>
    </div>
    <div class="badge {{ r.signal }}">{{ r.signal }}</div>
  </div>

  <div class="meta">
    <div class="chip">風險：<b>{{ r.risk_lv }}</b></div>
    <div class="chip">ATR%：<b>{% if r.atr_pct is not none %}{{ r.atr_pct }}{% else %}NA{% endif %}</b></div>
    <div class="chip">VaR95：<b>{% if r.var95_pct is not none %}{{ r.var95_pct }}%{% else %}NA{% endif %}</b></div>
    <div class="chip">MDD：<b>{{ r.mdd_pct }}%</b></div>
    <div class="chip">紀律線：<b>{% if r.trail_now is not none %}{{ r.trail_now }}{% else %}NA{% endif %}</b></div>

    <div class="chip">轉折分數：<b>{% if r.turn_score is not none %}{{ r.turn_score }}{% else %}NA{% endif %}</b></div>
    <div class="chip">轉折：<b>{{ r.turn_label }}</b></div>

    <div class="chip">20MA：<b>{{ r.ma20_now }}</b></div>
    <div class="chip">60MA：<b>{{ r.ma60_now }}</b></div>
    <div class="chip">RSI：<b>{{ r.rsi_now }}</b></div>
    <div class="chip">MACD柱：<b>{{ r.macd_hist_now }}</b></div>
    <div class="chip">均量比：<b>{{ r.vr_now }}</b></div>
    <div class="chip">20日乖離：<b>{{ r.bias20_now }}</b></div>

    <div class="chip">20日相關係數：<b>{% if r.corr20 is not none %}{{ r.corr20 }}{% else %}資料不足{% endif %}</b></div>
    <div class="chip">60日Beta：<b>{% if r.beta60 is not none %}{{ r.beta60 }}{% else %}資料不足{% endif %}</b></div>
  </div>

  <div class="teachbox">
    <div class="teach-title">🛡️ 風險（保命）</div>
    <div>{{ r.risk_text }}</div>
  </div>

  <div class="teachbox">
    <div class="teach-title">🧭 轉折（抓波段）</div>
    <div>{{ r.turn_text }}</div>
  </div>

  <div class="teachbox">
    <div class="teach-title">🧘 持有（安心紀律）</div>
    <div>{{ r.hold_text }}</div>
    <div class="hint" style="margin-top:8px;">
      持有期：<b>{{ r.plan_range }}</b>｜續抱：{{ r.plan_hold }}<br>
      警示：{{ r.plan_warn }}<br>
      規則：{{ r.plan_rule }}
    </div>
  </div>

  {% if r.tips %}
  <div class="teachbox">
    <div class="teach-title">📌 新手重點</div>
    <ul class="points">{% for p in r.tips %}<li>{{ p }}</li>{% endfor %}</ul>
  </div>
  {% endif %}

  <div class="teachbox">
    <div class="teach-title">🌊 大盤 × 個股（白話）</div>
    <div>{{ r.market_link }}</div>
  </div>

  <div class="teachbox">
    <div class="teach-title">🔮 預測（教學工具）</div>
    <div class="hint">3/5/10天上漲機率：{% if r.prob3 is not none %}{{ (r.prob3*100)|round|int }}%{% else %}NA{% endif %} / {% if r.prob5 is not none %}{{ (r.prob5*100)|round|int }}%{% else %}NA{% endif %} / {% if r.prob10 is not none %}{{ (r.prob10*100)|round|int }}%{% else %}NA{% endif %}</div>
    {% if r.forecast_points %}
    <div class="meta" style="margin-top:8px;">
      <div class="chip">5天目標：<b>{{ r.forecast_points[5].mid }}</b>（{{ r.forecast_points[5].lower }}~{{ r.forecast_points[5].upper }}）</div>
      <div class="chip">10天目標：<b>{{ r.forecast_points[10].mid }}</b>（{{ r.forecast_points[10].lower }}~{{ r.forecast_points[10].upper }}）</div>
      <div class="chip">30天目標：<b>{{ r.forecast_points[30].mid }}</b>（{{ r.forecast_points[30].lower }}~{{ r.forecast_points[30].upper }}）</div>
    </div>
    {% endif %}
  </div>

  <div class="charts">
    <div class="grid">
      <div><div style="font-weight:900;margin:4px 0 8px;">① K線＋均線（含預測/紀律線/箭頭）</div><canvas id="k{{ loop.index }}"></canvas></div>
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
            ctx.beginPath();
            ctx.moveTo(xPos, y.getPixelForValue(h));
            ctx.lineTo(xPos, y.getPixelForValue(l));
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

      const upPts = [], downPts = [], warnPts = [];
      const markers = data.markers || [];
      for (const m of markers){
        const xlab = m.at_label;
        const idx = data.labels.indexOf(xlab);
        const yv = (idx >=0 && data.close[idx]!=null) ? data.close[idx] : null;
        if (yv==null) continue;
        if (m.type==="up") upPts.push({x:xlab, y:yv, t:m.text});
        if (m.type==="down") downPts.push({x:xlab, y:yv, t:m.text});
        if (m.type==="warn") warnPts.push({x:xlab, y:yv, t:m.text});
      }

      new Chart(document.getElementById("k{{ loop.index }}"), {
        type:"line",
        data:{
          labels:data.labels,
          datasets:[
            { label:"20日均線", data:data.ma20, spanGaps:true },
            { label:"60日均線", data:data.ma60, spanGaps:true },
            { label:"紀律線（追蹤停損）", data:data.trail_stop, spanGaps:true, borderDash:[2,6], pointRadius:0 },

            { label:"預測上界（10天）", data:data.pred_upper, spanGaps:true, borderDash:[6,6], pointRadius:0, borderColor:"rgba(80,80,80,0.35)" },
            { label:"預測下界（10天）", data:data.pred_lower, spanGaps:true, borderDash:[6,6], pointRadius:0, borderColor:"rgba(80,80,80,0.35)" },
            { label:"預測中線（10天）", data:data.pred_mid, spanGaps:true, borderDash:[6,6], pointRadius:0, borderColor:"rgba(40,40,40,0.7)" },

            { type:"scatter", label:"偏多訊號", data:upPts, parsing:false, pointStyle:"triangle", pointRotation:0, pointRadius:7,
              backgroundColor:"rgba(0,182,106,0.85)", borderColor:"rgba(0,182,106,1)" },
            { type:"scatter", label:"轉弱訊號", data:downPts, parsing:false, pointStyle:"triangle", pointRotation:180, pointRadius:7,
              backgroundColor:"rgba(255,77,77,0.85)", borderColor:"rgba(255,77,77,1)" },
            { type:"scatter", label:"警告", data:warnPts, parsing:false, pointStyle:"rectRot", pointRadius:7,
              backgroundColor:"rgba(255,193,7,0.85)", borderColor:"rgba(255,193,7,1)" },
          ]
        },
        options:{
          plugins:{legend:{display:true}, tooltip:{callbacks:{label:(ctx)=>{ const raw=ctx.raw||{}; if(raw.t) return raw.t; return `${ctx.dataset.label}: ${ctx.formattedValue}`; }}}},
          scales:{x:{display:false}}
        },
        plugins:[candlePlugin]
      });

      new Chart(document.getElementById("v{{ loop.index }}"), {
        data:{labels:data.labels, datasets:[
          { type:"bar", label:"成交量", data:data.volume },
          { type:"line", label:"20日均量", data:data.vol_ma20, spanGaps:true },
        ]},
        options:{plugins:{legend:{display:true}}, scales:{x:{display:false}}}
      });

      new Chart(document.getElementById("rsi{{ loop.index }}"), {
        type:"line",
        data:{labels:data.labels, datasets:[{ label:"RSI(14)", data:data.rsi, spanGaps:true }]},
        options:{plugins:{legend:{display:true}}, scales:{x:{display:false}}}
      });

      new Chart(document.getElementById("macd{{ loop.index }}"), {
        data:{labels:data.labels, datasets:[
          { type:"bar", label:"MACD柱狀體", data:data.macd_hist },
          { type:"line", label:"MACD", data:data.macd, spanGaps:true },
          { type:"line", label:"MACD訊號線", data:data.macd_sig, spanGaps:true },
        ]},
        options:{plugins:{legend:{display:true}}, scales:{x:{display:false}}}
      });
    })();
  </script>
</div>
{% endfor %}

<div class="footer">提醒：此頁為教學示範。風險/轉折/持有皆為規則化解讀，不構成投資建議。</div>

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

# ===========================
# main
# ===========================
def main():
    client = genai.Client(api_key=require_env("GEMINI_API_KEY"))
    line_token = require_env("LINE_TOKEN")
    line_to = require_env("LINE_TO")

    errors = []
    market_results = []
    market_context = {}

    # 大盤
    for idx in MARKET_INDICES:
        try:
            print(f"🌏 分析大盤 {idx['symbol']} ...")
            r = analyze_market_index(client, idx["symbol"], idx["name_zh"])
            market_results.append(r)
            key = idx["symbol"].replace("^", "")
            market_context[key] = {"mood": r["mood"], "summary": r["summary"]}
            time.sleep(0.6)
        except Exception as e:
            errors.append(f"{idx['symbol']}: {e}")
            print(f"❌ 大盤 {idx['symbol']} 失敗：{e}")

    # benchmark cache
    bench_cache = {}
    try:
        for mk, info in BENCHMARK_FOR.items():
            sym = info["symbol"]
            bench_cache[sym] = fetch_history(sym, period=HIST_PERIOD, retries=3)
    except Exception as e:
        errors.append(f"benchmark: {e}")
        print(f"⚠️ benchmark 抓取失敗：{e}")

    # 個股
    stock_results = []
    for s in TARGET_STOCKS:
        try:
            mkt = market_of_symbol(s)
            bm_info = BENCHMARK_FOR[mkt]
            bm_symbol = bm_info["symbol"]
            bm_name = bm_info["name_zh"]

            bm_df = bench_cache.get(bm_symbol)
            if bm_df is None:
                bm_df = fetch_history(bm_symbol, period=HIST_PERIOD, retries=3)

            print(f"🔍 分析 {s} ...（基準：{bm_symbol}）")
            stock_results.append(analyze_stock(client, s, market_context, bm_df, bm_name))
            time.sleep(0.9)
        except Exception as e:
            errors.append(f"{s}: {e}")
            print(f"❌ {s} 失敗：{e}")

    # 產出 index.html
    html = render_html(market_results, stock_results, errors)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    # LINE
    page_url = f"https://{GITHUB_USER}.github.io/{REPO_NAME}/"
    msg = build_line_report(stock_results, page_url)

    try:
        line_push(line_token, line_to, msg)
        print("✅ LINE 推播成功")
    except Exception as e:
        print(f"⚠️ LINE 推播失敗（不影響網頁生成）：{e}")

if __name__ == "__main__":
    main()
