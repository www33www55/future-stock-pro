import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator
import twstock

st.set_page_config(layout="wide", page_title="台股智慧選股器")

# --- 核心運算模組 ---
def get_stock_data(ticker):
    # 獲取約 120 天數據以計算指標
    df = yf.download(f"{ticker}.TW", period="6mo", progress=False)
    return df

def analyze_stock(df):
    if len(df) < 60: return None
    
    c = df['Close']
    h = df['High']
    l = df['Low']
    v = df['Volume']
    
    score = 0
    
    # 1. 60日新高 (20分)
    if c.iloc[-1] >= c.rolling(60).max().iloc[-1]: score += 20
    
    # 2. MACD 主升段 (15分)
    macd = MACD(c)
    if macd.macd().iloc[-1] > macd.macd_signal().iloc[-1] and macd.macd().iloc[-1] > 0: score += 15
    
    # 3. 月線上彎 (10分)
    sma20 = SMAIndicator(c, window=20).sma_indicator()
    if sma20.iloc[-1] > sma20.iloc[-2]: score += 10
    
    # 4. 量能放大 (15分) - 較過去5日均量放大
    if v.iloc[-1] > v.rolling(5).mean().iloc[-1] * 1.5: score += 15
    
    # 5. KD 過熱/背離 (扣分)
    stoch = StochasticOscillator(h, l, c)
    if stoch.stoch().iloc[-1] > 80: score -= 10
    if (stoch.stoch().iloc[-1] < stoch.stoch().iloc[-2]) and (c.iloc[-1] > c.iloc[-2]): score -= 20
    
    # 6. RSI 過熱 (扣分)
    rsi = RSIIndicator(c).rsi()
    if rsi.iloc[-1] > 70: score -= 10
    
    # 停損與目標價
    atr = (h - l).rolling(14).mean().iloc[-1]
    stop_loss = c.iloc[-1] - (atr * 2)
    target1 = c.iloc[-1] + (atr * 3)
    target2 = c.iloc[-1] + (atr * 5)
    
    return {"分數": score, "停損價": round(stop_loss, 2), "目標價1": round(target1, 2), "目標價2": round(target2, 2)}

# --- UI 介面 ---
st.title("📊 台股即時智慧選股器")

if st.button("開始全市場掃描"):
    # 獲取上市櫃代碼
    stocks = twstock.codes.keys() 
    # 為節省 Demo 時間，這裡取前 20 檔作為範例，實測可移除 [:20]
    target_stocks = [s for s in stocks if len(s) == 4][:20]
    
    progress = st.progress(0)
    results = []
    
    for i, ticker in enumerate(target_stocks):
        df = get_stock_data(ticker)
        if not df.empty:
            analysis = analyze_stock(df)
            if analysis:
                analysis["股號"] = ticker
                results.append(analysis)
        progress.progress((i + 1) / len(target_stocks))
    
    df_result = pd.DataFrame(results).sort_values(by="分數", ascending=False)
    st.table(df_result)

