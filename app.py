import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import twstock
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="未來小股神 Pro", layout="wide")

st.title("🚀 未來小股神 Pro｜RS強度選股器")
st.write("核心邏輯：RS強度＋60日新高＋量能＋主升段＋突破回踩＋N字第二波")

# ======================
# 股票池
# ======================

@st.cache_data
def get_all_tw_stocks():
    rows = []
    for code, info in twstock.codes.items():
        if len(code) == 4 and info.market in ["上市", "上櫃"]:
            suffix = ".TW" if info.market == "上市" else ".TWO"
            rows.append({
                "股號": code,
                "股名": info.name,
                "市場": info.market,
                "代號": code + suffix
            })
    return pd.DataFrame(rows)


# ======================
# 技術指標
# ======================

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_kd(df, period=9):
    low_min = df["Low"].rolling(period).min()
    high_max = df["High"].rolling(period).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    return k, d


def calc_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


# ======================
# 型態判斷
# ======================

def detect_n_pattern(df):
    if len(df) < 50:
        return False

    close = df["Close"]

    left_high = close.iloc[-50:-25].max()
    pullback_low = close.iloc[-25:-8].min()
    recent_close = close.iloc[-1]

    if left_high <= 0:
        return False

    pullback_depth = (left_high - pullback_low) / left_high

    has_pullback = 0.03 <= pullback_depth <= 0.25
    back_near_high = recent_close >= left_high * 0.97

    return has_pullback and back_near_high


def detect_round_bottom(df):
    if len(df) < 80:
        return False

    close = df["Close"]
    ma20 = close.rolling(20).mean()

    left = close.iloc[-80:-55].mean()
    bottom = close.iloc[-55:-25].mean()
    right = close.iloc[-25:].mean()

    bottom_lower = bottom < left * 0.95
    right_recover = right > bottom * 1.05
    ma20_turn_up = ma20.iloc[-1] > ma20.iloc[-10]

    return bottom_lower and right_recover and ma20_turn_up


def detect_pullback_breakout(df):
    if len(df) < 60:
        return False

    close = df["Close"]
    ma20 = df["MA20"]

    old_high = close.iloc[-60:-25].max()
    recent_low = close.iloc[-25:-5].min()
    now = close.iloc[-1]

    breakout_before = close.iloc[-25:].max() > old_high * 1.01
    pullback_not_broken = recent_low >= ma20.iloc[-25:-5].min() * 0.98
    back_up = now >= old_high * 0.98

    return breakout_before and pullback_not_broken and back_up


def detect_kd_top_divergence(df):
    if len(df) < 40:
        return False

    recent = df.tail(40)

    price_now = recent["Close"].iloc[-1]
    price_before = recent["Close"].iloc[:-10].max()

    kd_now = recent["K"].iloc[-1]
    kd_before = recent["K"].iloc[:-10].max()

    return price_now > price_before and kd_now < kd_before


def detect_fake_breakout(df):
    if len(df) < 40:
        return False

    last = df.iloc[-1]
    prev_high = df["Close"].iloc[-40:-1].max()

    upper_shadow = last["High"] - max(last["Close"], last["Open"])
    body = abs(last["Close"] - last["Open"])

    if body == 0:
        body = 0.01

    long_upper = upper_shadow > body * 2
    breakout_fail = last["High"] > prev_high and last["Close"] < prev_high

    return long_upper and breakout_fail


# ======================
# 單股評分 Pro
# ======================

def score_stock(row, market_return_20):
    code = row["代號"]

    try:
        df = yf.download(
            code,
            period="6mo",
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False
        )

        if df.empty or len(df) < 90:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()

        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA60"] = df["Close"].rolling(60).mean()
        df["VOL20"] = df["Volume"].rolling(20).mean()
        df["RSI"] = calc_rsi(df["Close"])
        df["K"], df["D"] = calc_kd(df)
        df["MACD"], df["SIGNAL"], df["HIST"] = calc_macd(df["Close"])

        df = df.dropna()

        if len(df) < 60:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last["Close"])
        score = 0
        reasons = []

        # ======================
        # 第一層：RS強度＋創新高＋量能 40分
        # ======================

        stock_return_20 = (df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1) * 100
        rs_strength = stock_return_20 - market_return_20

        if rs_strength > 8:
            score += 20
            reasons.append("RS強度很強")
        elif rs_strength > 4:
            score += 15
            reasons.append("RS強度強")
        elif rs_strength > 0:
            score += 8
            reasons.append("RS強度略強")

        high_60 = df["Close"].tail(60).max()
        is_60_high = close >= high_60 * 0.99

        if is_60_high:
            score += 10
            reasons.append("接近/創60日新高")

        if last["Volume"] > last["VOL20"] * 2:
            score += 10
            reasons.append("爆量2倍")
        elif last["Volume"] > last["VOL20"] * 1.5:
            score += 8
            reasons.append("量增1.5倍")
        elif last["Volume"] > last["VOL20"]:
            score += 4
            reasons.append("量能增加")

        # ======================
        # 第二層：主升段 30分
        # ======================

        macd_gold = last["MACD"] > last["SIGNAL"]
        macd_above_zero = last["MACD"] > 0
        ma20_up = last["MA20"] > df["MA20"].iloc[-5]

        if macd_above_zero:
            score += 10
            reasons.append("MACD在0軸上")

        if macd_gold:
            score += 10
            reasons.append("MACD黃金交叉")

        if ma20_up:
            score += 10
            reasons.append("月線上彎")

        main_wave = macd_above_zero and macd_gold and ma20_up and is_60_high

        # ======================
        # 第三層：突破回踩 20分
        # ======================

        pullback_breakout = detect_pullback_breakout(df)

        if pullback_breakout:
            score += 20
            reasons.append("突破後回踩不破")

        # ======================
        # 第四層：型態加分 10分
        # ======================

        n_pattern = detect_n_pattern(df)
        round_bottom = detect_round_bottom(df)

        if n_pattern:
            score += 5
            reasons.append("N字第二波")

        if round_bottom:
            score += 5
            reasons.append("圓弧底")

        # ======================
        # 扣分
        # ======================

        kd_div = detect_kd_top_divergence(df)
        fake_breakout = detect_fake_breakout(df)

        if last["RSI"] > 90:
            score -= 20
            reasons.append("RSI極熱扣分")
        elif last["RSI"] > 85:
            score -= 10
            reasons.append("RSI偏熱扣分")

        if last["K"] > 90:
            score -= 10
            reasons.append("KD過熱扣分")

        if kd_div:
            score -= 20
            reasons.append("KD頂背離扣分")

        if fake_breakout:
            score -= 20
            reasons.append("假突破扣分")

        if close < last["MA20"]:
            score -= 15
            reasons.append("跌破月線")

        if close < last["MA60"]:
            score -= 15
            reasons.append("跌破季線")

        score = int(max(0, min(100, score)))

        # 分數校正，避免一堆100
        if score >= 98:
            score = 96
        elif score >= 95:
            score = 94
        elif score >= 90:
            score = 91

        # 真的神級才100
        if (
            rs_strength > 8
            and is_60_high
            and last["Volume"] > last["VOL20"] * 2
            and main_wave
            and pullback_breakout
            and not kd_div
            and not fake_breakout
            and last["RSI"] < 85
        ):
            score = 100

        wave_low = df["Low"].tail(30).min()
        wave_high = df["High"].tail(30).max()
        wave = wave_high - wave_low

        stop_loss = round(close * 0.94, 2)
        target1 = round(close + wave * 0.5, 2)
        target2 = round(close + wave, 2)

        if score >= 95:
            level = "🔥 超強"
        elif score >= 90:
            level = "🟢 強勢"
        elif score >= 85:
            level = "🟡 觀察"
        else:
            level = "—"

        return {
            "股號": row["股號"],
            "股名": row["股名"],
            "市場": row["市場"],
            "最新價": round(close, 2),
            "分數": score,
            "等級": level,
            "RS強度": round(rs_strength, 2),
            "60日新高": "✅" if is_60_high else "—",
            "主升段": "🔥 是" if main_wave else "—",
            "突破回踩": "✅ 是" if pullback_breakout else "—",
            "N字": "🔥 是" if n_pattern else "—",
            "圓弧底": "🌙 是" if round_bottom else "—",
            "停損價": stop_loss,
            "目標1": target1,
            "目標2": target2,
            "理由": "、".join(reasons)
        }

    except Exception:
        return None


# ======================
# 大盤RS基準
# ======================

@st.cache_data
def get_market_return_20():
    try:
        m = yf.download("^TWII", period="3mo", interval="1d", progress=False, auto_adjust=False)
        if isinstance(m.columns, pd.MultiIndex):
            m.columns = m.columns.get_level_values(0)
        m = m.dropna()
        return (m["Close"].iloc[-1] / m["Close"].iloc[-21] - 1) * 100
    except Exception:
        return 0


# ======================
# 介面
# ======================

stocks_df = get_all_tw_stocks()
market_return_20 = get_market_return_20()

st.success(f"股票池已載入：{len(stocks_df)} 檔")
st.info(f"大盤近20日漲跌幅：約 {market_return_20:.2f}%")

scan_mode = st.radio(
    "掃描範圍",
    ["前100檔測試", "自訂股票池", "全上市上櫃"],
    horizontal=True
)

if scan_mode == "前100檔測試":
    scan_df = stocks_df.head(100)

elif scan_mode == "自訂股票池":
    user_input = st.text_area(
        "輸入股號，用逗號分開，例如：2330,2303,2409,2313,3037,6272,3060",
        "2330,2303,2409,2313,3037,6272,3060"
    )
    codes = [x.strip() for x in user_input.split(",") if x.strip()]
    scan_df = stocks_df[stocks_df["股號"].isin(codes)]

else:
    scan_df = stocks_df

min_score = st.slider("最低顯示分數", 70, 100, 85)
max_workers = st.slider("掃描速度", 4, 16, 8)

st.warning("Pro版比較重視RS強度與60日新高。全市場掃描會比較久，手機版請耐心等待。")

if st.button("🔥 開始掃描 Pro"):
    results = []
    total = len(scan_df)

    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(score_stock, row, market_return_20)
            for _, row in scan_df.iterrows()
        ]

        for i, future in enumerate(as_completed(futures)):
            result = future.result()

            if result:
    results.append(result)

            progress.progress((i + 1) / total)
            status.write(f"掃描中：{i + 1}/{total}")

    if results:
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(["分數", "RS強度"], ascending=False).reset_index(drop=True)
        result_df.insert(0, "排名", result_df.index + 1)

        st.subheader("🏆 Pro強勢股排行榜")
        st.dataframe(result_df, use_container_width=True)

        st.subheader("🔥 Pro精選名單")

        top = result_df[result_df["分數"] >= 85]

        for _, r in top.head(30).iterrows():
            st.success(
                f"🏅 {r['股號']}｜{r['股名']}｜{r['分數']}分｜{r['等級']}\n\n"
                f"RS強度：{r['RS強度']}\n\n"
                f"60日新高：{r['60日新高']}\n\n"
                f"主升段：{r['主升段']}\n\n"
                f"突破回踩：{r['突破回踩']}\n\n"
                f"N字：{r['N字']}｜圓弧底：{r['圓弧底']}\n\n"
                f"🎯 停損：{r['停損價']}｜目標1：{r['目標1']}｜目標2：{r['目標2']}\n\n"
                f"理由：{r['理由']}"
            )
    else:
        st.warning("沒有符合條件的股票")
