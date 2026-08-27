import math
import random
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Penny Stock Scanner & Analytics",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
body, .stApp {
    background-color: #0e1117;
    color: #e6e6e6;
}
h1, h2, h3, h4 {
    color: #f2c14e;
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stMetric"] {
    background-color: #161b22;
    border: 1px solid #262d3a;
    border-radius: 10px;
    padding: 12px;
}
[data-testid="stMetricValue"] {
    color: #f2c14e;
}
.stDataFrame {
    background-color: #161b22;
}
section[data-testid="stSidebar"] {
    background-color: #10141c;
    border-right: 1px solid #262d3a;
}
div.stButton > button {
    background-color: #f2c14e;
    color: #0e1117;
    font-weight: 700;
    border-radius: 8px;
    border: none;
}
div.stButton > button:hover {
    background-color: #f7d878;
    color: #0e1117;
}
hr {
    border-color: #262d3a;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

SECTORS = [
    "Tümü",
    "Technology",
    "Healthcare",
    "Financial Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Basic Materials",
    "Industrials",
    "Real Estate",
    "Utilities",
    "Communication Services",
]

FALLBACK_SYMBOLS = [
    "SIRI", "SNDL", "NOK", "PLUG", "FCEL", "GEVO", "IDEX", "CTRM", "TOPS", "SHIP",
    "ZOM", "GNUS", "XELA", "NAKD", "CIDM", "MARA", "RIOT", "BBIG", "PHUN", "PROG",
    "ATER", "MULN", "SNTG", "OPTT", "BLNK", "WKHS", "RIDE", "NKLA", "GOEV", "HYLN",
    "APRN", "CLOV", "WISH", "SDC", "SOFI", "OCGN", "INPX", "VXRT", "TNXP", "SESN",
    "ADMP", "BNGO", "CTXR", "GEVO", "SPWR", "FUV", "AYRO", "ENVX", "MVIS", "VUZI",
    "AGEN", "AXSM", "AMPE", "ACRX", "AEZS", "BTAI", "CYCC", "CYRN", "DRIO", "EYES",
    "FTFT", "GNPX", "HUSA", "IMTE", "JAGX", "KNDI", "LGVN", "MDVL", "NVOS", "ONTX",
    "PASO", "QMCO", "RETO", "SLNH", "TENX", "UAVS", "VERB", "WATT", "XSPA", "YTRA",
    "ZKIN", "ATOS", "BKKT", "CEI", "DWAC", "EXPR", "GME", "AMC", "BBBY", "MMAT",
]


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


@st.cache_data(ttl=86400, show_spinner=False)
def load_symbol_universe(exchange_filter):
    try:
        nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        other_url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

        nasdaq_df = pd.read_csv(nasdaq_url, sep="|")
        nasdaq_df = nasdaq_df[nasdaq_df["Test Issue"] == "N"]
        nasdaq_df = nasdaq_df[["Symbol", "Security Name", "ETF"]].copy()
        nasdaq_df["Exchange"] = "NASDAQ"

        other_df = pd.read_csv(other_url, sep="|")
        other_df = other_df[other_df["Test Issue"] == "N"]
        other_df = other_df.rename(columns={"NASDAQ Symbol": "Symbol", "Exchange": "ExchangeCode"})
        exchange_map = {"A": "NYSE American", "N": "NYSE", "P": "NYSE Arca", "Z": "BATS", "V": "IEX"}
        other_df["Exchange"] = other_df["ExchangeCode"].map(exchange_map).fillna("OTHER")
        other_df = other_df[["Symbol", "Security Name", "ETF", "Exchange"]].copy()

        universe = pd.concat([nasdaq_df, other_df], ignore_index=True)
        universe = universe[universe["ETF"] != "Y"]
        universe["Symbol"] = universe["Symbol"].astype(str).str.strip()
        universe = universe[universe["Symbol"].str.match(r"^[A-Z]{1,5}$", na=False)]
        universe = universe.drop_duplicates(subset="Symbol").reset_index(drop=True)

        if exchange_filter == "Sadece NASDAQ":
            universe = universe[universe["Exchange"] == "NASDAQ"]
        elif exchange_filter == "Sadece NYSE":
            universe = universe[universe["Exchange"].isin(["NYSE", "NYSE American", "NYSE Arca"])]

        if universe.empty:
            raise ValueError("Sembol evreni boş döndü")

        return universe.reset_index(drop=True)
    except Exception:
        fallback = pd.DataFrame({
            "Symbol": FALLBACK_SYMBOLS,
            "Security Name": FALLBACK_SYMBOLS,
            "ETF": ["N"] * len(FALLBACK_SYMBOLS),
            "Exchange": ["NASDAQ"] * len(FALLBACK_SYMBOLS),
        })
        return fallback


@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_data(symbols_tuple, period="6mo", interval="1d"):
    symbols = list(symbols_tuple)
    data = {}
    errors = []
    chunk_size = 40

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        try:
            raw = yf.download(
                tickers=chunk,
                period=period,
                interval=interval,
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=False,
            )
        except Exception as exc:
            errors.append(f"{chunk[0]}...: {str(exc)}")
            continue

        if raw is None or raw.empty:
            continue

        if len(chunk) == 1:
            sym = chunk[0]
            df_sym = raw.dropna(how="all")
            if not df_sym.empty:
                data[sym] = df_sym
        else:
            for sym in chunk:
                try:
                    df_sym = raw[sym].dropna(how="all")
                    if not df_sym.empty and "Close" in df_sym.columns:
                        data[sym] = df_sym
                except Exception:
                    continue

    return data, errors


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_ticker_info(symbol):
    try:
        ticker_obj = yf.Ticker(symbol)
        info = ticker_obj.info
        return {
            "sector": info.get("sector") or "Bilinmiyor",
            "industry": info.get("industry") or "Bilinmiyor",
            "longName": info.get("longName") or info.get("shortName") or symbol,
            "marketCap": info.get("marketCap"),
        }
    except Exception:
        return {"sector": "Bilinmiyor", "industry": "Bilinmiyor", "longName": symbol, "marketCap": None}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_tradingview_scan(price_min, price_max, min_volume, vol_spike_mult,
                            rsi_min, rsi_max, sector, exchange_filter, macd_filter, sma_filter):
    filters = [
        {"left": "close", "operation": "in_range", "right": [price_min, price_max]},
        {"left": "volume", "operation": "greater", "right": min_volume},
        {"left": "relative_volume_10d_calc", "operation": "greater", "right": vol_spike_mult},
        {"left": "is_primary", "operation": "equal", "right": True},
        {"left": "type", "operation": "equal", "right": "stock"},
    ]

    if rsi_max is not None:
        filters.append({"left": "RSI", "operation": "less", "right": rsi_max})
    if rsi_min is not None:
        filters.append({"left": "RSI", "operation": "greater", "right": rsi_min})
    if sector and sector != "Tümü":
        filters.append({"left": "sector", "operation": "in_range", "right": [sector]})
    if exchange_filter == "Sadece NASDAQ":
        filters.append({"left": "exchange", "operation": "in_range", "right": ["NASDAQ"]})
    elif exchange_filter == "Sadece NYSE":
        filters.append({"left": "exchange", "operation": "in_range", "right": ["NYSE", "AMEX"]})

    if macd_filter == "MACD > Sinyal (Pozitif)":
        filters.append({"left": "MACD.macd", "operation": "greater", "right": ["MACD.signal"]})
    elif macd_filter == "MACD < Sinyal (Negatif)":
        filters.append({"left": "MACD.macd", "operation": "less", "right": ["MACD.signal"]})

    payload = {
        "filter": filters,
        "options": {"lang": "en"},
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": [
            "description", "close", "change", "volume",
            "average_volume_10d_calc", "relative_volume_10d_calc",
            "RSI", "MACD.macd", "MACD.signal",
            "SMA20", "SMA50", "SMA200", "sector",
        ],
        "sort": {"sortBy": "relative_volume_10d_calc", "sortOrder": "desc"},
        "range": [0, 300],
    }

    try:
        response = requests.post(
            "https://scanner.tradingview.com/america/scan",
            json=payload,
            headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"},
            timeout=15,
        )
        response.raise_for_status()
        raw_json = response.json()
    except Exception as exc:
        return None, str(exc)

    rows = []
    for item in raw_json.get("data", []):
        try:
            d = item.get("d", [])
            if len(d) < 12:
                continue

            symbol = item.get("s", "UNKNOWN:UNKNOWN")
            exchange_name = symbol.split(":")[0] if ":" in symbol else "UNKNOWN"
            ticker_symbol = symbol.split(":")[-1] if ":" in symbol else symbol

            close_price = safe_float(d[1])
            if close_price <= 0:
                continue

            sma20 = safe_float(d[9], close_price)
            sma50 = safe_float(d[10], close_price)
            sma200 = safe_float(d[11], close_price)

            strong_up = close_price > sma20 > sma50 > sma200
            strong_down = close_price < sma20 < sma50 < sma200

            if sma_filter == "Güçlü Yükseliş (Fiyat>SMA20>SMA50>SMA200)" and not strong_up:
                continue
            if sma_filter == "Güçlü Düşüş (Fiyat<SMA20<SMA50<SMA200)" and not strong_down:
                continue

            trend_label = "Güçlü Yükseliş" if strong_up else ("Güçlü Düşüş" if strong_down else "Nötr")

            rows.append({
                "Sembol": ticker_symbol,
                "Şirket": d[0] if d[0] else ticker_symbol,
                "Sektör": d[12] if d[12] else "Bilinmiyor",
                "Fiyat": round(close_price, 4),
                "Değişim%": round(safe_float(d[2]), 2),
                "Hacim": int(safe_float(d[3])),
                "Ort.Hacim10": int(safe_float(d[4])),
                "HacimArtış(x)": round(safe_float(d[5]), 2),
                "RSI14": round(safe_float(d[6], 50), 2),
                "MACD": round(safe_float(d[7]), 4),
                "MACD_Sinyal": round(safe_float(d[8]), 4),
                "SMA20": round(sma20, 4),
                "SMA50": round(sma50, 4),
                "SMA200": round(sma200, 4),
                "Trend": trend_label,
                "Exchange": exchange_name,
            })
        except Exception:
            continue

    result_df = pd.DataFrame(rows)
    return result_df, None


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_bollinger(series, period=20, std_mult=2):
    sma = series.rolling(window=period, min_periods=1).mean()
    std = series.rolling(window=period, min_periods=1).std().fillna(0)
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, sma, lower


def add_indicators(df):
    result = df.copy()
    result["SMA20"] = result["Close"].rolling(20, min_periods=1).mean()
    result["SMA50"] = result["Close"].rolling(50, min_periods=1).mean()
    result["SMA200"] = result["Close"].rolling(200, min_periods=1).mean()
    result["RSI14"] = compute_rsi(result["Close"], 14)
    macd_line, signal_line, hist = compute_macd(result["Close"])
    result["MACD"] = macd_line
    result["MACD_Signal"] = signal_line
    result["MACD_Hist"] = hist
    bb_upper, bb_mid, bb_lower = compute_bollinger(result["Close"])
    result["BB_Upper"] = bb_upper
    result["BB_Mid"] = bb_mid
    result["BB_Lower"] = bb_lower
    result["AvgVolume10"] = result["Volume"].rolling(10, min_periods=1).mean()
    result["VolumeSpikeRatio"] = result["Volume"] / result["AvgVolume10"].replace(0, np.nan)
    return result


def run_scan_yfinance(universe_df, max_scan, use_random, price_min, price_max, min_volume,
                       vol_spike_mult, period, interval, rsi_filter, macd_filter, sma_filter, sector_filter):
    all_symbols = universe_df["Symbol"].tolist()
    if use_random and len(all_symbols) > max_scan:
        symbols = random.sample(all_symbols, max_scan)
    else:
        symbols = all_symbols[:max_scan]

    price_data, fetch_errors = fetch_price_data(tuple(sorted(symbols)), period=period, interval=interval)

    results = []
    for sym in symbols:
        df = price_data.get(sym)
        if df is None or df.empty or len(df) < 5:
            continue
        try:
            df = df.dropna(subset=["Close", "Volume", "Open", "High", "Low"])
            if df.empty:
                continue

            last_close = float(df["Close"].iloc[-1])
            if last_close <= 0:
                continue
            if not (price_min <= last_close <= price_max):
                continue

            last_volume = float(df["Volume"].iloc[-1])
            if last_volume < min_volume:
                continue

            df_ind = add_indicators(df)

            avg_vol10_raw = df_ind["AvgVolume10"].iloc[-1]
            avg_vol10 = float(avg_vol10_raw) if not pd.isna(avg_vol10_raw) else 0.0
            vol_spike_raw = df_ind["VolumeSpikeRatio"].iloc[-1]
            vol_spike = float(vol_spike_raw) if not pd.isna(vol_spike_raw) else 0.0

            if avg_vol10 > 0 and vol_spike < vol_spike_mult:
                continue

            prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
            change_pct = ((last_close - prev_close) / prev_close * 100) if prev_close else 0.0

            rsi_val = float(df_ind["RSI14"].iloc[-1])
            if rsi_filter == "RSI Aşırı Satım (<30)" and not (rsi_val < 30):
                continue
            if rsi_filter == "RSI Aşırı Alım (>70)" and not (rsi_val > 70):
                continue

            macd_now = float(df_ind["MACD"].iloc[-1])
            macd_sig_now = float(df_ind["MACD_Signal"].iloc[-1])
            macd_prev = float(df_ind["MACD"].iloc[-2]) if len(df_ind) > 1 else macd_now
            macd_sig_prev = float(df_ind["MACD_Signal"].iloc[-2]) if len(df_ind) > 1 else macd_sig_now
            bullish_cross = (macd_prev <= macd_sig_prev) and (macd_now > macd_sig_now)
            bearish_cross = (macd_prev >= macd_sig_prev) and (macd_now < macd_sig_now)

            if macd_filter == "Yükseliş Kesişimi (Bullish)" and not bullish_cross:
                continue
            if macd_filter == "Düşüş Kesişimi (Bearish)" and not bearish_cross:
                continue

            sma20 = float(df_ind["SMA20"].iloc[-1])
            sma50 = float(df_ind["SMA50"].iloc[-1])
            sma200 = float(df_ind["SMA200"].iloc[-1])
            sma20_prev = float(df_ind["SMA20"].iloc[-2]) if len(df_ind) > 1 else sma20
            sma50_prev = float(df_ind["SMA50"].iloc[-2]) if len(df_ind) > 1 else sma50

            golden_cross = (sma20_prev <= sma50_prev) and (sma20 > sma50)
            death_cross = (sma20_prev >= sma50_prev) and (sma20 < sma50)
            strong_up = last_close > sma20 > sma50 > sma200
            strong_down = last_close < sma20 < sma50 < sma200

            if sma_filter == "Güçlü Yükseliş (Fiyat>SMA20>SMA50>SMA200)" and not strong_up:
                continue
            if sma_filter == "Güçlü Düşüş (Fiyat<SMA20<SMA50<SMA200)" and not strong_down:
                continue
            if sma_filter == "Golden Cross (20/50)" and not golden_cross:
                continue
            if sma_filter == "Death Cross (20/50)" and not death_cross:
                continue

            info = fetch_ticker_info(sym)
            if sector_filter != "Tümü" and info["sector"] != sector_filter:
                continue

            trend_label = "Güçlü Yükseliş" if strong_up else ("Güçlü Düşüş" if strong_down else "Nötr")

            results.append({
                "Sembol": sym,
                "Şirket": info["longName"],
                "Sektör": info["sector"],
                "Fiyat": round(last_close, 4),
                "Değişim%": round(change_pct, 2),
                "Hacim": int(last_volume),
                "Ort.Hacim10": int(avg_vol10),
                "HacimArtış(x)": round(vol_spike, 2),
                "RSI14": round(rsi_val, 2),
                "MACD": round(macd_now, 4),
                "MACD_Sinyal": round(macd_sig_now, 4),
                "SMA20": round(sma20, 4),
                "SMA50": round(sma50, 4),
                "SMA200": round(sma200, 4),
                "Trend": trend_label,
            })
        except Exception:
            continue

    result_df = pd.DataFrame(results)
    return result_df, fetch_errors


def build_chart(df, symbol):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(f"{symbol} — Fiyat, Bollinger & SMA", "Hacim", "RSI (14)", "MACD"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Fiyat", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], line=dict(color="rgba(173,216,230,0.5)", width=1), name="BB Üst"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Mid"], line=dict(color="rgba(173,216,230,0.8)", width=1, dash="dot"), name="BB Orta"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], line=dict(color="rgba(173,216,230,0.5)", width=1), fill="tonexty", fillcolor="rgba(173,216,230,0.07)", name="BB Alt"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], line=dict(color="#f2c14e", width=1.3), name="SMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], line=dict(color="#f27649", width=1.3), name="SMA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], line=dict(color="#9b59b6", width=1.3), name="SMA200"), row=1, col=1)

    vol_colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Hacim"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], line=dict(color="#00bcd4", width=1.5), name="RSI14"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", row=3, col=1)

    macd_colors = np.where(df["MACD_Hist"] >= 0, "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], marker_color=macd_colors, name="MACD Histogram"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="#42a5f5", width=1.3), name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], line=dict(color="#ffa726", width=1.3), name="Sinyal"), row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=950,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def calculate_risk(current_price, budget, target_pct, stop_pct):
    if current_price is None or current_price <= 0:
        return None

    shares = math.floor(budget / current_price)
    invested_amount = shares * current_price
    stop_price = current_price * (1 - stop_pct / 100)
    target_price = current_price * (1 + target_pct / 100)
    risk_per_share = current_price - stop_price
    reward_per_share = target_price - current_price
    total_risk = risk_per_share * shares
    total_reward = reward_per_share * shares
    rr_ratio = (reward_per_share / risk_per_share) if risk_per_share > 0 else 0.0

    return {
        "shares": shares,
        "invested_amount": invested_amount,
        "stop_price": stop_price,
        "target_price": target_price,
        "risk_per_share": risk_per_share,
        "reward_per_share": reward_per_share,
        "total_risk": total_risk,
        "total_reward": total_reward,
        "rr_ratio": rr_ratio,
    }


def display_risk_panel(risk_data):
    if risk_data is None:
        st.warning("Risk hesaplaması için geçerli bir fiyat bulunamadı.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alınabilecek Lot", f"{risk_data['shares']}")
    c2.metric("Stop-Loss Fiyatı", f"${risk_data['stop_price']:.4f}")
    c3.metric("Hedef Kâr Fiyatı (TP)", f"${risk_data['target_price']:.4f}")
    c4.metric("Risk/Ödül Oranı", f"{risk_data['rr_ratio']:.2f}")

    st.markdown("---")

    table_data = {
        "Parametre": [
            "Yatırılan Tutar", "Hisse Başı Risk", "Hisse Başı Ödül",
            "Toplam Risk Tutarı", "Toplam Ödül Tutarı", "Risk/Ödül Oranı",
        ],
        "Değer": [
            f"${risk_data['invested_amount']:.2f}",
            f"${risk_data['risk_per_share']:.4f}",
            f"${risk_data['reward_per_share']:.4f}",
            f"${risk_data['total_risk']:.2f}",
            f"${risk_data['total_reward']:.2f}",
            f"{risk_data['rr_ratio']:.2f} : 1",
        ],
    }
    st.table(pd.DataFrame(table_data))

    if risk_data["rr_ratio"] < 1.5:
        st.warning("⚠️ Risk/Ödül oranı 1.5'in altında. Bu işlem riskli kabul edilebilir.")
    else:
        st.success("✅ Risk/Ödül oranı kabul edilebilir seviyede.")


def main():
    st.title("📈 Penny Stock Scanner & Analytics")
    st.caption("NASDAQ & NYSE — Düşük Fiyatlı, Yüksek Potansiyelli Hisse Tarayıcısı")

    with st.sidebar:
        st.header("⚙️ Veri Kaynağı")
        data_source = st.selectbox(
            "Tarama Motoru",
            ["TradingView (Hızlı Tarama)", "yfinance (Detaylı Tarama)"],
            help="TradingView: binlerce hisseyi saniyeler içinde tarar. yfinance: daha yavaş ama MACD/SMA kesişim tespiti gibi geçmişe dayalı sinyaller sunar.",
        )

        st.markdown("---")
        st.header("🔍 Filtreleme Parametreleri")

        exchange_filter = st.selectbox("Borsa", ["Tümü", "Sadece NASDAQ", "Sadece NYSE"])

        if data_source == "yfinance (Detaylı Tarama)":
            max_scan = st.slider("Maksimum Taranacak Hisse Sayısı", 20, 500, 150, step=10)
            use_random = st.checkbox("Rastgele Örnekleme Kullan", value=False)
        else:
            max_scan = 300
            use_random = False

        price_min, price_max = st.slider("Fiyat Aralığı ($)", 0.01, 20.0, (0.50, 5.00), step=0.01)
        min_volume = st.number_input("Minimum Günlük Hacim", min_value=0, value=500000, step=50000)
        vol_spike_mult = st.slider("Min. Hacim Artış Katsayısı (Ort. 10 Güne Göre)", 1.0, 10.0, 3.0, step=0.1)

        st.markdown("---")
        st.subheader("📅 Grafik Zaman Ayarları")
        interval = st.selectbox("Zaman Aralığı", ["1d", "1h"], index=0)
        if interval == "1h":
            period = st.selectbox("Veri Periyodu", ["5d", "1mo"], index=1)
        else:
            period = st.selectbox("Veri Periyodu", ["3mo", "6mo", "1y"], index=1)

        st.markdown("---")
        sector_filter = st.selectbox("Sektör", SECTORS)
        rsi_filter = st.selectbox("RSI Koşulu", ["Tümü", "RSI Aşırı Satım (<30)", "RSI Aşırı Alım (>70)"])

        if data_source == "yfinance (Detaylı Tarama)":
            macd_filter = st.selectbox("MACD Koşulu", ["Tümü", "Yükseliş Kesişimi (Bullish)", "Düşüş Kesişimi (Bearish)"])
            sma_filter = st.selectbox(
                "SMA Trend / Kesişim Koşulu",
                ["Tümü", "Güçlü Yükseliş (Fiyat>SMA20>SMA50>SMA200)", "Güçlü Düşüş (Fiyat<SMA20<SMA50<SMA200)",
                 "Golden Cross (20/50)", "Death Cross (20/50)"],
            )
        else:
            macd_filter = st.selectbox("MACD Koşulu", ["Tümü", "MACD > Sinyal (Pozitif)", "MACD < Sinyal (Negatif)"])
            sma_filter = st.selectbox(
                "SMA Trend Koşulu",
                ["Tümü", "Güçlü Yükseliş (Fiyat>SMA20>SMA50>SMA200)", "Güçlü Düşüş (Fiyat<SMA20<SMA50<SMA200)"],
            )
            st.caption("ℹ️ Kesişim (cross) tabanlı sinyaller için 'yfinance (Detaylı Tarama)' modunu seçin.")

        scan_button = st.button("🚀 Taramayı Başlat / Yenile", use_container_width=True, type="primary")

    if "scan_results" not in st.session_state:
        st.session_state.scan_results = None
        st.session_state.price_data = {}
        st.session_state.last_scan_time = None
        st.session_state.active_source = None

    if scan_button or st.session_state.scan_results is None:
        with st.spinner("Piyasa taranıyor, lütfen bekleyin..."):
            if data_source == "TradingView (Hızlı Tarama)":
                rsi_min_val = 70 if rsi_filter == "RSI Aşırı Alım (>70)" else None
                rsi_max_val = 30 if rsi_filter == "RSI Aşırı Satım (<30)" else None

                result_df, tv_error = fetch_tradingview_scan(
                    price_min, price_max, min_volume, vol_spike_mult,
                    rsi_min_val, rsi_max_val, sector_filter, exchange_filter, macd_filter, sma_filter,
                )

                if tv_error is not None:
                    st.warning(
                        f"TradingView tarama servisine ulaşılamadı ({tv_error}). "
                        f"Otomatik olarak yfinance tarama motoruna geçiliyor."
                    )
                    universe_df = load_symbol_universe(exchange_filter)
                    result_df, fetch_errors = run_scan_yfinance(
                        universe_df, 150, False, price_min, price_max, min_volume,
                        vol_spike_mult, period, interval, rsi_filter,
                        "Tümü", "Tümü", sector_filter,
                    )
                    st.session_state.active_source = "yfinance (Otomatik Yedek)"
                    if fetch_errors:
                        st.warning(f"Bazı veri parçaları alınırken sorun oluştu ({len(fetch_errors)} hata).")
                else:
                    st.session_state.active_source = "TradingView"
            else:
                universe_df = load_symbol_universe(exchange_filter)
                if universe_df is None or universe_df.empty:
                    st.error("Hisse evreni yüklenemedi. Lütfen internet bağlantınızı kontrol edin veya daha sonra tekrar deneyin.")
                    st.stop()

                result_df, fetch_errors = run_scan_yfinance(
                    universe_df, max_scan, use_random, price_min, price_max, min_volume,
                    vol_spike_mult, period, interval, rsi_filter, macd_filter, sma_filter, sector_filter,
                )
                st.session_state.active_source = "yfinance"
                if fetch_errors:
                    st.warning(f"Bazı veri parçaları alınırken sorun oluştu ({len(fetch_errors)} hata). Sonuçlar kısmi olabilir.")

            st.session_state.scan_results = result_df
            st.session_state.last_scan_time = datetime.now()

    result_df = st.session_state.scan_results

    if st.session_state.last_scan_time:
        st.caption(
            f"Son Güncelleme: {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"— Kaynak: {st.session_state.active_source}"
        )

    if result_df is None or result_df.empty:
        st.warning("Belirtilen kriterlere uyan hisse bulunamadı. Lütfen filtre parametrelerini gevşetin veya taranacak hisse sayısını artırın.")
        return

    st.subheader("🔥 En Yüksek Hacim Patlaması Yaşayan İlk 5 Hisse")
    top5 = result_df.sort_values("HacimArtış(x)", ascending=False).head(5)
    if not top5.empty:
        cols = st.columns(len(top5))
        for col, (_, row) in zip(cols, top5.iterrows()):
            with col:
                st.metric(
                    label=f"{row['Sembol']}",
                    value=f"${row['Fiyat']:.2f}",
                    delta=f"{row['Değişim%']:.2f}%",
                )
                st.caption(f"Hacim Artışı: {row['HacimArtış(x)']:.2f}x")

    st.subheader("📊 Filtrelenmiş Hisse Listesi")
    display_columns = [c for c in result_df.columns if c != "Exchange"]
    st.dataframe(
        result_df[display_columns].sort_values("HacimArtış(x)", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("📈 Detaylı Teknik Analiz Grafiği")
    symbol_list = result_df["Sembol"].tolist()
    selected_symbol = st.selectbox("Analiz için hisse seçin", symbol_list)

    if selected_symbol:
        with st.spinner(f"{selected_symbol} için geçmiş fiyat verisi indiriliyor..."):
            chart_data, chart_errors = fetch_price_data((selected_symbol,), period=period, interval=interval)
            df_sel = chart_data.get(selected_symbol)

        if df_sel is None or df_sel.empty:
            st.error(
                f"{selected_symbol} için grafik verisi bulunamadı. "
                f"Bu sembolün yfinance üzerinde geçici olarak erişilemez durumda olması mümkündür."
            )
        else:
            try:
                df_ind = add_indicators(df_sel)
                fig = build_chart(df_ind, selected_symbol)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("💰 Pozisyon Boyutlandırma ve Risk Yönetim Paneli")
                current_price = float(df_ind["Close"].iloc[-1])

                col1, col2, col3 = st.columns(3)
                with col1:
                    budget = st.number_input("Giriş Bütçesi ($)", min_value=1.0, value=1000.0, step=50.0)
                with col2:
                    target_pct = st.number_input("Hedef Kâr Yüzdesi (%)", min_value=0.1, value=15.0, step=0.5)
                with col3:
                    stop_pct = st.number_input("Stop-Loss Yüzdesi (%)", min_value=0.1, value=7.0, step=0.5)

                risk_data = calculate_risk(current_price, budget, target_pct, stop_pct)
                display_risk_panel(risk_data)
            except Exception as exc:
                st.error(f"{selected_symbol} için grafik oluşturulurken bir hata oluştu: {str(exc)}")

    st.markdown("---")
    st.caption("⚠️ Bu uygulama yalnızca eğitim ve bilgilendirme amaçlıdır, yatırım tavsiyesi niteliği taşımaz.")


if __name__ == "__main__":
    main()
