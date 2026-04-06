import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Stockin",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0f;
    color: #e2e8f0;
}
.stApp { background-color: #0a0a0f; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* Top bar */
.top-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 0; margin-bottom: 8px;
    border-bottom: 1px solid #1e1e2e;
}
.logo { font-size: 22px; font-weight: 700; color: #fff; letter-spacing: -0.5px; }
.logo span { color: #6366f1; }

/* Search bar */
.stSelectbox > div > div {
    background: #12121a !important;
    border: 1px solid #2d2d3d !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

/* Cards */
.card {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
}
.card-title {
    font-size: 11px; font-weight: 600; color: #6b7280;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;
}

/* Stock header */
.stock-name { font-size: 20px; font-weight: 700; color: #fff; margin-bottom: 2px; }
.stock-meta { font-size: 12px; color: #6b7280; margin-bottom: 12px; }
.price-big { font-size: 36px; font-weight: 700; color: #fff; }
.price-change-pos { font-size: 16px; color: #22c55e; font-weight: 600; }
.price-change-neg { font-size: 16px; color: #ef4444; font-weight: 600; }

/* Badges */
.badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 20px; font-size: 11px; font-weight: 600; margin: 2px;
}
.badge-green { background: #052e16; color: #22c55e; border: 1px solid #166534; }
.badge-red { background: #1f0707; color: #ef4444; border: 1px solid #7f1d1d; }
.badge-blue { background: #0c1445; color: #6366f1; border: 1px solid #3730a3; }
.badge-yellow { background: #1c1407; color: #f59e0b; border: 1px solid #92400e; }

/* Metric row */
.metric-row { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; }
.metric-item { flex: 1; min-width: 80px; }
.metric-label { font-size: 10px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-size: 15px; font-weight: 600; color: #e2e8f0; margin-top: 2px; }

/* Indicator pills */
.ind-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #1a1a2e; border: 1px solid #2d2d3d;
    border-radius: 10px; padding: 8px 14px; margin: 4px;
    font-size: 13px;
}
.ind-label { color: #9ca3af; font-size: 11px; }
.ind-value { color: #e2e8f0; font-weight: 600; }

/* Chatbot */
.chat-container {
    background: #0e0e1a;
    border: 1px solid #1e1e2e;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
    min-height: 300px;
    max-height: 420px;
    overflow-y: auto;
}
.chat-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 16px;
}
.chat-title { font-size: 16px; font-weight: 700; color: #fff; }
.online-dot { display: inline-block; width: 8px; height: 8px; background: #22c55e; border-radius: 50%; margin-right: 6px; }
.chat-subtitle { font-size: 12px; color: #6b7280; }
.ensemble-badge {
    background: #1e1b4b; color: #818cf8; border: 1px solid #3730a3;
    border-radius: 8px; padding: 4px 10px; font-size: 11px; font-weight: 600;
}
.msg-bot {
    background: #1a1a2e; border-radius: 12px 12px 12px 0;
    padding: 12px 16px; margin: 8px 0; max-width: 85%;
    font-size: 14px; color: #e2e8f0; line-height: 1.5;
}
.msg-user {
    background: #3730a3; border-radius: 12px 12px 0 12px;
    padding: 12px 16px; margin: 8px 0 8px auto; max-width: 75%;
    font-size: 14px; color: #fff; text-align: right;
}
.empty-chat {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 40px 0; color: #4b5563;
}
.empty-icon { font-size: 28px; margin-bottom: 12px; }
.empty-title { font-size: 15px; font-weight: 600; color: #6b7280; margin-bottom: 6px; }
.empty-sub { font-size: 12px; color: #4b5563; }

/* SR levels */
.sr-bar {
    height: 6px; border-radius: 3px;
    background: linear-gradient(90deg, #ef4444, #f59e0b, #22c55e);
    margin: 8px 0; position: relative;
}

/* Prediction card */
.pred-up { color: #22c55e; font-size: 28px; font-weight: 700; }
.pred-down { color: #ef4444; font-size: 28px; font-weight: 700; }
.conf-bar-bg {
    background: #1e1e2e; border-radius: 99px;
    height: 8px; margin: 8px 0;
}
.conf-bar-fill {
    height: 8px; border-radius: 99px;
    background: linear-gradient(90deg, #6366f1, #22c55e);
}

/* Buttons */
.stButton > button {
    background: #6366f1 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
}
.stTextInput > div > div > input {
    background: #12121a !important;
    border: 1px solid #2d2d3d !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 14px !important;
}
div[data-testid="stPlotlyChart"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

from core.data_pipeline import fetch_stock_data, get_stock_info, STOCKS
from core.charts import plot_price_chart, plot_indicators_chart
from core.analysis import get_technical_indicators, get_support_resistance, simple_prediction
from core.chatbot import get_chat_response

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = "Reliance Industries"

# ── Top bar ───────────────────────────────────────────────────────────────────
st.markdown('<div class="top-bar"><div class="logo">Stock<span>in</span></div></div>', unsafe_allow_html=True)

# ── Stock selector ────────────────────────────────────────────────────────────
col_sel, col_spacer = st.columns([2, 5])
with col_sel:
    selected_name = st.selectbox(
        "Select Stock",
        list(STOCKS.keys()),
        index=list(STOCKS.keys()).index(st.session_state.selected_stock),
        label_visibility="collapsed"
    )
    st.session_state.selected_stock = selected_name

ticker = STOCKS[selected_name]

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner(""):
    df = fetch_stock_data(ticker)
    info = get_stock_info(ticker)

# ── Layout: Chatbot left, Stock info right ────────────────────────────────────
col_chat, col_stock = st.columns([1.1, 1], gap="large")

# ════════════════════════════════════════════════════════════════════════════
# LEFT — Chatbot
# ════════════════════════════════════════════════════════════════════════════
with col_chat:
    st.markdown(f"""
    <div class="card" style="padding:16px 20px 8px;">
        <div class="chat-header">
            <div>
                <div class="chat-title">AI Assistant</div>
                <div class="chat-subtitle">Ask anything about {ticker.replace('.NS','')} — predictions, risk, news, trends, investment insights</div>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
            <span class="online-dot"></span>
            <span style="font-size:13px;font-weight:600;color:#fff;">Stockin AI</span>
            <span style="font-size:11px;color:#6b7280;">• Online • {ticker.replace('.NS','')}</span>
            <span style="margin-left:auto;" class="ensemble-badge">Ensemble AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    

# Chat input label
st.markdown(
    f"<div style='color:#9ca3af;font-size:13px;margin-bottom:6px;'>Ask anything about {ticker.replace('.NS','')}</div>",
    unsafe_allow_html=True
)

# Input box
user_input = st.text_input(
    "chat",
    placeholder=f"Ask about {ticker.replace('.NS','')}...",
    label_visibility="collapsed"
)

# Buttons row (clean)
col1, col2 = st.columns(2)

with col1:
    send = st.button("Send", use_container_width=True)

with col2:
    clear = st.button("Clear", use_container_width=True)

# Logic
if clear:
    st.session_state.messages = []
    st.rerun()

if send and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = get_chat_response(user_input, ticker, df, info)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# RIGHT — Stock overview
# ════════════════════════════════════════════════════════════════════════════
with col_stock:
    if df is not None and not df.empty:
        current = df["Close"].iloc[-1]
        prev = df["Close"].iloc[-2] if len(df) > 1 else current
        change = current - prev
        change_pct = (change / prev) * 100
        color = "#22c55e" if change >= 0 else "#ef4444"
        arrow = "▲" if change >= 0 else "▼"
        change_cls = "price-change-pos" if change >= 0 else "price-change-neg"

        company = info.get("longName", selected_name)
        sector = info.get("sector", "")
        industry = info.get("industry", "")

        st.markdown(f"""
        <div class="card">
            <div class="stock-name">{company}</div>
            <div class="stock-meta">{sector} · {industry}</div>
            <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:12px;">
                <div class="price-big">₹ {current:,.1f}</div>
                <div class="{change_cls}">{arrow} {abs(change):.1f} ({abs(change_pct):.2f}%)</div>
            </div>
            <div class="metric-row">
                <div class="metric-item">
                    <div class="metric-label">52W High</div>
                    <div class="metric-value">₹ {info.get('fiftyTwoWeekHigh', 0):,.1f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">52W Low</div>
                    <div class="metric-value">₹ {info.get('fiftyTwoWeekLow', 0):,.1f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Mkt Cap</div>
                    <div class="metric-value">₹ {info.get('marketCap', 0)/1e9:.0f}B</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">P/E Ratio</div>
                    <div class="metric-value">{info.get('trailingPE', 'N/A')}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Could not load stock data. Please try again.")

# ── Price Chart ───────────────────────────────────────────────────────────────
if df is not None and not df.empty:
    st.markdown('<div class="card-title" style="padding:0 4px;margin-top:8px;"> Price Chart</div>', unsafe_allow_html=True)
    fig = plot_price_chart(df, ticker)
    st.plotly_chart(fig, use_container_width=True)

    # ── Two column section ────────────────────────────────────────────────────
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        # Technical Indicators
        indicators = get_technical_indicators(df)
        st.markdown("""<div class="card-title" style="margin-top:8px;"> Technical Indicators</div>""", unsafe_allow_html=True)

        rsi = indicators.get("RSI", 50)
        rsi_color = "#22c55e" if rsi < 40 else "#ef4444" if rsi > 70 else "#f59e0b"
        rsi_label = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"

        ma10 = indicators.get("MA10", 0)
        ma20 = indicators.get("MA20", 0)
        ma50 = indicators.get("MA50", 0)
        close = df["Close"].iloc[-1]
        ma_signal = "Bullish" if close > ma20 else "Bearish"
        ma_color = "#22c55e" if ma_signal == "Bullish" else "#ef4444"

        vol_20 = df["Volume"].rolling(20).mean().iloc[-1]
        vol_now = df["Volume"].iloc[-1]
        vol_ratio = vol_now / vol_20 if vol_20 > 0 else 1

        bb_upper = indicators.get("BB_Upper", close * 1.02)
        bb_lower = indicators.get("BB_Lower", close * 0.98)
        bb_pos = (close - bb_lower) / (bb_upper - bb_lower + 1e-8) * 100

        ind_rows = [
            ("RSI (14)", f"{rsi:.1f}", rsi_color, rsi_label),
            ("MA Signal", ma_signal, ma_color, f"Price vs MA20"),
            ("MA 10", f"₹{ma10:,.1f}", "#6366f1", "Short-term"),
            ("MA 50", f"₹{ma50:,.1f}", "#6366f1", "Mid-term"),
            ("Volume", f"{vol_ratio:.1f}x avg", "#f59e0b", "vs 20-day avg"),
            ("BB Position", f"{bb_pos:.0f}%", "#818cf8", "in band"),
        ]

        for label, value, color, sub in ind_rows:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:8px 0;border-bottom:1px solid #1e1e2e;">
                <div>
                    <div style="font-size:13px;color:#9ca3af;">{label}</div>
                    <div style="font-size:11px;color:#4b5563;">{sub}</div>
                </div>
                <div style="font-size:14px;font-weight:600;color:{color};">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        # Support & Resistance
        sr = get_support_resistance(df)
        st.markdown("""<div class="card-title" style="margin-top:8px;"> Support & Resistance</div>""", unsafe_allow_html=True)

        cur = sr.get("current", close)
        res = sr.get("resistance", close * 1.05)
        sup = sr.get("support", close * 0.95)
        pct_to_res = (res - cur) / cur * 100
        pct_to_sup = (cur - sup) / cur * 100
        position_pct = (cur - sup) / (res - sup + 1e-8) * 100

        st.markdown(f"""
        <div style="margin-bottom:16px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="font-size:12px;color:#ef4444;">Support ₹{sup:,.1f}</span>
                <span style="font-size:12px;color:#22c55e;">Resistance ₹{res:,.1f}</span>
            </div>
            <div style="background:#1e1e2e;border-radius:99px;height:10px;position:relative;">
                <div style="background:linear-gradient(90deg,#ef4444,#f59e0b,#22c55e);
                            border-radius:99px;height:10px;width:100%;opacity:0.3;"></div>
                <div style="position:absolute;top:-3px;left:{min(position_pct,95):.0f}%;
                            width:16px;height:16px;background:#fff;border-radius:50%;
                            border:2px solid #6366f1;transform:translateX(-50%);"></div>
            </div>
            <div style="text-align:center;margin-top:8px;font-size:13px;color:#9ca3af;">
                Current ₹{cur:,.1f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        for label, val, color, pct, pct_label in [
            ("Resistance", f"₹{res:,.1f}", "#22c55e", f"+{pct_to_res:.1f}%", "upside"),
            ("Support", f"₹{sup:,.1f}", "#ef4444", f"-{pct_to_sup:.1f}%", "downside"),
            ("Current", f"₹{cur:,.1f}", "#6366f1", "", ""),
        ]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:8px 0;border-bottom:1px solid #1e1e2e;">
                <div style="font-size:13px;color:#9ca3af;">{label}</div>
                <div style="text-align:right;">
                    <div style="font-size:14px;font-weight:600;color:{color};">{val}</div>
                    <div style="font-size:11px;color:#4b5563;">{pct} {pct_label}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Prediction
        pred = simple_prediction(df)
        direction = pred.get("direction", "HOLD")
        confidence = pred.get("confidence", 50)
        reason = pred.get("reason", "")
        pred_color = "#22c55e" if direction == "BUY" else "#ef4444" if direction == "SELL" else "#f59e0b"

        st.markdown(f"""
        <div style="margin-top:20px;padding-top:16px;border-top:1px solid #1e1e2e;">
            <div class="card-title"> AI Signal</div>
            <div style="display:flex;align-items:center;gap:12px;margin:8px 0;">
                <div style="font-size:26px;font-weight:700;color:{pred_color};">{direction}</div>
                <div style="flex:1;">
                    <div style="background:#1e1e2e;border-radius:99px;height:7px;">
                        <div style="background:{pred_color};border-radius:99px;height:7px;width:{confidence}%;"></div>
                    </div>
                    <div style="font-size:11px;color:#6b7280;margin-top:3px;">Confidence {confidence}%</div>
                </div>
            </div>
            <div style="font-size:12px;color:#9ca3af;">{reason}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
