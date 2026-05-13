import pandas as pd
import numpy as np
from core.analysis import get_technical_indicators, get_support_resistance, simple_prediction


def get_chat_response(question: str, ticker: str, df: pd.DataFrame, info: dict) -> str:
    q = question.lower().strip()
    name = ticker.replace(".NS", "")

    if df is None or df.empty:
        return f"Sorry, I couldn't load data for {name} right now. Please try again."

    indicators = get_technical_indicators(df)
    sr         = get_support_resistance(df)
    pred       = simple_prediction(df)

    close      = df["Close"]
    current    = float(close.iloc[-1])
    prev       = float(close.iloc[-2]) if len(close) > 1 else current
    change     = current - prev
    change_pct = (change / prev) * 100
    arrow      = "▲" if change >= 0 else "▼"

    rsi        = indicators.get("RSI", 50)
    ma10       = indicators.get("MA10", current)
    ma20       = indicators.get("MA20", current)
    ma50       = indicators.get("MA50", current)
    bb_upper   = indicators.get("BB_Upper", current * 1.02)
    bb_lower   = indicators.get("BB_Lower", current * 0.98)
    resistance = sr.get("resistance", current * 1.05)
    support    = sr.get("support", current * 0.95)
    direction  = pred.get("direction", "HOLD")
    confidence = pred.get("confidence", 50)
    reason     = pred.get("reason", "")
    score      = pred.get("score", 0)

    high52 = info.get("fiftyTwoWeekHigh") or float(df["High"].tail(252).max())
    low52  = info.get("fiftyTwoWeekLow")  or float(df["Low"].tail(252).min())

    ret_1w = (current - float(close.iloc[-5]))   / float(close.iloc[-5])   * 100 if len(close) >= 5   else None
    ret_1m = (current - float(close.iloc[-21]))  / float(close.iloc[-21])  * 100 if len(close) >= 21  else None
    ret_3m = (current - float(close.iloc[-63]))  / float(close.iloc[-63])  * 100 if len(close) >= 63  else None
    ret_6m = (current - float(close.iloc[-126])) / float(close.iloc[-126]) * 100 if len(close) >= 126 else None
    ret_1y = (current - float(close.iloc[-252])) / float(close.iloc[-252]) * 100 if len(close) >= 252 else None

    daily_returns = close.pct_change().dropna()
    annual_vol    = float(daily_returns.std()) * (252 ** 0.5) * 100
    max_dd        = float(((close - close.cummax()) / close.cummax()).min() * 100)

    vol_today = float(df["Volume"].iloc[-1])
    vol_avg20 = float(df["Volume"].rolling(20).mean().iloc[-1])
    vol_ratio = vol_today / vol_avg20 if vol_avg20 > 0 else 1

    company  = info.get("longName", name)
    sector   = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    mktcap   = info.get("marketCap", 0)
    pe       = info.get("trailingPE", "N/A")
    pe_str   = f"{pe:.1f}" if isinstance(pe, (int, float)) else "N/A"
    mktcap_str = f"₹{mktcap/1e9:.0f}B" if mktcap else "N/A"

    clr_change = "#22c55e" if change >= 0 else "#ef4444"
    pred_color = "#22c55e" if direction == "BUY" else "#ef4444" if direction == "SELL" else "#f59e0b"

    def fmt_ret(r):
        if r is None: return "<span style='color:#6b7280;'>N/A</span>"
        c = "#22c55e" if r >= 0 else "#ef4444"
        return f"<span style='color:{c};font-weight:600;'>{r:+.1f}%</span>"

    # ─────────────────────────────────────────────────────────────────────────
    # INTENT BUCKETS — very broad keyword sets so almost anything matches
    # ─────────────────────────────────────────────────────────────────────────

    PRICE_KW       = ["price", "current", "today", "now", "trading", "rate", "value", "worth",
                      "how much", "what is it", "quote", "last price", "at what"]

    RETURNS_KW     = ["return", "returns", "performance", "perform", "gain", "profit", "loss",
                      "grow", "growth", "grown", "how has", "how did", "done well", "doing well",
                      "past", "history", "historical", "last year", "last month", "last week",
                      "since", "change over", "percent", "percentage", "up by", "down by",
                      "how much has", "how much did", "appreciat", "depreciat"]

    PREDICT_KW     = ["predict", "prediction", "forecast", "signal", "buy", "sell", "hold",
                      "should i", "recommend", "recommendation", "tomorrow", "next week",
                      "next month", "go up", "go down", "will it", "is it going", "worth buying",
                      "worth selling", "good time", "right time", "entry", "exit", "target price",
                      "upside", "downside", "outlook", "future", "expect", "expectation",
                      "invest", "investment", "position", "long", "short"]

    INDICATOR_KW   = ["rsi", "indicator", "technical", "macd", "bollinger", "band", "moving average",
                      "momentum", "oscillator", "stochastic", "signal line", "crossover",
                      "overbought", "oversold", "ma10", "ma20", "ma50", "sma", "ema"]

    SUPPORT_KW     = ["support", "resistance", "level", "range", "floor", "ceiling", "zone",
                      "breakout", "breakdown", "pivot", "key level", "price level", "bounce",
                      "stop loss", "stoploss", "stop-loss"]

    RISK_KW        = ["risk", "volatile", "volatility", "safe", "risky", "dangerous", "stable",
                      "stability", "drawdown", "swing", "fluctuat", "crash", "fall", "drop",
                      "std", "standard deviation", "beta", "sharp", "how risky", "how safe"]

    WEEK52_KW      = ["52", "52w", "52-week", "year high", "year low", "all time", "ath",
                      "annual high", "annual low", "highest", "lowest", "peak", "bottom",
                      "record high", "record low"]

    COMPANY_KW     = ["about", "company", "what is", "what does", "sector", "industry",
                      "overview", "who", "tell me", "background", "business", "founded",
                      "headquarters", "products", "services", "profile", "description", "info"]

    VOLUME_KW      = ["volume", "traded", "liquidity", "shares traded", "trading volume",
                      "how many shares", "activity", "active"]

    TREND_KW       = ["trend", "direction", "bullish", "bearish", "momentum", "moving",
                      "uptrend", "downtrend", "sideways", "consolidat", "rally", "correction",
                      "recovery", "reversal", "market sentiment", "sentiment", "bias",
                      "which way", "where is it headed", "heading"]

    COMPARE_KW     = ["compare", "vs", "versus", "better than", "worse than", "relative",
                      "against", "benchmark", "nifty", "sensex", "index", "peer"]

    SUMMARY_KW     = ["summary", "summarize", "overview", "everything", "all", "full report",
                      "tell me everything", "complete", "detail", "analysis", "analyse", "analyze",
                      "overall", "in short", "brief", "briefing", "snapshot", "status"]

    def matches(kw_list):
        return any(w in q for w in kw_list)

    # ── PRICE ────────────────────────────────────────────────────────────────
    if matches(PRICE_KW):
        return (
            f"<b>{name}</b> is currently at <b>₹{current:,.1f}</b> "
            f"<span style='color:{clr_change};'>{arrow} {abs(change):.1f} ({abs(change_pct):.2f}%)</span> today.<br>"
            f"Yesterday's close was ₹{prev:,.1f}."
        )

    # ── RETURNS ──────────────────────────────────────────────────────────────
    elif matches(RETURNS_KW):
        best = max([(ret_1w,"1W"),(ret_1m,"1M"),(ret_3m,"3M"),(ret_6m,"6M"),(ret_1y,"1Y")],
                   key=lambda x: x[0] if x[0] is not None else -999)
        worst = min([(ret_1w,"1W"),(ret_1m,"1M"),(ret_3m,"3M"),(ret_6m,"6M"),(ret_1y,"1Y")],
                    key=lambda x: x[0] if x[0] is not None else 999)
        return (
            f"<b>Returns for {name}:</b><br><br>"
            f"<table style='border-collapse:collapse;width:100%;'>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>1 Week</td><td>{fmt_ret(ret_1w)}</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>1 Month</td><td>{fmt_ret(ret_1m)}</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>3 Months</td><td>{fmt_ret(ret_3m)}</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>6 Months</td><td>{fmt_ret(ret_6m)}</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>1 Year</td><td>{fmt_ret(ret_1y)}</td></tr>"
            f"</table><br>"
            f"<span style='color:#6b7280;font-size:12px;'>Past returns don't guarantee future performance.</span>"
        )

    # ── PREDICTION / BUY-SELL ────────────────────────────────────────────────
    elif matches(PREDICT_KW):
        color_word = {"BUY": "bullish 📈", "SELL": "bearish 📉", "HOLD": "neutral ➡️"}[direction]
        return (
            f"Based on technicals, <b>{name}</b> shows a "
            f"<b style='color:{pred_color};'>{direction}</b> signal with <b>{confidence}%</b> confidence.<br><br>"
            f"The setup is {color_word}: {reason}<br><br>"
            f"<span style='color:#6b7280;font-size:12px;'>⚠️ This is a technical signal only — always do your own research before investing.</span>"
        )

    # ── TECHNICAL INDICATORS ─────────────────────────────────────────────────
    elif matches(INDICATOR_KW):
        rsi_clr    = "#22c55e" if rsi < 35 else "#ef4444" if rsi > 70 else "#f59e0b"
        rsi_status = "Oversold 🟢" if rsi < 35 else "Overbought 🔴" if rsi > 70 else "Neutral"
        ma_pos     = "above" if current > ma20 else "below"
        return (
            f"<b>Technical indicators for {name}:</b><br><br>"
            f"<table style='border-collapse:collapse;width:100%;'>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>RSI (14)</td><td><b style='color:{rsi_clr};'>{rsi:.1f}</b> — {rsi_status}</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>MA10</td><td>₹{ma10:,.1f}</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>MA20</td><td>₹{ma20:,.1f}</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>MA50</td><td>₹{ma50:,.1f}</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>BB Upper</td><td>₹{bb_upper:,.1f}</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>BB Lower</td><td>₹{bb_lower:,.1f}</td></tr>"
            f"</table><br>"
            f"Price is <b>{ma_pos}</b> MA20. MA10 {'>' if ma10 > ma20 else '<'} MA20 → "
            f"{'short-term uptrend 📈' if ma10 > ma20 else 'short-term downtrend 📉'}."
        )

    # ── SUPPORT & RESISTANCE ─────────────────────────────────────────────────
    elif matches(SUPPORT_KW):
        pct_to_res = (resistance - current) / current * 100
        pct_to_sup = (current - support)    / current * 100
        closer = "resistance" if pct_to_res < pct_to_sup else "support"
        return (
            f"<b>Support & Resistance for {name}:</b><br><br>"
            f"<table style='border-collapse:collapse;width:100%;'>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>🔴 Resistance</td><td><b>₹{resistance:,.1f}</b> <span style='color:#6b7280;font-size:12px;'>(+{pct_to_res:.1f}% upside)</span></td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>📍 Current</td><td><b>₹{current:,.1f}</b></td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>🟢 Support</td><td><b>₹{support:,.1f}</b> <span style='color:#6b7280;font-size:12px;'>(-{pct_to_sup:.1f}% downside)</span></td></tr>"
            f"</table><br>"
            f"Stock is currently closer to <b>{closer}</b>."
        )

    # ── RISK ─────────────────────────────────────────────────────────────────
    elif matches(RISK_KW):
        risk_level = "High 🔴" if annual_vol > 40 else "Moderate 🟡" if annual_vol > 20 else "Low 🟢"
        return (
            f"<b>Risk profile for {name}:</b><br><br>"
            f"<table style='border-collapse:collapse;width:100%;'>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>Annual Volatility</td><td><b>{annual_vol:.1f}%</b> — {risk_level} risk</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>Max Drawdown (1Y)</td><td><b style='color:#ef4444;'>{max_dd:.1f}%</b></td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>Daily Std Dev</td><td>{float(daily_returns.std())*100:.2f}%</td></tr>"
            f"</table>"
        )

    # ── 52-WEEK RANGE ────────────────────────────────────────────────────────
    elif matches(WEEK52_KW):
        from_high = (current - high52) / high52 * 100
        from_low  = (current - low52)  / low52  * 100
        return (
            f"<b>52-Week range for {name}:</b><br><br>"
            f"<table style='border-collapse:collapse;width:100%;'>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>52W High</td><td><b>₹{high52:,.1f}</b> <span style='color:#ef4444;font-size:12px;'>({from_high:.1f}% from now)</span></td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>Current</td><td><b>₹{current:,.1f}</b></td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>52W Low</td><td><b>₹{low52:,.1f}</b> <span style='color:#22c55e;font-size:12px;'>(+{from_low:.1f}% from low)</span></td></tr>"
            f"</table>"
        )

    # ── COMPANY INFO ─────────────────────────────────────────────────────────
    elif matches(COMPANY_KW):
        return (
            f"<b>{company}</b> ({name})<br><br>"
            f"<table style='border-collapse:collapse;width:100%;'>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>Sector</td><td>{sector}</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>Industry</td><td>{industry}</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>Market Cap</td><td>{mktcap_str}</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>P/E Ratio</td><td>{pe_str}</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>Exchange</td><td>NSE India</td></tr>"
            f"</table>"
        )

    # ── VOLUME ───────────────────────────────────────────────────────────────
    elif matches(VOLUME_KW):
        vol_clr = "#22c55e" if vol_ratio > 1.2 else "#ef4444" if vol_ratio < 0.8 else "#f59e0b"
        note = "High volume — confirms price move." if vol_ratio > 1.5 else \
               "Low volume — weak conviction." if vol_ratio < 0.7 else \
               "Normal trading activity."
        return (
            f"<b>Volume for {name}:</b><br><br>"
            f"<table style='border-collapse:collapse;width:100%;'>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>Today</td><td><b>{vol_today:,.0f}</b> shares</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>20-Day Avg</td><td>{vol_avg20:,.0f} shares</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>Ratio</td><td><b style='color:{vol_clr};'>{vol_ratio:.1f}x</b></td></tr>"
            f"</table><br>{note}"
        )

    # ── TREND ────────────────────────────────────────────────────────────────
    elif matches(TREND_KW):
        ma_trend  = "uptrend 📈" if ma10 > ma20 else "downtrend 📉"
        rsi_trend = "bullish" if rsi > 50 else "bearish"
        mom_5d    = ret_1w if ret_1w is not None else 0
        overall   = "Bullish 📈" if score > 0 else "Bearish 📉" if score < 0 else "Neutral ➡️"
        return (
            f"<b>Trend analysis for {name}:</b><br><br>"
            f"<table style='border-collapse:collapse;width:100%;'>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>MA Crossover</td><td>{ma_trend} (MA10 {'>' if ma10 > ma20 else '<'} MA20)</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>RSI Trend</td><td>{rsi_trend} (RSI {rsi:.0f})</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>5-Day Momentum</td><td>{fmt_ret(mom_5d)}</td></tr>"
            f"<tr><td style='color:#6b7280;padding:4px 16px 4px 0;'>Overall Bias</td><td><b>{overall}</b></td></tr>"
            f"</table>"
        )

    # ── COMPARISON ───────────────────────────────────────────────────────────
    elif matches(COMPARE_KW):
        ret_1y_str = f"{ret_1y:+.1f}%" if ret_1y is not None else "N/A"
        return (
            f"I can only show data for <b>{name}</b> right now — multi-stock comparison isn't supported yet.<br><br>"
            f"Here's a quick snapshot of {name}: ₹{current:,.1f} | 1Y return: {fmt_ret(ret_1y)} | "
            f"Volatility: {annual_vol:.1f}% | Signal: <span style='color:{pred_color};'><b>{direction}</b></span>"
        )

    # ── FULL SUMMARY ─────────────────────────────────────────────────────────
    elif matches(SUMMARY_KW):
        ma_trend = "uptrend" if ma10 > ma20 else "downtrend"
        rsi_status = "oversold" if rsi < 35 else "overbought" if rsi > 70 else "neutral"
        return (
            f"<b>Full snapshot — {company} ({name})</b><br><br>"
            f"<b>Price:</b> ₹{current:,.1f} <span style='color:{clr_change};'>{arrow} {abs(change_pct):.2f}%</span> today<br>"
            f"<b>Returns:</b> 1M {fmt_ret(ret_1m)} · 3M {fmt_ret(ret_3m)} · 1Y {fmt_ret(ret_1y)}<br>"
            f"<b>Signal:</b> <span style='color:{pred_color};'><b>{direction}</b></span> ({confidence}% confidence)<br>"
            f"<b>Trend:</b> {'Uptrend 📈' if ma10 > ma20 else 'Downtrend 📉'} · RSI {rsi:.0f} ({rsi_status})<br>"
            f"<b>Support:</b> ₹{support:,.1f} · <b>Resistance:</b> ₹{resistance:,.1f}<br>"
            f"<b>Volatility:</b> {annual_vol:.1f}% · Max DD: {max_dd:.1f}%<br>"
            f"<b>52W:</b> ₹{low52:,.1f} – ₹{high52:,.1f}<br><br>"
            f"<span style='color:#6b7280;font-size:12px;'>⚠️ Not financial advice. Do your own research.</span>"
        )

    # ── SMART FALLBACK — tries to give useful info instead of just help menu ─
    else:
        # Detect if question has any stock-related words at all
        any_finance = any(w in q for w in [
            "stock", "share", "nse", "bse", "market", "price", "money", "fund",
            "portfolio", "dividend", "earnings", "revenue", "profit", name.lower()
        ])

        if any_finance:
            # Give a relevant default: quick summary
            return (
                f"Not sure exactly what you're asking, but here's a quick snapshot of <b>{name}</b>:<br><br>"
                f"Price: <b>₹{current:,.1f}</b> <span style='color:{clr_change};'>{arrow} {abs(change_pct):.2f}%</span> · "
                f"Signal: <span style='color:{pred_color};'><b>{direction}</b></span> ({confidence}%) · "
                f"RSI: {rsi:.0f} · Trend: {'📈 Up' if ma10 > ma20 else '📉 Down'}<br><br>"
                f"Try asking about: <b>price, returns, prediction, RSI, support, risk, trend, volume, summary</b>."
            )
        else:
            return (
                f"I'm focused on stock analysis for <b>{name}</b>. Try asking:<br><br>"
                f"<span style='color:#9ca3af;'>price · returns · prediction · RSI · support & resistance · risk · trend · volume · summary</span>"
            )