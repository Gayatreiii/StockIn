import pandas as pd
import numpy as np
from core.analysis import get_technical_indicators, get_support_resistance, simple_prediction


def get_chat_response(question: str, ticker: str, df: pd.DataFrame, info: dict) -> str:
    """
    Simple rule-based chatbot that answers questions about the stock
    using live data. No external AI API needed.
    """
    q = question.lower().strip()
    name = ticker.replace(".NS", "")

    if df is None or df.empty:
        return f"Sorry, I couldn't load data for {name} right now. Please try again."

    indicators = get_technical_indicators(df)
    sr = get_support_resistance(df)
    pred = simple_prediction(df)

    close = df["Close"]
    current = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else current
    change = current - prev
    change_pct = (change / prev) * 100
    arrow = "▲" if change >= 0 else "▼"

    rsi = indicators.get("RSI", 50)
    ma10 = indicators.get("MA10", current)
    ma20 = indicators.get("MA20", current)
    ma50 = indicators.get("MA50", current)
    resistance = sr.get("resistance", current * 1.05)
    support = sr.get("support", current * 0.95)

    # ── Keyword routing ───────────────────────────────────────────────────────

    if any(w in q for w in ["price", "current", "today", "now", "trading"]):
        direction = "up" if change >= 0 else "down"
        return (
            f"**{name}** is currently trading at **₹{current:,.1f}** "
            f"({arrow} {abs(change):.1f} / {abs(change_pct):.2f}% today). "
            f"It's {direction} from yesterday's close of ₹{prev:,.1f}."
        )

    elif any(w in q for w in ["predict", "prediction", "forecast", "signal", "buy", "sell", "should i"]):
        direction = pred["direction"]
        conf = pred["confidence"]
        reason = pred["reason"]
        color_word = {"BUY": "bullish", "SELL": "bearish", "HOLD": "neutral"}[direction]
        return (
            f"Based on technical indicators, {name} shows a **{direction}** signal "
            f"with {conf}% confidence. The analysis is {color_word}: {reason}. "
            f"Note: This is a technical signal only — always do your own research before investing."
        )

    elif any(w in q for w in ["rsi", "indicator", "technical", "momentum"]):
        rsi_status = "oversold (bullish territory)" if rsi < 35 else "overbought (bearish territory)" if rsi > 70 else "neutral zone"
        ma_signal = "above" if current > ma20 else "below"
        return (
            f"**Technical indicators for {name}:**\n\n"
            f"- RSI (14): **{rsi:.1f}** — {rsi_status}\n"
            f"- MA10: ₹{ma10:,.1f} | MA20: ₹{ma20:,.1f} | MA50: ₹{ma50:,.1f}\n"
            f"- Price is **{ma_signal}** the 20-day moving average\n"
            f"- MA10 {'>' if ma10 > ma20 else '<'} MA20 suggests {'short-term uptrend' if ma10 > ma20 else 'short-term downtrend'}"
        )

    elif any(w in q for w in ["support", "resistance", "level", "range"]):
        pct_to_res = (resistance - current) / current * 100
        pct_to_sup = (current - support) / current * 100
        return (
            f"**Support & Resistance for {name}:**\n\n"
            f"- 🟢 Support: ₹{support:,.1f} ({pct_to_sup:.1f}% downside)\n"
            f"- 🔴 Resistance: ₹{resistance:,.1f} ({pct_to_res:.1f}% upside)\n"
            f"- Current: ₹{current:,.1f}\n\n"
            f"The stock is positioned {'closer to resistance' if pct_to_res < pct_to_sup else 'closer to support'} right now."
        )

    elif any(w in q for w in ["risk", "volatile", "volatility", "safe"]):
        returns = close.pct_change().dropna()
        daily_std = float(returns.std())
        annual_vol = daily_std * (252 ** 0.5) * 100
        max_dd = float(((close - close.cummax()) / close.cummax()).min() * 100)
        risk_level = "High" if annual_vol > 40 else "Moderate" if annual_vol > 20 else "Low"
        return (
            f"**Risk profile for {name}:**\n\n"
            f"- Annual Volatility: **{annual_vol:.1f}%** → {risk_level} risk\n"
            f"- Max Drawdown (1Y): {max_dd:.1f}%\n"
            f"- Daily Std Dev: {daily_std*100:.2f}%\n\n"
            f"{'Higher volatility means larger price swings — suitable for risk-tolerant investors.' if risk_level == 'High' else 'Moderate volatility — balanced risk profile.' if risk_level == 'Moderate' else 'Low volatility — relatively stable stock.'}"
        )

    elif any(w in q for w in ["52", "high", "low", "week"]):
        high52 = info.get("fiftyTwoWeekHigh", close.tail(252).max())
        low52 = info.get("fiftyTwoWeekLow", close.tail(252).min())
        from_high = (current - high52) / high52 * 100
        from_low = (current - low52) / low52 * 100
        return (
            f"**52-Week range for {name}:**\n\n"
            f"- 52W High: ₹{high52:,.1f} ({from_high:.1f}% from current)\n"
            f"- 52W Low: ₹{low52:,.1f} (+{from_low:.1f}% from current)\n"
            f"- Current: ₹{current:,.1f}"
        )

    elif any(w in q for w in ["about", "company", "what is", "sector", "industry", "overview"]):
        company = info.get("longName", name)
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        mktcap = info.get("marketCap", 0)
        pe = info.get("trailingPE", "N/A")
        return (
            f"**{company}** ({name})\n\n"
            f"- Sector: {sector}\n"
            f"- Industry: {industry}\n"
            f"- Market Cap: ₹{mktcap/1e9:.0f}B\n"
            f"- P/E Ratio: {pe}\n"
            f"- Exchange: NSE (National Stock Exchange, India)"
        )

    elif any(w in q for w in ["volume", "traded"]):
        vol = df["Volume"].iloc[-1]
        avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
        ratio = vol / avg_vol if avg_vol > 0 else 1
        high_low = "above" if ratio > 1 else "below"
        return (
            f"**Volume for {name} today:**\n\n"
            f"- Today's volume: {vol:,.0f} shares\n"
            f"- 20-day avg: {avg_vol:,.0f} shares\n"
            f"- Today is {ratio:.1f}x {high_low} average\n\n"
            f"{'High volume often confirms the price move.' if ratio > 1.5 else 'Normal trading activity today.'}"
        )

    elif any(w in q for w in ["trend", "direction", "up", "down", "bullish", "bearish"]):
        ma_trend = "uptrend" if ma10 > ma20 else "downtrend"
        rsi_trend = "bullish" if rsi > 50 else "bearish"
        mom_5d = (current - float(close.iloc[-5])) / float(close.iloc[-5]) * 100 if len(close) >= 5 else 0
        return (
            f"**Trend analysis for {name}:**\n\n"
            f"- MA Crossover: {ma_trend} (MA10 {'>' if ma10 > ma20 else '<'} MA20)\n"
            f"- RSI trend: {rsi_trend} (RSI {rsi:.0f})\n"
            f"- 5-day momentum: {'+' if mom_5d > 0 else ''}{mom_5d:.1f}%\n"
            f"- Overall: {'Bullish' if pred['score'] > 0 else 'Bearish' if pred['score'] < 0 else 'Neutral'} bias"
        )

    else:
        return (
            f"I can help you with information about **{name}**. Try asking about:\n\n"
            f"- 📈 **Price** — current price and today's movement\n"
            f"- 🤖 **Prediction** — AI buy/sell signal\n"
            f"- ⚡ **RSI / Indicators** — technical analysis\n"
            f"- 🎯 **Support & Resistance** — key levels\n"
            f"- ⚠️ **Risk** — volatility and drawdown\n"
            f"- 📊 **Trend** — direction and momentum\n"
            f"- 🏢 **About** — company overview"
        )
