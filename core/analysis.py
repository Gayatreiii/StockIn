import numpy as np
import pandas as pd


def get_technical_indicators(df: pd.DataFrame) -> dict:
    if df is None or df.empty or len(df) < 20:
        return {}

    close = df["Close"]

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    # Moving averages
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean() if len(close) >= 50 else close.rolling(len(close)).mean()

    # Bollinger Bands (20-day, 2 std)
    std20 = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20

    return {
        "RSI": round(float(rsi.iloc[-1]), 2),
        "MA10": round(float(ma10.iloc[-1]), 2),
        "MA20": round(float(ma20.iloc[-1]), 2),
        "MA50": round(float(ma50.iloc[-1]), 2),
        "BB_Upper": round(float(bb_upper.iloc[-1]), 2),
        "BB_Lower": round(float(bb_lower.iloc[-1]), 2),
    }


def get_support_resistance(df: pd.DataFrame) -> dict:
    if df is None or len(df) < 20:
        return {}

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    current = float(close.iloc[-1])
    resistance = float(high.tail(60).max())
    support = float(low.tail(60).min())

    # Tighter S/R using recent 20 days
    res_near = float(high.tail(20).max())
    sup_near = float(low.tail(20).min())

    return {
        "current": round(current, 2),
        "resistance": round(res_near, 2),
        "support": round(sup_near, 2),
        "resistance_strong": round(resistance, 2),
        "support_strong": round(support, 2),
    }


def simple_prediction(df: pd.DataFrame) -> dict:
    """
    Simple rule-based prediction using RSI + MA crossover + momentum.
    Straightforward enough for academic review.
    """
    if df is None or df.empty or len(df) < 20:
        return {"direction": "HOLD", "confidence": 50, "reason": "Insufficient data"}

    indicators = get_technical_indicators(df)
    close = df["Close"]

    rsi = indicators.get("RSI", 50)
    ma10 = indicators.get("MA10", close.iloc[-1])
    ma20 = indicators.get("MA20", close.iloc[-1])
    current = float(close.iloc[-1])

    # Score system: positive = bullish, negative = bearish
    score = 0
    reasons = []

    # RSI signal
    if rsi < 35:
        score += 2
        reasons.append(f"RSI {rsi:.0f} oversold")
    elif rsi > 65:
        score -= 2
        reasons.append(f"RSI {rsi:.0f} overbought")
    elif rsi > 50:
        score += 1
        reasons.append(f"RSI {rsi:.0f} bullish zone")
    else:
        score -= 1
        reasons.append(f"RSI {rsi:.0f} bearish zone")

    # MA crossover
    if ma10 > ma20:
        score += 2
        reasons.append("MA10 > MA20 uptrend")
    else:
        score -= 2
        reasons.append("MA10 < MA20 downtrend")

    # Price vs MA20
    if current > ma20:
        score += 1
        reasons.append("Price above MA20")
    else:
        score -= 1
        reasons.append("Price below MA20")

    # 5-day momentum
    if len(close) >= 5:
        mom = (current - float(close.iloc[-5])) / float(close.iloc[-5]) * 100
        if mom > 1.5:
            score += 1
            reasons.append(f"+{mom:.1f}% momentum")
        elif mom < -1.5:
            score -= 1
            reasons.append(f"{mom:.1f}% momentum")

    # Map score to signal
    if score >= 3:
        direction = "BUY"
        confidence = min(55 + score * 5, 82)
    elif score <= -3:
        direction = "SELL"
        confidence = min(55 + abs(score) * 5, 82)
    else:
        direction = "HOLD"
        confidence = 50 + abs(score) * 3

    return {
        "direction": direction,
        "confidence": confidence,
        "reason": " · ".join(reasons[:3]),
        "score": score,
    }
