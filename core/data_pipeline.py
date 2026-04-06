import yfinance as yf
import pandas as pd
import numpy as np

# ── Known stocks dropdown list (for the selectbox) ───────────────────────────
STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Wipro": "WIPRO.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "HUL": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "L&T": "LT.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "ONGC": "ONGC.NS",
    "NTPC": "NTPC.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Zomato": "ZOMATO.NS",
    "Paytm": "PAYTM.NS",
    "IRCTC": "IRCTC.NS",
    "Nykaa": "NYKAA.NS",
    "Tata Power": "TATAPOWER.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Adani Green": "ADANIGREEN.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Britannia": "BRITANNIA.NS",
    "Cipla": "CIPLA.NS",
    "Coal India": "COALINDIA.NS",
    "Divis Labs": "DIVISLAB.NS",
    "Dr Reddys": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Grasim": "GRASIM.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Hindalco": "HINDALCO.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Power Grid": "POWERGRID.NS",
    "Tech Mahindra": "TECHM.NS",
    "Titan": "TITAN.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Vedanta": "VEDL.NS",
    "Nestle India": "NESTLEIND.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "BPCL": "BPCL.NS",
    "SAIL": "SAIL.NS",
    "NMDC": "NMDC.NS",
    "PNB": "PNB.NS",
    "Bank of Baroda": "BANKBARODA.NS",
    "IRFC": "IRFC.NS",
    "RVNL": "RVNL.NS",
    "HAL": "HAL.NS",
    "BEL": "BEL.NS",
    "Godrej Consumer": "GODREJCP.NS",
    "Pidilite": "PIDILITIND.NS",
    "SBI Life": "SBILIFE.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "ICICI Prudential": "ICICIPRULI.NS",
    "LIC": "LICI.NS",
    "Mankind Pharma": "MANKIND.NS",
    "Varun Beverages": "VBL.NS",
    "Polycab": "POLYCAB.NS",
    "Havells": "HAVELLS.NS",
    "Voltas": "VOLTAS.NS",
    "ABB India": "ABB.NS",
    "Siemens": "SIEMENS.NS",
}


def resolve_ticker(raw: str) -> str:
    """
    Convert any user input to a valid Yahoo Finance ticker.
    Handles: symbol only (RELIANCE), with suffix (RELIANCE.NS),
    BSE code, full name match, partial match.
    """
    raw = raw.strip().upper()

    # Already has exchange suffix
    if raw.endswith(".NS") or raw.endswith(".BO"):
        return raw

    # Index
    if raw.startswith("^"):
        return raw

    # Check known name map (case-insensitive)
    raw_lower = raw.lower()
    for name, sym in STOCKS.items():
        if name.lower() == raw_lower:
            return sym
        base = sym.replace(".NS", "").lower()
        if base == raw_lower:
            return sym

    # Default: append .NS for NSE
    return raw + ".NS"


def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    """
    Fetch OHLCV data from Yahoo Finance.
    Tries NSE first, then BSE as fallback.
    Works for ANY Indian stock — not just the dropdown list.
    """
    try:
        # Try as-is first (may already have .NS)
        df = yf.Ticker(ticker).history(period=period)
        if df is not None and not df.empty:
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.dropna(inplace=True)
            return df
    except Exception:
        pass

    # Fallback: try .BO (BSE)
    try:
        base = ticker.replace(".NS", "").replace(".BO", "")
        df = yf.Ticker(base + ".BO").history(period=period)
        if df is not None and not df.empty:
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.dropna(inplace=True)
            return df
    except Exception:
        pass

    return None


def get_stock_info(ticker: str) -> dict:
    """Fetch stock metadata from Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).info
        return info if info else {}
    except Exception:
        return {}
