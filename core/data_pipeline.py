import yfinance as yf
import pandas as pd
import numpy as np

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
}


def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        return df
    except Exception:
        return None


def get_stock_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return info if info else {}
    except Exception:
        return {}
