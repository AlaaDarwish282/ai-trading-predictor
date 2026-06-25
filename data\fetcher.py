import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


def fetch_ohlcv(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data for a given ticker from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)
    df.dropna(inplace=True)
    return df


def fetch_multiple(tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    """Fetch data for multiple tickers."""
    return {t: fetch_ohlcv(t, period=period) for t in tickers}


def get_fundamentals(ticker: str) -> dict:
    """Fetch key fundamental metrics."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "pe_ratio": info.get("trailingPE"),
        "market_cap": info.get("marketCap"),
        "revenue": info.get("totalRevenue"),
        "eps": info.get("trailingEps"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
    }
