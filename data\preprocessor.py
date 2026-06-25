import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to a price DataFrame."""
    close = df["Close"]

    # Moving averages
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]

    # Volume indicators
    df["Volume_SMA"] = df["Volume"].rolling(20).mean()
    df["Volume_ratio"] = df["Volume"] / (df["Volume_SMA"] + 1e-10)

    # Price momentum
    df["ROC_5"] = close.pct_change(5)
    df["ROC_20"] = close.pct_change(20)

    df.dropna(inplace=True)
    return df


def create_sequences(data: np.ndarray, seq_len: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Create (X, y) sequences for LSTM training."""
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 0])   # predict Close price (index 0)
    return np.array(X), np.array(y)


def scale_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols].values)
    return scaled, scaler
