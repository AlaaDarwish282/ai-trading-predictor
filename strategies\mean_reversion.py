import pandas as pd
import numpy as np


class MeanReversionStrategy:
    """
    Bollinger Band mean-reversion strategy.
    Buys when price touches lower band, sells at upper band.
    """

    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        sma = df["Close"].rolling(self.window).mean()
        std = df["Close"].rolling(self.window).std()
        df["upper"] = sma + self.num_std * std
        df["lower"] = sma - self.num_std * std
        df["mid"] = sma

        df["signal"] = 0
        df.loc[df["Close"] < df["lower"], "signal"] = 1   # oversold → buy
        df.loc[df["Close"] > df["upper"], "signal"] = -1  # overbought → sell

        df["position"] = df["signal"].shift(1).fillna(0)
        df["strategy_return"] = df["position"] * df["Close"].pct_change()
        df["cumulative_return"] = (1 + df["strategy_return"]).cumprod()
        return df

    def performance(self, df: pd.DataFrame) -> dict:
        df = self.generate_signals(df)
        total_return = df["cumulative_return"].iloc[-1] - 1
        sharpe = (df["strategy_return"].mean() / (df["strategy_return"].std() + 1e-10)) * np.sqrt(252)
        max_dd = (df["cumulative_return"] / df["cumulative_return"].cummax() - 1).min()
        return {
            "total_return_pct": round(float(total_return * 100), 2),
            "sharpe_ratio": round(float(sharpe), 3),
            "max_drawdown_pct": round(float(max_dd * 100), 2),
        }
