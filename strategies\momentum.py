import pandas as pd
import numpy as np


class MomentumStrategy:
    """
    Momentum-based trading strategy.
    Buys when short-term momentum is positive and RSI is not overbought.
    """

    def __init__(self, fast: int = 10, slow: int = 30, rsi_threshold: float = 70.0):
        self.fast = fast
        self.slow = slow
        self.rsi_threshold = rsi_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["fast_ma"] = df["Close"].rolling(self.fast).mean()
        df["slow_ma"] = df["Close"].rolling(self.slow).mean()

        # RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

        df["signal"] = 0
        df.loc[(df["fast_ma"] > df["slow_ma"]) & (df["rsi"] < self.rsi_threshold), "signal"] = 1
        df.loc[(df["fast_ma"] < df["slow_ma"]), "signal"] = -1

        df["position"] = df["signal"].shift(1).fillna(0)
        df["strategy_return"] = df["position"] * df["Close"].pct_change()
        df["cumulative_return"] = (1 + df["strategy_return"]).cumprod()
        return df

    def performance(self, df: pd.DataFrame) -> dict:
        df = self.generate_signals(df)
        total_return = df["cumulative_return"].iloc[-1] - 1
        sharpe = (df["strategy_return"].mean() / (df["strategy_return"].std() + 1e-10)) * np.sqrt(252)
        max_dd = (df["cumulative_return"] / df["cumulative_return"].cummax() - 1).min()
        wins = (df["strategy_return"] > 0).sum()
        total_trades = (df["signal"].diff() != 0).sum()
        return {
            "total_return_pct": round(float(total_return * 100), 2),
            "sharpe_ratio": round(float(sharpe), 3),
            "max_drawdown_pct": round(float(max_dd * 100), 2),
            "win_rate": round(float(wins / max(total_trades, 1)), 3),
            "total_trades": int(total_trades),
        }
