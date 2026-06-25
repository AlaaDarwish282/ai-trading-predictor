import pandas as pd
import numpy as np
from typing import Callable


class BacktestEngine:
    """
    Event-driven backtesting engine with position sizing and transaction costs.
    """

    def __init__(self, initial_capital: float = 100_000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission  # 0.1% per trade

    def run(self, df: pd.DataFrame, signals: pd.Series) -> dict:
        capital = self.initial_capital
        position = 0
        trades = []
        portfolio_values = []

        for i, (date, row) in enumerate(df.iterrows()):
            price = row["Close"]
            signal = signals.iloc[i] if i < len(signals) else 0
            trade_cost = 0

            if signal == 1 and position == 0:
                shares = int(capital * 0.95 / price)
                cost = shares * price * (1 + self.commission)
                if cost <= capital:
                    capital -= cost
                    position = shares
                    trade_cost = shares * price * self.commission
                    trades.append({"date": date, "type": "BUY", "price": price, "shares": shares})

            elif signal == -1 and position > 0:
                proceeds = position * price * (1 - self.commission)
                trade_cost = position * price * self.commission
                capital += proceeds
                trades.append({"date": date, "type": "SELL", "price": price, "shares": position})
                position = 0

            portfolio_value = capital + position * price
            portfolio_values.append({"date": date, "value": portfolio_value})

        portfolio_df = pd.DataFrame(portfolio_values).set_index("date")
        final_value = portfolio_df["value"].iloc[-1]
        returns = portfolio_df["value"].pct_change().dropna()

        buy_hold_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

        return {
            "initial_capital": self.initial_capital,
            "final_value": round(final_value, 2),
            "total_return_pct": round((final_value / self.initial_capital - 1) * 100, 2),
            "buy_hold_return_pct": round(float(buy_hold_return), 2),
            "sharpe_ratio": round(float(returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)), 3),
            "max_drawdown_pct": round(float((portfolio_df["value"] / portfolio_df["value"].cummax() - 1).min() * 100), 2),
            "num_trades": len(trades),
            "trades": trades[:10],  # return first 10 trades as sample
        }
