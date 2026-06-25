# AI Trading Predictor

A machine learning system for financial market analysis and trading strategy backtesting. Combines **LSTM neural networks**, **XGBoost regression**, and rule-based strategies with a full event-driven backtesting engine.

## Features

- 📈 **Real-time market data** via Yahoo Finance
- 🧠 **LSTM price forecasting** (multi-layer with early stopping)
- 🌲 **XGBoost predictor** with time-series cross-validation
- ⚡ **Momentum & Mean-Reversion strategies** with signal generation
- 🔁 **Event-driven backtesting engine** with transaction costs & position sizing
- 📊 **Performance metrics**: Sharpe ratio, max drawdown, CAGR, win rate
- 🚀 **FastAPI REST interface**

## Architecture

```
Market Data (Yahoo Finance)
        │
        ▼
  [Data Preprocessor]  ← Technical Indicators (RSI, MACD, BB, SMA)
        │
        ├──► [LSTM Model]        ← Price Forecasting
        ├──► [XGBoost Model]     ← Feature-based Prediction
        └──► [Strategy Engine]   ← Signal Generation
                    │
                    ▼
          [Backtesting Engine]   ← P&L, Sharpe, Drawdown
```

## Quick Start

```bash
git clone https://github.com/AlaaDarwish282/ai-trading-predictor.git
cd ai-trading-predictor
pip install -r requirements.txt
python main.py
```

## API Usage

```bash
# Backtest a momentum strategy on Apple stock
curl -X POST http://localhost:8001/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "strategy": "momentum", "period": "2y", "initial_capital": 100000}'

# Get fundamentals
curl http://localhost:8001/api/v1/fundamentals/TSLA
```

## Technical Indicators

| Indicator | Description |
|-----------|-------------|
| SMA 20/50 | Simple moving averages |
| EMA 12/26 | Exponential moving averages |
| MACD | Trend-following momentum |
| RSI | Relative strength index |
| Bollinger Bands | Volatility bands |
| ROC | Rate of change (5d, 20d) |

## Disclaimer

This project is for educational purposes only. Do not use it for actual trading decisions.

## License

MIT
