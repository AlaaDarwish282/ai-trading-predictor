from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from data.fetcher import fetch_ohlcv, get_fundamentals
from data.preprocessor import add_technical_indicators
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from backtesting.engine import BacktestEngine

router = APIRouter(prefix="/api/v1", tags=["trading"])


class PredictRequest(BaseModel):
    ticker: str
    strategy: str = "momentum"  # "momentum" | "mean_reversion"
    period: str = "2y"
    initial_capital: float = 100000.0


@router.post("/analyze")
async def analyze(req: PredictRequest):
    try:
        df = fetch_ohlcv(req.ticker, period=req.period)
        df = add_technical_indicators(df)

        if req.strategy == "momentum":
            strat = MomentumStrategy()
        else:
            strat = MeanReversionStrategy()

        result_df = strat.generate_signals(df)
        perf = strat.performance(df)

        engine = BacktestEngine(initial_capital=req.initial_capital)
        backtest = engine.run(df, result_df["signal"])

        return {
            "ticker": req.ticker,
            "strategy": req.strategy,
            "period": req.period,
            "performance": perf,
            "backtest": backtest,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fundamentals/{ticker}")
async def fundamentals(ticker: str):
    try:
        return get_fundamentals(ticker)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    return {"status": "ok", "service": "ai-trading-predictor"}
