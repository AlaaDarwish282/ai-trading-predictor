import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import joblib
import math


class XGBoostPredictor:
    """
    XGBoost-based price predictor using technical indicators as features.
    Uses time-series cross-validation to avoid look-ahead bias.
    """

    DEFAULT_PARAMS = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self, params: dict = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = XGBRegressor(**self.params)
        self.feature_names: list[str] = []

    def train(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
        self.feature_names = list(X.columns)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            m = XGBRegressor(**self.params)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = m.predict(X_val)
            cv_scores.append(mean_absolute_error(y_val, preds))

        # Final fit on full data
        self.model.fit(X, y, verbose=False)

        return {
            "cv_mae_mean": float(np.mean(cv_scores)),
            "cv_mae_std": float(np.std(cv_scores)),
            "feature_importance": dict(zip(self.feature_names, self.model.feature_importances_.tolist())),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        preds = self.predict(X)
        mae = mean_absolute_error(y, preds)
        rmse = math.sqrt(mean_squared_error(y, preds))
        mape = float(np.mean(np.abs((y.values - preds) / (y.values + 1e-10))) * 100)
        return {"mae": mae, "rmse": rmse, "mape": mape}

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
