import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class LSTMPricePredictor:
    """
    LSTM neural network for stock price prediction.
    Architecture: LSTM(128) -> Dropout -> LSTM(64) -> Dropout -> Dense(32) -> Dense(1)
    """

    def __init__(self, seq_len: int = 60, n_features: int = 1):
        self.seq_len = seq_len
        self.n_features = n_features
        self.model = None
        self.history = None

        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

    def build(self) -> None:
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.seq_len, self.n_features)),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="huber",
            metrics=["mae"],
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32) -> dict:
        if self.model is None:
            self.build()

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6),
        ]

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        return {
            "train_loss": float(self.history.history["loss"][-1]),
            "val_loss": float(self.history.history["val_loss"][-1]),
            "epochs_run": len(self.history.history["loss"]),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0).flatten()

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = load_model(path)
