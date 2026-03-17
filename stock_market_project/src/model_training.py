import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError as e:
    print(f"Warning: XGBoost not available: {e}")
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.models import Sequential
    TENSORFLOW_AVAILABLE = True
except ImportError:
    LSTM = None
    Dense = None
    Sequential = None
    TENSORFLOW_AVAILABLE = False


def train_regression_models(df: pd.DataFrame, target_col: str = "Close") -> dict:
    """Train Linear Regression, Random Forest, and XGBoost models on features derived from stock data."""
    df = df.copy()

    # Drop non-feature columns and ensure only numeric data is used for training.
    features = df.drop(columns=["Date", target_col])
    # Remove any non-numeric columns that may exist in the data (e.g., ticker/name strings).
    features = features.select_dtypes(include=[np.number])

    targets = pd.to_numeric(df[target_col], errors="coerce")

    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    models["linear_regression"] = {
        "model": lr,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": lr.predict(X_test_scaled),
    }

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    models["random_forest"] = {
        "model": rf,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": rf.predict(X_test_scaled),
    }

    # XGBoost
    if XGBOOST_AVAILABLE:
        xgb = XGBRegressor(n_estimators=100, random_state=42)
        xgb.fit(X_train_scaled, y_train)
        models["xgboost"] = {
            "model": xgb,
            "scaler": scaler,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": xgb.predict(X_test_scaled),
        }

    return models


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute evaluation metrics for regression predictions."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
    }


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate direction accuracy: proportion of times predicted direction matches actual."""
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0

    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float((true_dir == pred_dir).mean())


def build_lstm_model(input_shape, units=32):
    if Sequential is None:
        raise ImportError("TensorFlow/Keras is required for LSTM model")

    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape, return_sequences=False))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="mse")
    return model


def create_lstm_dataset(series: pd.Series, lookback: int = 20):
    """Build sequences for training LSTM models."""
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback : i].values)
        y.append(series[i])
    X = np.array(X)
    y = np.array(y)
    return X, y
