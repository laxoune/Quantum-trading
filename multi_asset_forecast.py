"""
Multi-Asset LSTM + Qiskit VQC Forecasting Pipeline
---------------------------------------------------------------------------
Extends the single-asset (Tech-only) forecasting model to all 5 assets in
the portfolio universe, using real historical data pulled via yfinance.

For each asset:
  1. Train a small LSTM on the training split (one-step-ahead price forecast)
  2. Generate one-step-ahead predictions on the testing split
  3. Train a Qiskit VQC to correct the LSTM's residual error, using the
     LSTM prediction + VIX + % change as features (same method as the
     original Tech-only model)
  4. Derive an annualized expected return from the quantum-corrected
     forecast path

Output: real mu (expected returns) for all 5 assets, AND a real covariance
matrix computed from actual historical daily returns (replacing the
illustrative placeholder numbers used previously).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
tf.get_logger().setLevel("ERROR")

from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_algorithms.optimizers import COBYLA

np.random.seed(42)
tf.random.set_seed(42)

ASSETS = ["tech", "utility", "energy", "healthcare", "bond_like"]
LABELS = {"tech": "Tech", "utility": "Utility", "energy": "Energy",
          "healthcare": "Healthcare", "bond_like": "Bond-like"}
WINDOW = 20  # days of history the LSTM looks back on


def build_lstm_sequences(closes: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(closes)):
        X.append(closes[i - window:i])
        y.append(closes[i])
    return np.array(X), np.array(y)


def train_lstm_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame):
    scaler = MinMaxScaler()
    train_close = scaler.fit_transform(train_df[["Close"]].values)

    X_train, y_train = build_lstm_sequences(train_close.flatten(), WINDOW)
    X_train = X_train.reshape(-1, WINDOW, 1)

    model = Sequential([
        LSTM(32, input_shape=(WINDOW, 1)),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

    # One-step-ahead predictions across the test period, using the real
    # trailing window each time (not recursive multi-step forecasting)
    full_close = np.concatenate([train_df["Close"].values, test_df["Close"].values])
    full_scaled = scaler.transform(full_close.reshape(-1, 1)).flatten()

    start = len(train_df)
    X_test = []
    for i in range(start, len(full_scaled)):
        X_test.append(full_scaled[i - WINDOW:i])
    X_test = np.array(X_test).reshape(-1, WINDOW, 1)

    pred_scaled = model.predict(X_test, verbose=0)
    predicted_price = scaler.inverse_transform(pred_scaled).flatten()
    return predicted_price


def train_vqc_correction(predicted_price, real_price, vix, pct_change):
    residual = real_price - predicted_price
    features = np.column_stack([predicted_price, vix, pct_change])
    target = residual.reshape(-1, 1)

    split = int(len(features) * 0.8)
    x_scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = x_scaler.fit_transform(features[:split])
    X_test = x_scaler.transform(features[split:])
    y_train = y_scaler.fit_transform(target[:split]).flatten()

    feature_map = ZFeatureMap(feature_dimension=3, reps=1)
    ansatz = RealAmplitudes(num_qubits=3, reps=2)
    qnn = EstimatorQNN(circuit=feature_map.compose(ansatz),
                        input_params=feature_map.parameters,
                        weight_params=ansatz.parameters)
    regressor = NeuralNetworkRegressor(neural_network=qnn, optimizer=COBYLA(maxiter=100))
    regressor.fit(X_train, y_train)

    pred_residual = y_scaler.inverse_transform(
        regressor.predict(X_test).reshape(-1, 1)
    ).flatten()
    corrected_price = predicted_price[split:] + pred_residual
    real_price_test = real_price[split:]

    baseline_mse = mean_squared_error(real_price_test, predicted_price[split:])
    corrected_mse = mean_squared_error(real_price_test, corrected_price)

    return corrected_price, baseline_mse, corrected_mse


results = {}
daily_return_series = {}

for asset in ASSETS:
    label = LABELS[asset]
    print(f"\n{'=' * 60}\n{label} ({asset})\n{'=' * 60}")

    train_df = pd.read_csv(f"{asset}_training_data.csv")
    test_df = pd.read_csv(f"{asset}_testing_data.csv")

    print("Training LSTM...")
    predicted_price = train_lstm_and_predict(train_df, test_df)
    real_price = test_df["Close"].values
    vix = test_df["Vix Price"].values
    pct_change = test_df["Pct Change"].values

    print("Training VQC correction...")
    corrected_price, baseline_mse, corrected_mse = train_vqc_correction(
        predicted_price, real_price, vix, pct_change
    )

    daily_ret = np.diff(corrected_price) / corrected_price[:-1]
    annualized_return = np.mean(daily_ret) * 252
    annualized_vol = np.std(daily_ret) * np.sqrt(252)

    print(f"LSTM MSE: {baseline_mse:.4f} -> LSTM+VQC MSE: {corrected_mse:.4f} "
          f"({(baseline_mse - corrected_mse) / baseline_mse * 100:+.1f}%)")
    print(f"Implied annualized return: {annualized_return:+.2%} | vol: {annualized_vol:.2%}")

    results[label] = {
        "mu": annualized_return,
        "vol": annualized_vol,
        "baseline_mse": baseline_mse,
        "corrected_mse": corrected_mse,
    }
    # Save arrays for visualization
    split_idx = int(len(predicted_price) * 0.8)
    pd.DataFrame({
        "Real Price": real_price[split_idx:],
        "LSTM Prediction": predicted_price[split_idx:],
        "Quantum-Corrected Prediction": corrected_price,
    }).to_csv(f"{asset}_forecast_comparison.csv", index=False)
    # Use raw historical daily returns (not just the test-window forecast)
    # for the covariance matrix -- more data, more stable correlations
    full_close = pd.concat([train_df["Close"], test_df["Close"]]).values
    daily_return_series[label] = pd.Series(full_close).pct_change().dropna().values

# ---------------------------------------------------------------------
# Build a REAL covariance matrix from actual historical daily returns
# ---------------------------------------------------------------------
min_len = min(len(v) for v in daily_return_series.values())
returns_matrix = np.column_stack([daily_return_series[a][-min_len:] for a in LABELS.values()])
real_cov_annualized = np.cov(returns_matrix.T) * 252

print("\n" + "=" * 60)
print("SUMMARY -- real, data-derived inputs for the portfolio optimizer")
print("=" * 60)
mu_vector = np.array([results[a]["mu"] for a in LABELS.values()])
for a, m in zip(LABELS.values(), mu_vector):
    print(f"  {a:12s} expected return: {m:+.2%}")

np.save("real_mu.npy", mu_vector)
np.save("real_cov.npy", real_cov_annualized)
print("\nSaved real_mu.npy and real_cov.npy for the portfolio optimizer.")
