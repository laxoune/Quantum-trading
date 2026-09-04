"""
Qiskit Variational Quantum Circuit (VQC) — LSTM Residual Correction Model
---------------------------------------------------------------------------
Uses the project's existing LSTM backtest output (Predicted Price vs Real
Price) plus VIX and daily % change as auxiliary market-regime features.
A trainable quantum circuit (feature map + variational ansatz, optimized
via COBYLA) learns to predict the LSTM's *residual error*, i.e. it is a
genuine quantum machine-learning correction layer, not a fixed encode-and-
sample step.

Baseline  : LSTM Predicted Price vs Real Price
Quantum   : (LSTM Predicted Price + learned quantum correction) vs Real Price
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_algorithms.optimizers import COBYLA

np.random.seed(42)

# ---------------------------------------------------------------------
# 1. Load & merge existing project data (real LSTM backtest output)
# ---------------------------------------------------------------------
bt = pd.read_csv("calssical_version_1.1/classical_version_1.1.1/Backtesting_Results.csv")
bt["Date"] = pd.to_datetime(bt["Date"]).dt.date
test = pd.read_csv("calssical_version_1.1/classical_version_1.1.1/Testing_Data.csv")
test["Date"] = pd.to_datetime(test["Date"]).dt.date

df = pd.merge(bt, test, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
print(f"Merged dataset: {df.shape[0]} rows (real out-of-sample LSTM backtest days)")

# Target the model actually needs to learn: how wrong was the LSTM?
df["Residual"] = df["Real Price"] - df["Predicted Price"]

features = df[["Predicted Price", "Vix Price", "Pct Change"]].values
target = df["Residual"].values.reshape(-1, 1)

# Time-respecting split (no shuffling — this is a time series)
split = int(len(df) * 0.8)
X_train_raw, X_test_raw = features[:split], features[split:]
y_train_raw, y_test_raw = target[:split], target[split:]

x_scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
y_scaler = MinMaxScaler(feature_range=(-1, 1))

X_train = x_scaler.fit_transform(X_train_raw)
X_test = x_scaler.transform(X_test_raw)
y_train = y_scaler.fit_transform(y_train_raw).flatten()
y_test = y_scaler.transform(y_test_raw).flatten()

print(f"Train: {X_train.shape[0]} days | Test: {X_test.shape[0]} days")

# ---------------------------------------------------------------------
# 2. Build the real trainable quantum circuit
# ---------------------------------------------------------------------
num_qubits = 3  # one per feature: LSTM prediction, VIX, % change

feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=1)
ansatz = RealAmplitudes(num_qubits=num_qubits, reps=2)  # TRAINABLE weights

qnn = EstimatorQNN(
    circuit=feature_map.compose(ansatz),
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
)

print(f"Circuit: {num_qubits} qubits | {len(ansatz.parameters)} trainable weights | "
      f"{qnn.circuit.depth()} depth")

regressor = NeuralNetworkRegressor(
    neural_network=qnn,
    optimizer=COBYLA(maxiter=150),
)

# ---------------------------------------------------------------------
# 3. Train — the circuit's weights are actually optimized here
# ---------------------------------------------------------------------
print("\nTraining variational quantum circuit (COBYLA, 150 iters)...")
regressor.fit(X_train, y_train)
print("Done.")

# ---------------------------------------------------------------------
# 4. Evaluate: does the learned quantum correction beat the raw LSTM?
# ---------------------------------------------------------------------
pred_residual_scaled = regressor.predict(X_test)
pred_residual = y_scaler.inverse_transform(pred_residual_scaled.reshape(-1, 1)).flatten()

real_price_test = df["Real Price"].values[split:]
lstm_price_test = df["Predicted Price"].values[split:]
quantum_corrected_price = lstm_price_test + pred_residual

lstm_mse = mean_squared_error(real_price_test, lstm_price_test)
lstm_mae = mean_absolute_error(real_price_test, lstm_price_test)
quantum_mse = mean_squared_error(real_price_test, quantum_corrected_price)
quantum_mae = mean_absolute_error(real_price_test, quantum_corrected_price)

print("\n=== RESULTS (held-out test days) ===")
print(f"LSTM alone            -> MSE: {lstm_mse:.4f} | MAE: {lstm_mae:.4f}")
print(f"LSTM + Quantum-VQC     -> MSE: {quantum_mse:.4f} | MAE: {quantum_mae:.4f}")
improvement = (lstm_mse - quantum_mse) / lstm_mse * 100
print(f"MSE change vs LSTM baseline: {improvement:+.2f}%")

results = pd.DataFrame({
    "Date": df["Date"].values[split:],
    "Real Price": real_price_test,
    "LSTM Prediction": lstm_price_test,
    "Quantum-Corrected Prediction": quantum_corrected_price,
})
results.to_csv("qiskit_vqc_test_results.csv", index=False)
print("\nSaved test predictions to qiskit_vqc_test_results.csv")
