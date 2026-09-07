"""
Hybrid Quantum Pipeline: Forecast -> Portfolio Optimization
---------------------------------------------------------------------------
This connects the two previously separate pieces into one real pipeline,
Haydock's original hybrid-model idea:

  STAGE 1 (forecasting)   qiskit_vqc_correction.py
    Trainable Qiskit VQC corrects an LSTM's residual error for ONE real,
    historically-traded asset ("Tech"), using its actual backtest data.

  STAGE 2 (allocation)    qaoa_portfolio_optimization.py
    QAOA solves a Markowitz QUBO to pick the best combination of assets
    to hold, given expected-return estimates.

  THE WEAVE: Stage 1's real, data-derived forecast becomes the expected
  return for "Tech" in Stage 2's optimizer, instead of a made-up number.

HONESTY NOTE: only "Tech" has a genuine forecast, derived from real
historical price/VIX/%-change data via the trained VQC. The other four
assets (Utility, Energy, Healthcare, Bond-like) still use illustrative
placeholder estimates — extending real forecasts to them would require
the same data-collection + training pipeline run per asset, a natural
next step, not something to fake here.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_algorithms.optimizers import COBYLA as VQC_COBYLA

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA as QAOA_COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler

np.random.seed(42)
algorithm_globals.random_seed = 42

# ===========================================================================
# STAGE 1 — VQC forecast for the one real asset ("Tech")
# ===========================================================================
print("=" * 70)
print("STAGE 1: Training VQC forecast-correction model on real asset data")
print("=" * 70)

bt = pd.read_csv("data/backtesting_results.csv")
bt["Date"] = pd.to_datetime(bt["Date"]).dt.date
test = pd.read_csv("data/testing_data.csv")
test["Date"] = pd.to_datetime(test["Date"]).dt.date
df = pd.merge(bt, test, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
df["Residual"] = df["Real Price"] - df["Predicted Price"]

features = df[["Predicted Price", "Vix Price", "Pct Change"]].values
target = df["Residual"].values.reshape(-1, 1)
split = int(len(df) * 0.8)

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
regressor = NeuralNetworkRegressor(neural_network=qnn, optimizer=VQC_COBYLA(maxiter=150))
regressor.fit(X_train, y_train)

pred_residual = y_scaler.inverse_transform(
    regressor.predict(X_test).reshape(-1, 1)
).flatten()
lstm_price_test = df["Predicted Price"].values[split:]
quantum_corrected_price = lstm_price_test + pred_residual

# Derive an annualized expected return from the quantum-corrected forecast path
daily_returns = np.diff(quantum_corrected_price) / quantum_corrected_price[:-1]
tech_expected_return = np.mean(daily_returns) * 252  # annualize
tech_realized_vol = np.std(daily_returns) * np.sqrt(252)

print(f"\nQuantum-corrected forecast path -> implied annualized return: "
      f"{tech_expected_return:+.2%}")
print(f"Implied annualized volatility: {tech_realized_vol:.2%}")

# ===========================================================================
# STAGE 2 — feed that real forecast into the QAOA portfolio optimizer
# ===========================================================================
print("\n" + "=" * 70)
print("STAGE 2: QAOA portfolio optimization, using Stage 1's real forecast")
print("=" * 70)

assets = ["Tech", "Utility", "Energy", "Healthcare", "Bond-like"]
n = len(assets)

# mu[0] = REAL, data-derived forecast from Stage 1.
# mu[1:] = illustrative placeholders (see honesty note above).
mu = np.array([
    tech_expected_return,   # <-- real, from the VQC forecast
    0.05,                   # Utility (placeholder)
    0.09,                   # Energy (placeholder)
    0.10,                   # Healthcare (placeholder)
    0.03,                   # Bond-like (placeholder)
])

vol = np.array([tech_realized_vol, 0.10, 0.22, 0.16, 0.05])
corr = np.array([
    [1.00, 0.10, 0.30, 0.20, -0.05],
    [0.10, 1.00, 0.15, 0.10,  0.20],
    [0.30, 0.15, 1.00, 0.05, -0.10],
    [0.20, 0.10, 0.05, 1.00,  0.05],
    [-0.05, 0.20, -0.10, 0.05, 1.00],
])
sigma = np.outer(vol, vol) * corr

print("\nAssets:", assets)
print("Expected returns (mu) fed to optimizer:", np.round(mu, 4))
print("  ^ mu[0] (Tech) is real; mu[1:] are illustrative placeholders")

risk_factor = 0.5
budget = 3
qp = QuadraticProgram(name="hybrid_portfolio")
for a in assets:
    qp.binary_var(name=a)
linear = {assets[i]: -mu[i] for i in range(n)}
quadratic = {(assets[i], assets[j]): risk_factor * sigma[i, j]
             for i in range(n) for j in range(n)}
qp.minimize(linear=linear, quadratic=quadratic)
qp.linear_constraint(linear={a: 1 for a in assets}, sense="==", rhs=budget, name="budget")

exact_result = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
exact_portfolio = [assets[i] for i, bit in enumerate(exact_result.x) if bit == 1]

qaoa = QAOA(sampler=StatevectorSampler(), optimizer=QAOA_COBYLA(maxiter=200), reps=3)
qaoa_result = MinimumEigenOptimizer(qaoa).solve(qp)
qaoa_portfolio = [assets[i] for i, bit in enumerate(qaoa_result.x) if bit == 1]

print("\n=== RESULTS ===")
print("Exact (classical) portfolio:", exact_portfolio, "| objective:", round(exact_result.fval, 5))
print("QAOA (quantum) portfolio:   ", qaoa_portfolio, "| objective:", round(qaoa_result.fval, 5))
print("QAOA matched exact optimum:", set(exact_portfolio) == set(qaoa_portfolio))

print("\n" + "=" * 70)
print("PIPELINE SUMMARY")
print("=" * 70)
print("Stage 1 (VQC) produced a real, data-derived return forecast for Tech.")
print("Stage 2 (QAOA) used that forecast, alongside placeholder estimates for")
print("the other four assets, to select an optimal 3-asset portfolio.")
print("Next step to make this fully real: run Stage 1's forecasting pipeline")
print("on historical data for Utility, Energy, Healthcare, and Bond-like too.")
