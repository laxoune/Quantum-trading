"""
Fully Real Hybrid Pipeline: 5-Asset LSTM+VQC Forecasts -> QAOA Allocation
---------------------------------------------------------------------------
Loads the real, data-derived expected returns and covariance matrix
produced by multi_asset_forecast.py (all 5 assets, no placeholders) and
feeds them into the QAOA portfolio optimizer.
"""
import numpy as np

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler

algorithm_globals.random_seed = 42

assets = ["Tech", "Utility", "Energy", "Healthcare", "Bond-like"]
n = len(assets)

mu = np.load("real_mu.npy")
sigma = np.load("real_cov.npy")

print("Real, data-derived expected returns (annualized):")
for a, m in zip(assets, mu):
    print(f"  {a:12s} {m:+.2%}")

risk_factor = 0.5
budget = 3
qp = QuadraticProgram(name="real_hybrid_portfolio")
for a in assets:
    qp.binary_var(name=a)
linear = {assets[i]: -mu[i] for i in range(n)}
quadratic = {(assets[i], assets[j]): risk_factor * sigma[i, j]
             for i in range(n) for j in range(n)}
qp.minimize(linear=linear, quadratic=quadratic)
qp.linear_constraint(linear={a: 1 for a in assets}, sense="==", rhs=budget, name="budget")

exact_result = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
exact_portfolio = [assets[i] for i, bit in enumerate(exact_result.x) if bit == 1]

qaoa = QAOA(sampler=StatevectorSampler(), optimizer=COBYLA(maxiter=200), reps=3)
qaoa_result = MinimumEigenOptimizer(qaoa).solve(qp)
qaoa_portfolio = [assets[i] for i, bit in enumerate(qaoa_result.x) if bit == 1]

print("\n=== RESULTS (fully real inputs, no placeholders) ===")
print("Exact (classical) portfolio:", exact_portfolio, "| objective:", round(exact_result.fval, 5))
print("QAOA (quantum) portfolio:   ", qaoa_portfolio, "| objective:", round(qaoa_result.fval, 5))
print("QAOA matched exact optimum:", set(exact_portfolio) == set(qaoa_portfolio))
