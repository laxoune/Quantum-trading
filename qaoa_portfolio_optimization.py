"""
Qiskit QAOA — Quantum Portfolio Optimization
---------------------------------------------------------------------------
This is the OTHER half of "quantum in finance": not price forecasting
(that's qiskit_vqc_correction.py), but the textbook quantum-finance
problem — deciding WHICH assets to hold, formulated as a QUBO and solved
with QAOA, a hybrid quantum-classical algorithm.

Terms, in order of use:
  QUBO   — Quadratic Unconstrained Binary Optimization. The problem written
           as "choose 0/1 for each asset to minimize this equation," which
           balances expected return against risk (variance + covariance).
  QAOA   — Quantum Approximate Optimization Algorithm. A parameterized
           quantum circuit that proposes candidate 0/1 portfolios; measured
           outcomes are scored against the QUBO objective.
  COBYLA — the classical optimizer that tunes QAOA's circuit parameters
           between rounds (same role it played for the VQC forecasting
           model — the classical half of every hybrid quantum-classical
           loop).

NOTE ON DATA: this uses a small, clearly-synthetic 5-asset universe
(distinct return/volatility/correlation profiles standing in for a tech
stock, a utility, an energy stock, a healthcare stock, and a bond-like
low-vol asset) to demonstrate the method. It is NOT live market data —
label it as illustrative if this goes in front of anyone technical.
"""
import numpy as np

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler

algorithm_globals.random_seed = 42
np.random.seed(42)

# ---------------------------------------------------------------------
# 1. Define a small, clearly-synthetic 5-asset universe
# ---------------------------------------------------------------------
assets = ["Tech", "Utility", "Energy", "Healthcare", "Bond-like"]
n = len(assets)

# Expected annual returns (mu) — illustrative, distinct risk/return profiles
mu = np.array([0.14, 0.05, 0.09, 0.10, 0.03])

# Volatilities (annualized)
vol = np.array([0.28, 0.10, 0.22, 0.16, 0.05])

# A plausible correlation structure (tech/energy move together more than
# tech/bond, etc.) — synthetic but structurally realistic
corr = np.array([
    [1.00, 0.10, 0.30, 0.20, -0.05],
    [0.10, 1.00, 0.15, 0.10,  0.20],
    [0.30, 0.15, 1.00, 0.05, -0.10],
    [0.20, 0.10, 0.05, 1.00,  0.05],
    [-0.05, 0.20, -0.10, 0.05, 1.00],
])
sigma = np.outer(vol, vol) * corr  # covariance matrix

print("Assets:", assets)
print("Expected returns (mu):", mu)
print()

# ---------------------------------------------------------------------
# 2. Formulate as a QUBO: pick a fixed-size portfolio balancing
#    return against risk, using Qiskit's built-in finance application
# ---------------------------------------------------------------------
risk_factor = 0.5   # how much we penalize variance vs. reward return
budget = 3           # hold exactly 3 of the 5 assets

# Manual Markowitz QUBO:
#   minimize   risk_factor * x^T Sigma x  -  mu^T x
#   subject to sum(x) == budget,  x_i in {0, 1}
qp = QuadraticProgram(name="portfolio_optimization")
for a in assets:
    qp.binary_var(name=a)

linear = {assets[i]: -mu[i] for i in range(n)}
quadratic = {(assets[i], assets[j]): risk_factor * sigma[i, j]
             for i in range(n) for j in range(n)}
qp.minimize(linear=linear, quadratic=quadratic)
qp.linear_constraint(linear={a: 1 for a in assets}, sense="==", rhs=budget, name="budget")

print("QUBO formulated:", n, "binary variables (one per asset), budget =", budget)

# ---------------------------------------------------------------------
# 3. Ground truth — exact classical solver (brute-force eigensolver)
# ---------------------------------------------------------------------
exact_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
exact_result = exact_solver.solve(qp)
exact_portfolio = [assets[i] for i, bit in enumerate(exact_result.x) if bit == 1]

print("\n=== EXACT (classical, brute-force) ===")
print("Selected:", exact_portfolio)
print("Objective value:", round(exact_result.fval, 5))

# ---------------------------------------------------------------------
# 4. QAOA — the quantum approach, with COBYLA tuning the circuit
# ---------------------------------------------------------------------
qaoa = QAOA(sampler=StatevectorSampler(), optimizer=COBYLA(maxiter=200), reps=3)
qaoa_solver = MinimumEigenOptimizer(qaoa)
qaoa_result = qaoa_solver.solve(qp)
qaoa_portfolio = [assets[i] for i, bit in enumerate(qaoa_result.x) if bit == 1]

print("\n=== QAOA (quantum, COBYLA-tuned) ===")
print("Selected:", qaoa_portfolio)
print("Objective value:", round(qaoa_result.fval, 5))

# ---------------------------------------------------------------------
# 5. Compare
# ---------------------------------------------------------------------
match = set(exact_portfolio) == set(qaoa_portfolio)
print("\n=== COMPARISON ===")
print(f"QAOA matched the exact optimal portfolio: {match}")
if exact_result.fval != 0:
    ratio = qaoa_result.fval / exact_result.fval
    print(f"Approximation ratio (QAOA/exact objective): {ratio:.4f}")
