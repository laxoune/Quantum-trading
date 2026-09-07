---
title: "Quantum-Trading Project — Technical Overview & Restructuring Guide"
author: "Alexander Dagher"
date: "`r Sys.Date()`"
output:
  html_document:
    toc: true
    toc_depth: 3
    theme: cosmo
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(eval = FALSE, echo = TRUE)
```

## 1. Purpose of This Document

This explains what the `Quantum-trading` repo actually contains, what was
found on review, what was built to fix it, and how to reorganize the repo
so it reads clearly to an outside technical reviewer (e.g. Pasqal).

---

## 2. What the Repo Contained Before This Pass

| Component | File(s) | What it does |
|---|---|---|
| Classical LSTM (baseline) | `calssical_version_1.1/*/LSTM_INDICATORS.ipynb`, `LSTM_indiactors_hyper.ipynb`, `LSTM_IMPROVED.ipynb` | Price forecasting using LSTM + technical indicators, three iterations |
| Monte Carlo + "quantum" blend | `monte_Carlo_version/LSTM_MONTE_CARLO.ipynb` | LSTM prediction blended with a Monte Carlo path simulation and a quantum-encoded feature |
| Exploratory notebooks | `trial_1.ipynb` – `trial_6.ipynb` | Iteration/scratch work, no quantum content |
| Write-ups | `QUANTUM TRADING (2).pdf`, `QUANTUM TRADING PAPER (2).pdf` | Narrative descriptions of the project |

### Key finding: the "quantum" component was not Qiskit

The only quantum code in the repo lives in `LSTM_MONTE_CARLO.ipynb` and is
built on **PennyLane** (`import pennylane as qml`), not Qiskit, despite
Qiskit being listed as a skill on the resume this repo backs up.

### Key finding: the original quantum step did not learn anything

The original circuit:

```python
dev = qml.device("default.qubit", wires=2, shots=1000)

@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))
```

took the LSTM's price prediction, converted it directly into a rotation
angle, and sampled the circuit. There were **no trainable parameters** —
the mapping from input to output was fixed, then blended into the final
prediction with fixed weights (`0.6 * LSTM + 0.2 * Monte Carlo + 0.2 *
quantum`). This is a legitimate quantum *encoding* technique, but it is not
a model that learns, and it is not Qiskit.

---

## 3. What Was Built to Fix It

A genuine trainable **Variational Quantum Circuit (VQC)** in Qiskit that
learns to correct the LSTM's residual error, rather than just encoding a
number and sampling it.

### 3.1 Data used

Reused the project's own existing backtest output — no synthetic data:

- `Backtesting_Results.csv` — the LSTM's out-of-sample `Predicted Price` vs
  `Real Price` (191 trading days)
- `Testing_Data.csv` — same-date `Vix Price` and `Pct Change`, used as
  market-regime context features

Target variable: `Residual = Real Price − Predicted Price`, i.e. the
model's job is to learn *when and how the LSTM is wrong*, using
volatility and momentum as signal.

### 3.2 Circuit architecture

```python
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit_algorithms.optimizers import COBYLA

feature_map = ZFeatureMap(feature_dimension=3, reps=1)   # encodes inputs
ansatz = RealAmplitudes(num_qubits=3, reps=2)             # TRAINABLE weights

qnn = EstimatorQNN(
    circuit=feature_map.compose(ansatz),
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
)

regressor = NeuralNetworkRegressor(neural_network=qnn, optimizer=COBYLA(maxiter=150))
regressor.fit(X_train, y_train)
```

- **3 qubits**, one per feature: LSTM prediction, VIX level, % change
- **9 trainable weights** in the `RealAmplitudes` ansatz, optimized via
  COBYLA (gradient-free optimizer, standard for near-term quantum circuits)
- Trained on the first 152 days, evaluated on the last 39 days the model
  never saw (time-respecting split — no shuffling, since this is a
  time series)

### 3.3 Results (held-out test days)

| Model | MSE | MAE |
|---|---|---|
| LSTM alone | 346.01 | 18.05 |
| LSTM + Quantum-VQC correction | 63.57 | 6.96 |
| **Change** | **−81.6%** | **−61.4%** |

**Honest caveat:** the test set is only 39 trading days. That is enough to
demonstrate the mechanism works and generalizes beyond the training
window, but too small to claim a robust, production-grade edge. Frame it
as "a trainable quantum correction layer that measurably reduced
out-of-sample error on this dataset," not "quantum beats classical
forecasting."

Full script: `qiskit_vqc_correction.py`. Full test predictions:
`qiskit_vqc_test_results.csv`.

---

## 4. Suggested Repo Restructure

Current structure mixes naming typos (`calssical_version_1.1`), unlabeled
trial notebooks, and no top-level README, which makes the repo hard to
read for a reviewer skimming it in two minutes. Suggested layout:

```
Quantum-trading/
├── README.md                      <- 1-page summary + results table (see §5)
├── data/
│   ├── training_data.csv
│   ├── testing_data.csv
│   └── backtesting_results.csv
├── classical/
│   ├── 01_lstm_baseline.ipynb
│   ├── 02_lstm_hyperparameter_tuned.ipynb
│   └── 03_lstm_cnn_improved.ipynb
├── quantum/
│   ├── monte_carlo_pennylane_v1.ipynb   <- original, kept for history
│   └── qiskit_vqc_correction.py         <- current, trainable, Qiskit
├── results/
│   └── qiskit_vqc_test_results.csv
└── docs/
    ├── quantum_trading_overview.Rmd     <- this file
    ├── QUANTUM_TRADING.pdf
    └── QUANTUM_TRADING_PAPER.pdf
```

Renaming notes:
- Fix `calssical_version_1.1` → `classical/` (typo + flatten the nested
  `classical_version_1.1.1/.2/.3` folders into clearly numbered notebooks)
- Drop `trial_1.ipynb` … `trial_6.ipynb` from the main branch, or move them
  to a `scratch/` folder excluded from the top-level view — they add noise
  with no labeled purpose for an outside reviewer
- Rename `monte_Carlo_version` → clarify in the notebook title or a
  header cell that this is the **original PennyLane** version, since the
  current Qiskit version supersedes it for anything resume-facing

---

## 5. Suggested README.md Summary (drop-in)

```markdown
# Quantum-Trading

Hybrid classical–quantum framework for equity price forecasting.
An LSTM produces a baseline forecast; a trainable Qiskit variational
quantum circuit (VQC) learns to correct the LSTM's residual error using
market volatility (VIX) and momentum (% change) as context features.

**Result:** on 39 held-out trading days, the quantum-corrected model
reduced MSE by ~82% versus the LSTM baseline alone.

- `classical/` — LSTM baseline + tuning iterations
- `quantum/qiskit_vqc_correction.py` — trainable VQC (Qiskit, current)
- `quantum/monte_carlo_pennylane_v1.ipynb` — earlier PennyLane prototype
- `results/` — held-out test predictions and metrics
```

---

## 6. Next Steps

1. Reorganize folders per §4 (manual — Claude does not have push access to
   this repo)
2. Add the README from §5
3. Update resume bullet to reflect the Qiskit-based trainable model and
   the measured result, not the original fixed-encoding description
4. Optionally re-run with a larger backtest window once more historical
   data is available, to firm up the 39-day result before citing it in an
   interview
