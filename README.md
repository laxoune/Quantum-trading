# Quantum-Trading

Hybrid classical–quantum framework for equity price forecasting. An LSTM
produces a baseline price forecast; a trainable **Qiskit Variational
Quantum Circuit (VQC)** learns to correct the LSTM's residual error using
market volatility (VIX) and momentum (% change) as context features.

**Result:** on 39 held-out trading days, the quantum-corrected model
reduced MSE by ~82% versus the LSTM baseline alone (346.01 → 63.57).

| Model | MSE | MAE |
|---|---|---|
| LSTM alone | 346.01 | 18.05 |
| LSTM + Quantum-VQC correction | 63.57 | 6.96 |

## Circuit

- 3 qubits (LSTM prediction, VIX level, % change)
- `ZFeatureMap` for data encoding + `RealAmplitudes` ansatz (9 trainable weights)
- Optimized via COBYLA, trained on 152 days, tested on 39 unseen days

## Files

- `qiskit_vqc_correction.py` — full training + evaluation script
- `qiskit_vqc_test_results.csv` — held-out test predictions
- `quantum_trading_overview.Rmd` — full technical write-up and methodology
- `data/` — LSTM backtest output the model trains on *(see note below)*

## ⚠️ Data dependency

This script trains on real out-of-sample LSTM backtest output, not
synthetic data. It expects:

```
data/backtesting_results.csv   # LSTM Predicted Price vs Real Price
data/testing_data.csv          # VIX Price, Pct Change (same dates)
```

Re-add these two files under `data/` (and update the two `pd.read_csv(...)`
paths in `qiskit_vqc_correction.py` to match) before running — the script
will not execute without them.

## Honest caveat

39 test days is enough to show the mechanism generalizes beyond its
training window, but too small a sample to claim a robust, production-grade
edge. Framed accurately: *"a trainable quantum correction layer that
measurably reduced out-of-sample error on this dataset,"* not "quantum
beats classical forecasting" as a general claim.
