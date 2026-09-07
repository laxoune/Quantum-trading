"""
Visualizations for the Quantum-Trading hybrid pipeline.
Reads outputs already produced by multi_asset_forecast.py and
real_hybrid_pipeline.py (must be run first).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ASSETS = ["tech", "utility", "energy", "healthcare", "bond_like"]
LABELS = {"tech": "Tech", "utility": "Utility", "energy": "Energy",
          "healthcare": "Healthcare", "bond_like": "Bond-like"}
COLORS = {"tech": "#4C72B0", "utility": "#DD8452", "energy": "#55A868",
          "healthcare": "#C44E52", "bond_like": "#8172B2"}

# ---------------------------------------------------------------------
# 1. Per-asset forecast comparison charts (Real vs LSTM vs Quantum-corrected)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(13, 12))
axes = axes.flatten()

for i, asset in enumerate(ASSETS):
    df = pd.read_csv(f"{asset}_forecast_comparison.csv")
    ax = axes[i]
    ax.plot(df["Real Price"], label="Real Price", color="black", linewidth=1.8)
    ax.plot(df["LSTM Prediction"], label="LSTM Prediction", color="#999999",
             linestyle="--", linewidth=1.3)
    ax.plot(df["Quantum-Corrected Prediction"], label="Quantum-Corrected",
             color=COLORS[asset], linewidth=1.6)
    ax.set_title(LABELS[asset], fontweight="bold")
    ax.set_xlabel("Held-out test day")
    ax.set_ylabel("Price")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

axes[-1].axis("off")
fig.suptitle("LSTM vs Quantum-Corrected Forecast vs Real Price (held-out test days)",
             fontsize=14, fontweight="bold", y=1.00)
fig.tight_layout()
fig.savefig("forecast_comparison_all_assets.png", dpi=150, bbox_inches="tight")
print("Saved forecast_comparison_all_assets.png")

# ---------------------------------------------------------------------
# 2. MSE improvement bar chart (LSTM vs LSTM+VQC, per asset)
# ---------------------------------------------------------------------
mse_data = []
for asset in ASSETS:
    df = pd.read_csv(f"{asset}_forecast_comparison.csv")
    lstm_mse = ((df["Real Price"] - df["LSTM Prediction"]) ** 2).mean()
    vqc_mse = ((df["Real Price"] - df["Quantum-Corrected Prediction"]) ** 2).mean()
    mse_data.append((LABELS[asset], lstm_mse, vqc_mse))

labels = [d[0] for d in mse_data]
lstm_mses = [d[1] for d in mse_data]
vqc_mses = [d[2] for d in mse_data]

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width / 2, lstm_mses, width, label="LSTM alone", color="#999999")
bars2 = ax.bar(x + width / 2, vqc_mses, width, label="LSTM + Quantum-VQC", color="#4C72B0")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("MSE (lower is better)")
ax.set_title("Forecast Error: LSTM Alone vs Quantum-Corrected, by Asset", fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.25)
for i, (l, v) in enumerate(zip(lstm_mses, vqc_mses)):
    change = (l - v) / l * 100
    color = "green" if change > 0 else "red"
    ax.annotate(f"{change:+.0f}%", (x[i], max(l, v) * 1.03),
                ha="center", fontsize=9, color=color, fontweight="bold")
fig.tight_layout()
fig.savefig("mse_comparison_by_asset.png", dpi=150, bbox_inches="tight")
print("Saved mse_comparison_by_asset.png")

# ---------------------------------------------------------------------
# 3. Expected return bar chart + portfolio selection highlight
# ---------------------------------------------------------------------
mu = np.load("real_mu.npy")
asset_labels = list(LABELS.values())
selected = ["Tech", "Energy", "Healthcare"]  # from real_hybrid_pipeline.py output

fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = ["#4C72B0" if a in selected else "#CCCCCC" for a in asset_labels]
bars = ax.bar(asset_labels, mu * 100, color=bar_colors, edgecolor="black", linewidth=0.6)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Implied annualized return (%)")
ax.set_title("Real, Data-Derived Forecasts -> QAOA Portfolio Selection", fontweight="bold")
for bar, val in zip(bars, mu * 100):
    ax.annotate(f"{val:+.1f}%", (bar.get_x() + bar.get_width() / 2, val),
                ha="center", va="bottom" if val > 0 else "top", fontsize=9)
# legend proxy
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#4C72B0", label="Selected by QAOA"),
                    Patch(color="#CCCCCC", label="Excluded by QAOA")])
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("portfolio_selection.png", dpi=150, bbox_inches="tight")
print("Saved portfolio_selection.png")

# ---------------------------------------------------------------------
# 4. Correlation heatmap (from the real covariance matrix)
# ---------------------------------------------------------------------
cov = np.load("real_cov.npy")
std = np.sqrt(np.diag(cov))
corr = cov / np.outer(std, std)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(asset_labels)))
ax.set_yticks(range(len(asset_labels)))
ax.set_xticklabels(asset_labels, rotation=45, ha="right")
ax.set_yticklabels(asset_labels)
for i in range(len(asset_labels)):
    for j in range(len(asset_labels)):
        ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                 color="white" if abs(corr[i, j]) > 0.5 else "black", fontsize=9)
ax.set_title("Real Historical Return Correlation Matrix", fontweight="bold")
fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
fig.tight_layout()
fig.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved correlation_heatmap.png")

print("\nAll 4 visualizations saved.")
