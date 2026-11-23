# Draws the report figures from the committed real and simulated series.
# Run simulate.py first (see README.md).
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

from series import DOMAINS, FIG_DIR, REAL_COLOR, SIM_COLOR, SIM_DIR

MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
MAX_LAG = 36

plt.rcParams.update({
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.5,
    "legend.frameon": False,
})


def load_sims(name):
    return pd.read_csv(SIM_DIR / f"{name}_holt_winters.csv", index_col="date", parse_dates=["date"])


def ratio_to_trend(series):
    # divide by a centred 2x12 moving average, which removes trend but keeps the seasonal pattern
    trend = series.rolling(12, center=True).mean().rolling(2).mean().shift(-1)
    return (series / trend).dropna()


FIG_DIR.mkdir(parents=True, exist_ok=True)
data = {}
for name, (label, load) in DOMAINS.items():
    sims = load_sims(name)
    # restrict to the months simulate.py used
    data[name] = (load().reindex(sims.index), sims)

# STL decomposition of the real series
fig, axes = plt.subplots(4, 2, figsize=(7, 5), sharex="col")
for col, (name, (label, _)) in enumerate(DOMAINS.items()):
    real = data[name][0]
    stl = STL(real, period=12, robust=True).fit()
    parts = [("Observed", real), ("Trend", stl.trend), ("Seasonal", stl.seasonal), ("Random", stl.resid)]
    for row, (title, part) in enumerate(parts):
        ax = axes[row, col]
        if title == "Random":
            ax.axhline(0, color="#999999", linewidth=0.5)
            ax.plot(part.index, part, color=REAL_COLOR, linewidth=0.6, marker="o", markersize=1.5, linestyle="none")
        else:
            ax.plot(part.index, part, color=REAL_COLOR, linewidth=0.9)
        if col == 0:
            ax.set_ylabel(title)
    axes[0, col].set_title(label)
fig.align_ylabels(axes[:, 0])
fig.tight_layout()
fig.savefig(FIG_DIR / "stl_decomposition.pdf")
plt.close(fig)

# average month-of-year shape, real vs. simulated
fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
for ax, (name, (label, _)) in zip(axes, DOMAINS.items()):
    real, sims = data[name]
    profile = lambda s: ratio_to_trend(s).groupby(lambda d: d.month).mean()
    sim_profiles = pd.concat([profile(sims[c]) for c in sims.columns], axis=1)
    x = np.arange(1, 13)
    ax.fill_between(x, sim_profiles.min(axis=1), sim_profiles.max(axis=1), color=SIM_COLOR, alpha=0.2,
                    linewidth=0, label=f"Simulated, range (n={sims.shape[1]})")
    ax.plot(x, sim_profiles.mean(axis=1), color=SIM_COLOR, linewidth=1.2, label="Simulated, mean")
    ax.plot(x, profile(real), color=REAL_COLOR, linewidth=1.5, marker="o", markersize=3, label="Real")
    ax.axhline(1, color="#999999", linewidth=0.5)
    ax.set_xticks(x, MONTHS)
    ax.set_title(label)
axes[0].set_ylabel("Value / 12-month trend")
axes[0].legend(loc="upper center")
fig.tight_layout()
fig.savefig(FIG_DIR / "seasonal_profile.pdf")
plt.close(fig)

# autocorrelation of the detrended series, real vs. simulated
fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
for ax, (name, (label, _)) in zip(axes, DOMAINS.items()):
    real, sims = data[name]
    lags = np.arange(MAX_LAG + 1)
    sim_acfs = np.array([acf(ratio_to_trend(sims[c]), nlags=MAX_LAG) for c in sims.columns])
    ax.fill_between(lags, sim_acfs.min(axis=0), sim_acfs.max(axis=0), color=SIM_COLOR, alpha=0.2,
                    linewidth=0, label=f"Simulated, range (n={sims.shape[1]})")
    ax.plot(lags, sim_acfs.mean(axis=0), color=SIM_COLOR, linewidth=1.2, label="Simulated, mean")
    ax.plot(lags, acf(ratio_to_trend(real), nlags=MAX_LAG), color=REAL_COLOR, linewidth=1.5, label="Real")
    # approximate 95% band for white noise
    bound = 1.96 / np.sqrt(len(ratio_to_trend(real)))
    ax.axhspan(-bound, bound, color="#999999", alpha=0.15, linewidth=0)
    ax.axhline(0, color="#999999", linewidth=0.5)
    ax.set_xticks(range(0, MAX_LAG + 1, 12))
    ax.set_xlabel("Lag (months)")
    ax.set_title(label)
axes[0].set_ylabel("Autocorrelation (detrended)")
fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=3)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIG_DIR / "acf_real_vs_simulated.pdf")
plt.close(fig)
