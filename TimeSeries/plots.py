# Draws the report figures from the committed real and simulated series.
# Run simulate.py first (see README.md).
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SIM_DIR = DATA_DIR / "simulated"
FIG_DIR = BASE_DIR / "report" / "figures"

REAL_COLOR = "#1a1a19"
SIM_COLOR = "#2a78d6"
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


def load_real(name, sims):
    if name == "gym":
        # the Trends export has a "Category: ..." line and a blank line before the header
        real = pd.read_csv(DATA_DIR / "gym_trends.csv", skiprows=2)
    else:
        real = pd.read_csv(DATA_DIR / "AirPassengers.csv")
    real.columns = ["date", "value"]
    real["date"] = pd.to_datetime(real["date"])
    # restrict to the months simulate.py used, which drops the incomplete current month
    return real.set_index("date")["value"].astype(float).reindex(sims.index)


def load_sims(name):
    return pd.read_csv(SIM_DIR / f"{name}_holt_winters.csv", index_col="date", parse_dates=["date"])


def ratio_to_trend(series):
    # divide by a centred 2x12 moving average, which removes trend but keeps the seasonal pattern
    trend = series.rolling(12, center=True).mean().rolling(2).mean().shift(-1)
    return (series / trend).dropna()


def znorm(values):
    return (values - values.mean()) / values.std()


def euclidean(a, b):
    # the domains differ in length, so compare their most recent common number of months
    n = min(len(a), len(b))
    return np.linalg.norm(znorm(a[-n:]) - znorm(b[-n:]))


DOMAINS = {
    "gym": "Gym interest (Google Trends, 0-100)",
    "air": "Air passengers (thousands)",
}

FIG_DIR.mkdir(parents=True, exist_ok=True)
data = {}
for name in DOMAINS:
    sims = load_sims(name)
    data[name] = (load_real(name, sims), sims)

# STL decomposition of the real series
fig, axes = plt.subplots(4, 2, figsize=(7, 5), sharex="col")
for col, (name, label) in enumerate(DOMAINS.items()):
    real = data[name][0]
    stl = STL(real, period=12, robust=True).fit()
    parts = [("Observed", real), ("Trend", stl.trend), ("Seasonal", stl.seasonal), ("Residual", stl.resid)]
    for row, (title, part) in enumerate(parts):
        ax = axes[row, col]
        if title == "Residual":
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
for ax, (name, label) in zip(axes, DOMAINS.items()):
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
for ax, (name, label) in zip(axes, DOMAINS.items()):
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

# pairwise distances, real series first within each domain
series, labels, domains = [], [], []
for name in DOMAINS:
    real, sims = data[name]
    series += [real.to_numpy()] + [sims[c].to_numpy() for c in sims.columns]
    labels += [f"{name} real"] + [f"{name} sim {i + 1}" for i in range(sims.shape[1])]
    domains += [name] * (sims.shape[1] + 1)
domains = np.array(domains)
n = len(series)
same = domains[:, None] == domains[None, :]
off_diagonal = ~np.eye(n, dtype=bool)

matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        matrix[i, j] = matrix[j, i] = euclidean(series[i], series[j])

average_rows = []
for name in DOMAINS:
    in_domain = (domains[:, None] == name) & (domains[None, :] == name) & off_diagonal
    average_rows.append({"pair": f"within {name}", "mean": matrix[in_domain].mean()})
average_rows.append({"pair": "between domains", "mean": matrix[~same].mean()})
print("average z-normalised Euclidean distance: " + ", ".join(f"{r['pair']}={r['mean']:.2f}" for r in average_rows))
pd.DataFrame(average_rows).to_csv(SIM_DIR / "average_distances.csv", index=False, float_format="%.4g")

fig, ax = plt.subplots(figsize=(4.6, 4))
image = ax.imshow(matrix, cmap="Blues")
boundary = (domains == "gym").sum() - 0.5
ax.axhline(boundary, color="white", linewidth=1.5)
ax.axvline(boundary, color="white", linewidth=1.5)
ax.set_xticks(range(n), labels, rotation=90, fontsize=5)
ax.set_yticks(range(n), labels, fontsize=5)
ax.grid(False)
ax.spines[:].set_visible(False)
fig.colorbar(image, ax=ax, label="z-normalised Euclidean distance", shrink=0.8)
fig.tight_layout()
fig.savefig(FIG_DIR / "distance_matrix.pdf")
plt.close(fig)
