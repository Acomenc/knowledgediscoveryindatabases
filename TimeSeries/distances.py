# Computes Euclidean and DTW distances between all real and simulated series, averages them
# within and between the domains, and draws the pairwise Euclidean distance matrix.
# Run simulate.py first.
#
# All series are cut to the last 144 months, the length of AirPassengers, and then z-normalised,
# so that every pair has the same length and the averages are comparable across groups.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from series import DOMAINS, FIG_DIR, SIM_DIR

WINDOW = 144


def znorm(values):
    return (values - values.mean()) / values.std()


def euclidean(s, t):
    return np.linalg.norm(s - t)


def dtw(s, t):
    # dynamic time warping with absolute difference as the local cost
    n, m = len(s), len(t)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(s[i - 1] - t[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return cost[n, m]


# real series first within each domain
series, labels, domains = [], [], []
for name, (_, load) in DOMAINS.items():
    sims = pd.read_csv(SIM_DIR / f"{name}_holt_winters.csv", index_col="date", parse_dates=["date"])
    real = load().reindex(sims.index)
    series += [real.to_numpy()] + [sims[c].to_numpy() for c in sims.columns]
    labels += [f"{name} real"] + [f"{name} sim {i + 1}" for i in range(sims.shape[1])]
    domains += [name] * (sims.shape[1] + 1)
series = [znorm(s[-WINDOW:]) for s in series]
domains = np.array(domains)
is_real = np.array([label.endswith("real") for label in labels])
n = len(series)

matrices = {"euclidean": np.zeros((n, n)), "dtw": np.zeros((n, n))}
for i in range(n):
    for j in range(i + 1, n):
        for measure, distance in (("euclidean", euclidean), ("dtw", dtw)):
            matrices[measure][i, j] = matrices[measure][j, i] = distance(series[i], series[j])

upper = np.triu(np.ones((n, n), dtype=bool), k=1)
name_a, name_b = DOMAINS
groups = {}
for name in DOMAINS:
    in_domain = (domains[:, None] == name) & (domains[None, :] == name)
    groups[f"within {name}"] = in_domain & upper
    groups[f"{name} real vs simulated"] = in_domain & is_real[:, None] & ~is_real[None, :]
groups["between domains"] = (domains[:, None] == name_a) & (domains[None, :] == name_b)
groups["between domains, real only"] = groups["between domains"] & is_real[:, None] & is_real[None, :]

rows = []
for group, mask in groups.items():
    row = {"pair": group, "pairs": mask.sum()}
    for measure, matrix in matrices.items():
        row[f"{measure}_mean"] = matrix[mask].mean()
        row[f"{measure}_std"] = matrix[mask].std()
    rows.append(row)
result = pd.DataFrame(rows)
print(result.to_string(index=False, float_format="%.2f"))
result.to_csv(SIM_DIR / "average_distances.csv", index=False, float_format="%.4g")

fig, ax = plt.subplots(figsize=(4.6, 4))
image = ax.imshow(matrices["euclidean"], cmap="Blues")
boundary = (domains == name_a).sum() - 0.5
ax.axhline(boundary, color="white", linewidth=1.5)
ax.axvline(boundary, color="white", linewidth=1.5)
ax.set_xticks(range(n), labels, rotation=90, fontsize=5)
ax.set_yticks(range(n), labels, fontsize=5)
ax.spines[:].set_visible(False)
fig.colorbar(image, ax=ax, label="z-normalised Euclidean distance", shrink=0.8)
fig.tight_layout()
fig.savefig(FIG_DIR / "distance_matrix.pdf")
plt.close(fig)
