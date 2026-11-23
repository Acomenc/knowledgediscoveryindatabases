# Fits a Holt-Winters model to each real series and simulates new series from it.
# Run fetch_data.py first and save the Google Trends export as data/gym_trends.csv (see README.md).
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SIM_DIR = DATA_DIR / "simulated"
FIG_DIR = BASE_DIR / "report" / "figures"

N_SIMULATIONS = 10
SEED = 42

REAL_COLOR = "#1a1a19"
SIM_COLOR = "#2a78d6"


def load_gym():
    # the Trends export has a "Category: ..." line and a blank line before the header
    gym = pd.read_csv(DATA_DIR / "gym_trends.csv", skiprows=2)
    gym.columns = ["date", "value"]
    gym["date"] = pd.to_datetime(gym["date"])
    # the current month is still incomplete
    gym = gym[gym["date"].dt.to_period("M") < pd.Timestamp.today().to_period("M")]
    return gym.set_index("date")["value"].asfreq("MS").astype(float)


def load_air():
    air = pd.read_csv(DATA_DIR / "AirPassengers.csv", parse_dates=["date"])
    return air.set_index("date")["value"].asfreq("MS").astype(float)


DOMAINS = {
    "gym": ("Gym interest (Google Trends, 0-100)", load_gym),
    "air": ("Air passengers (thousands)", load_air),
}

SIM_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

for name, (label, load) in DOMAINS.items():
    real = load()

    model = ExponentialSmoothing(
        real, trend="add", seasonal="mul", seasonal_periods=12, initialization_method="estimated"
    )
    fit = model.fit()

    # anchor="start" runs each simulation from the fitted initial state,
    # so simulated and real series cover the same months
    sims = fit.simulate(len(real), anchor="start", repetitions=N_SIMULATIONS, error="mul", rng=rng)
    sims.columns = [f"sim_{i + 1:02d}" for i in range(N_SIMULATIONS)]
    sims.index.name = "date"
    sims.to_csv(SIM_DIR / f"{name}_holt_winters.csv")

    params = fit.params
    print(f"{name}: {len(real)} months, {real.index.min():%Y-%m} to {real.index.max():%Y-%m}")
    print(f"  alpha={params['smoothing_level']:.3f}  beta={params['smoothing_trend']:.3f}  gamma={params['smoothing_seasonal']:.3f}")
    print(f"  initial level={params['initial_level']:.2f}  initial trend={params['initial_trend']:.3f}")
    print(f"  initial seasonal factors={np.round(params['initial_seasons'], 3).tolist()}")
    print(f"  residual std (relative)={np.std(fit.resid / fit.fittedvalues):.4f}")

    fig, ax = plt.subplots(figsize=(7, 3))
    for i, col in enumerate(sims.columns):
        ax.plot(sims.index, sims[col], color=SIM_COLOR, linewidth=0.8, alpha=0.35,
                label=f"Simulated (n={N_SIMULATIONS})" if i == 0 else None)
    ax.plot(real.index, real, color=REAL_COLOR, linewidth=1.5, label="Real")
    ax.set_ylabel(label)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_real_vs_simulated.pdf")
    plt.close(fig)
