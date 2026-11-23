# Fits a Holt-Winters model to each real series and simulates new series from it.
# Run fetch_data.py first and save the Google Trends export as data/gym_trends.csv (see README.md).
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from series import DOMAINS, FIG_DIR, REAL_COLOR, SIM_COLOR, SIM_DIR

N_SIMULATIONS = 10
SEED = 42

SIM_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)
param_rows = []

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
    resid_std = np.std(fit.resid / fit.fittedvalues)
    print(f"{name}: {len(real)} months, {real.index.min():%Y-%m} to {real.index.max():%Y-%m}")
    print(f"  alpha={params['smoothing_level']:.3f}  beta={params['smoothing_trend']:.3f}  gamma={params['smoothing_seasonal']:.3f}")
    print(f"  initial level={params['initial_level']:.2f}  initial trend={params['initial_trend']:.3f}")
    print(f"  initial seasonal factors={np.round(params['initial_seasons'], 3).tolist()}")
    print(f"  residual std (relative)={resid_std:.4f}")

    param_rows.append({
        "domain": name,
        "months": len(real),
        "start": f"{real.index.min():%Y-%m}",
        "end": f"{real.index.max():%Y-%m}",
        "alpha": params["smoothing_level"],
        "beta": params["smoothing_trend"],
        "gamma": params["smoothing_seasonal"],
        "initial_level": params["initial_level"],
        "initial_trend": params["initial_trend"],
        **{f"season_{m + 1:02d}": s for m, s in enumerate(params["initial_seasons"])},
        "resid_std": resid_std,
    })

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

pd.DataFrame(param_rows).to_csv(SIM_DIR / "holt_winters_params.csv", index=False, float_format="%.6g")
