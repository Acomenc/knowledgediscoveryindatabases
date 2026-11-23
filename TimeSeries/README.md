# Time Series

Assignment: get familiar with time series and their internal structure by comparing real series from two domains with simulated ones.

## Task

Summary below; the verbatim assignment text is in [TASK.md](TASK.md).

- Download the random time series generator (Moodle).
- Generate typical time series for two domains.
- Submit a brief report (2–3 pages) that includes:
  - a description of each domain and the characteristics of its time series
  - the parameters used to simulate the time series in the app
  - images of both types of time series (real and simulated)
  - the average distance between series from different domains and between series from the same domain
  - conclusions (how well the series were simulated, pros and cons of the tool, etc.)

Only one group member submits the report, which must list the names of all group members.

## Domains

| Domain | Real series | Source |
|---|---|---|
| Gym interest | Google Trends, search term "gym", worldwide, monthly since Jan 2004, relative scale 0–100 | <https://trends.google.com/trends/explore?date=all&q=gym> |
| Air travel | International airline passengers, monthly totals in thousands, Jan 1949 – Dec 1960 (Box & Jenkins, Series G) | <https://github.com/selva86/datasets/blob/master/AirPassengers.csv> |

## Generators

- **Moodle app**: the course's random time series generator, with parameters set by hand for each domain.
- **statsmodels**: Holt-Winters / SARIMA models fitted to each real series and then simulated with random innovations (`.simulate()`). This is a model-based counterpart to the app's hand-set parameters.

Keras' `TimeseriesGenerator` was considered and dropped because it only slices an existing series into training windows and does not create new series.

## Setup

Requires [uv](https://docs.astral.sh/uv/). From this folder:

```sh
uv sync
uv run fetch_data.py
```

The data used for the report is committed in `data/`, so this step is only needed to refresh it. `fetch_data.py` downloads AirPassengers into `data/`. The Google Trends series has to be downloaded by hand: open the link above, download the "Interest over time" CSV and save it as `data/gym_trends.csv`.

The CSVs in `data/` are committed (an exception to the root `.gitignore`), because Trends rescales its values on every request and a fresh download can differ slightly from the one behind the figures. The committed `gym_trends.csv` was downloaded on 2026-11-23.

Then simulate:

```sh
uv run simulate.py
```

`simulate.py` fits a Holt-Winters model (additive trend, multiplicative seasonality, period 12) to each real series and simulates 10 series per domain with multiplicative errors (seed 42). Each simulation starts from the fitted initial state and has the same length as the real series. It prints the fitted parameters and writes:

- `data/simulated/{gym,air}_holt_winters.csv`: one column per simulated series
- `data/simulated/holt_winters_params.csv`: fitted parameters per domain
- `report/figures/{gym,air}_real_vs_simulated.pdf`: real series against the simulated ones

The current, incomplete month is dropped from the gym series.

Then draw the remaining report figures from the committed CSVs:

```sh
uv run plots.py
```

It writes to `report/figures/`:

- `stl_decomposition.pdf`: trend, seasonal and residual parts of each real series (STL, period 12)
- `seasonal_profile.pdf`: average month-of-year shape (value divided by a centred 12-month moving average), real against the simulated range
- `acf_real_vs_simulated.pdf`: autocorrelation of the detrended series up to lag 36, real against the simulated range
- `distance_matrix.pdf`: pairwise distances between all real and simulated series

It also writes the average distances within each domain and between the domains to `data/simulated/average_distances.csv`. Distances are Euclidean on z-normalised series; because the gym series is longer, each pair is compared over its most recent common number of months (144). DTW was tried and gave the same picture, so it was dropped.

## Report

`report/main.tex` uses the same setup as the Titanic report. To build it, run this from `report/`:

```sh
latexmk -pdf main.tex
```

## Open items

- Get the generator from Moodle and note here how to run it.
- Confirm the group members listed in the report.
