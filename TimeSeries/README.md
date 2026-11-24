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

- **Moodle app**: the course's random time series generator. Not used: it runs only on Windows, and no group member has a Windows machine.
- **statsmodels** (used): a Holt-Winters model fitted to each real series and then simulated with random errors (`.simulate()`), so the parameters come from the data instead of being set by hand.

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

- `stl_decomposition.pdf`: trend, seasonal and random parts of each real series (robust STL, period 12)
- `seasonal_profile.pdf`: average month-of-year shape (value divided by a centred 12-month moving average), real against the simulated range
- `acf_real_vs_simulated.pdf`: autocorrelation of the detrended series up to lag 36, real against the simulated range

Finally, compute the distances:

```sh
uv run distances.py
```

It writes the mean Euclidean and DTW distances within each domain, between real and simulated series, and between the domains to `data/simulated/average_distances.csv`, and the pairwise Euclidean distances to `report/figures/distance_matrix.pdf`. All series are cut to the last 144 months (the length of AirPassengers) and then z-normalised, so every pair, including pairs within the longer gym domain, is compared over the same length. DTW gives the same ranking as the Euclidean distance; both are reported because the lecture presents DTW as the answer to the alignment problem of the Euclidean distance.

Shared paths, loaders and colours live in `series.py`.

## Lecture slides

The report follows the terms and distance measures of the lecture slides "Time Series Data Mining" (Moodle). Save them as `lecture/DM on TS.pdf`; the folder is gitignored because this repository is public.

## Report

`report/main.tex` uses the same setup as the Titanic report. To build it, run this from `report/`:

```sh
latexmk -pdf main.tex
```

The built `report/main.pdf` is committed, as in the Titanic report, so the current version can be read without LaTeX. Rebuild it and commit it together with any change to the report or its figures.

## Open items

- Write the report text using the comment blocks in `report/main.tex`. Each block combines the suggested order, supporting details and narration ideas.
- Adapt the suggested AI statement to what we actually did, after checking the course's AI policy.
- Confirm the group members listed in the report.
