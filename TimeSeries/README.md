# Time Series

Assignment: get familiar with time series and their internal structure by comparing real series from two domains with simulated ones.

## Task

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

`fetch_data.py` downloads AirPassengers into `data/`. The Google Trends series has to be downloaded by hand: open the link above, download the "Interest over time" CSV and save it as `data/gym_trends.csv`.

CSVs are not committed (see the root `.gitignore`). Trends rescales its values on every request, so numbers can differ slightly between downloads.

## Report

`report/main.tex` uses the same setup as the Titanic report. To build it, run this from `report/`:

```sh
latexmk -pdf main.tex
```

## Open items

- Get the generator from Moodle and note here how to run it.
- Choose the distance measure (z-normalised Euclidean and/or DTW) and decide how to compare series of different lengths (about 270 months of gym data vs. 144 for AirPassengers).
- Confirm the group members listed in the report.
