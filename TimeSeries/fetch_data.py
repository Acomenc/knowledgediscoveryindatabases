# Downloads the real series that can be fetched automatically.
# The Google Trends "gym" series has to be downloaded by hand (see README.md).
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
AIR_PASSENGERS_URL = "https://raw.githubusercontent.com/selva86/datasets/master/AirPassengers.csv"

DATA_DIR.mkdir(exist_ok=True)

# monthly international airline passengers (thousands), Jan 1949 - Dec 1960
air = pd.read_csv(AIR_PASSENGERS_URL, parse_dates=["date"])
air.to_csv(DATA_DIR / "AirPassengers.csv", index=False)
print(f"AirPassengers: {len(air)} months, {air['date'].min():%Y-%m} to {air['date'].max():%Y-%m}")
