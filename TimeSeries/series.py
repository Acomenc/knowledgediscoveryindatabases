# Shared paths, loaders and plot colours for the scripts in this folder.
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SIM_DIR = DATA_DIR / "simulated"
FIG_DIR = BASE_DIR / "report" / "figures"

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
