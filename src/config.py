from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project directories
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = ROOT_DIR / "results"

RAW_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Analysis universe
# ---------------------------------------------------------

UNIVERSE_FILE = DATA_DIR / "universe.csv"

universe = pd.read_csv(
    UNIVERSE_FILE,
    dtype=str
)

required_columns = {
    "ticker",
    "exchange",
    "currency",
}

missing_columns = (
    required_columns
    - set(universe.columns)
)

if missing_columns:
    raise ValueError(
        "universe.csv is missing columns: "
        + ", ".join(sorted(missing_columns))
    )


# Make sure all analysis securities are USD
if not (universe["currency"] == "USD").all():
    bad = universe.loc[
        universe["currency"] != "USD",
        "ticker"
    ].tolist()

    raise ValueError(
        "Non-USD securities found in universe: "
        + ", ".join(bad)
    )


TICKERS = universe["ticker"].tolist()


# ---------------------------------------------------------
# Tiingo
# ---------------------------------------------------------

TIINGO_BASE_URL = (
    "https://api.tiingo.com/tiingo/daily"
)

# Deliberately early. Tiingo returns observations
# only from the date on which the security has data.
START_DATE = "1980-01-01"


# ---------------------------------------------------------
# Clustering
# ---------------------------------------------------------

# We will automatically test every feasible k from
# MIN_CLUSTERS through the number of stocks - 1.
MIN_CLUSTERS = 2

# Keep this as an optional upper bound. None means
# "test every feasible number of clusters."
MAX_CLUSTERS = None


# ---------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------

MAX_ACF_LAG = 20

# Approximately one trading year
ROLLING_WINDOW = 252

ROLLING_ACF_LAG = 1