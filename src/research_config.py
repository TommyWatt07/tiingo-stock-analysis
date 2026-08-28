"""
Research configuration for the Tiingo stock time-series project.

This file contains the stock groups and pairwise comparisons used by
the current research experiments. Keeping them here means main.py
contains the analysis pipeline rather than hard-coded research choices.
"""


# ==================================================
# Supermajors
# ==================================================

SUPERMAJOR_TICKERS = [
    "XOM",
    "CVX",
    "SHEL",
    "BP",
    "TTE",
]


# ==================================================
# Geographic groups
# ==================================================

CANADA_TICKERS = [
    "ENB",
    "CNQ",
    "CCJ",
    "CVE",
    "IMO",
    "SU",
    "TRP",
    "PBA",
]


REST_OF_WORLD_TICKERS = [
    # Canada
    "ENB",
    "CNQ",
    "CCJ",
    "CVE",
    "IMO",
    "SU",
    "TRP",
    "PBA",

    # Other non-US companies
    "WDS",
    "PBR",
    "SHEL",
    "BP",
    "TS",
    "TTE",
    "E",
    "EQNR",
]


CANADA_EXCLUDING_CCJ = [
    "ENB",
    "CNQ",
    "CVE",
    "IMO",
    "SU",
    "TRP",
    "PBA",
]


REST_OF_WORLD_EXCLUDING_CCJ = [
    "ENB",
    "CNQ",
    "CVE",
    "IMO",
    "SU",
    "TRP",
    "PBA",
    "WDS",
    "PBR",
    "SHEL",
    "BP",
    "TS",
    "TTE",
    "E",
    "EQNR",
]


# ==================================================
# Focused diagnostic groups
# ==================================================

INTEGRATED_GROUP_TICKERS = [
    "BP",
    "CVX",
    "E",
    "EQNR",
    "SHEL",
    "TS",
    "WDS",
    "XOM",
    "TTE",
]


OILFIELD_SERVICES_TS_TICKERS = [
    "BKR",
    "FTI",
    "HAL",
    "SLB",
    "TS",
]


# ==================================================
# Special outlier exclusions
# ==================================================

OUTLIERS_TO_EXCLUDE = {
    "CCJ",
    "TPL",
}


# ==================================================
# Rolling-correlation pairs
# ==================================================

ROLLING_CORRELATION_PAIRS = [
    ("XOM", "CVX"),
    ("COP", "XOM"),
    ("COP", "CVX"),
    ("COP", "EOG"),
    ("EQT", "EXE"),
]


TTE_TS_DIAGNOSTIC_PAIRS = [
    # TTE against the integrated group
    ("TTE", "BP"),
    ("TTE", "CVX"),
    ("TTE", "E"),
    ("TTE", "EQNR"),
    ("TTE", "SHEL"),
    ("TTE", "TS"),
    ("TTE", "WDS"),
    ("TTE", "XOM"),

    # TS against the integrated group
    ("TS", "BP"),
    ("TS", "CVX"),
    ("TS", "E"),
    ("TS", "EQNR"),
    ("TS", "SHEL"),
    ("TS", "WDS"),
    ("TS", "XOM"),
]


# ==================================================
# Research parameters
# ==================================================

ROLLING_CORRELATION_WINDOW = 252

RECENT_CLUSTER_WINDOW = 252

REPLICATION_MIN_STOCKS = 2
REPLICATION_MAX_STOCKS = 4