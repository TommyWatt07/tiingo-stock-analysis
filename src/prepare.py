import numpy as np
import pandas as pd

from .config import RESULTS_DIR


def build_price_matrix(histories):
    """
    Combine individual ticker histories into one
    adjusted-close price matrix.

    Rows = dates
    Columns = tickers
    """

    series = []

    for ticker, df in histories.items():

        if "adjClose" in df.columns:
            price_column = "adjClose"
        elif "close" in df.columns:
            price_column = "close"
        else:
            print(
                f"Skipping {ticker}: "
                "no close column found"
            )
            continue

        s = (
            df.set_index("date")[price_column]
            .rename(ticker)
        )

        series.append(s)

    prices = pd.concat(series, axis=1)

    prices = prices.sort_index()

    prices.to_csv(
        RESULTS_DIR / "all_adjusted_closes.csv"
    )

    return prices


def build_return_matrix(prices):
    """
    Calculate daily log returns.

    r_t = log(P_t / P_(t-1))
    """

    returns = np.log(
        prices / prices.shift(1)
    )

    # Replace impossible values if present
    returns = returns.replace(
        [np.inf, -np.inf],
        np.nan
    )

    returns.to_csv(
        RESULTS_DIR / "all_daily_returns.csv"
    )

    return returns


def get_common_returns(returns):
    """
    Return only dates for which every stock has data.

    This gives an apples-to-apples matrix for
    PCA and clustering.
    """

    common = returns.dropna()

    common.to_csv(
        RESULTS_DIR / "common_period_returns.csv"
    )

    return common


def create_history_summary(prices):
    """
    Show how much historical data is available
    for each stock.
    """

    rows = []

    for ticker in prices.columns:

        data = prices[ticker].dropna()

        if data.empty:
            continue

        rows.append(
            {
                "ticker": ticker,
                "first_date": data.index.min(),
                "last_date": data.index.max(),
                "observations": len(data),
            }
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        "first_date"
    )

    summary.to_csv(
        RESULTS_DIR / "data_history_summary.csv",
        index=False
    )

    return summary