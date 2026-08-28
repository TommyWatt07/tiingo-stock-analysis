import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import RESULTS_DIR


def filter_period(
    returns,
    start_date=None,
    end_date=None,
):
    """
    Restrict a return DataFrame to an optional date range.
    """

    data = returns.copy()

    if start_date is not None:
        data = data.loc[
            pd.Timestamp(start_date):
        ]

    if end_date is not None:
        data = data.loc[
            :pd.Timestamp(end_date)
        ]

    return data


def analyze_pair(
    returns,
    ticker_1,
    ticker_2,
    window=252,
    start_date=None,
    end_date=None,
    save=True,
    plot=True,
):
    """
    Analyze the relationship between two stocks.

    Parameters
    ----------
    returns:
        DataFrame containing daily returns.

    ticker_1, ticker_2:
        Stock tickers.

    window:
        Rolling correlation window in observations.
        252 = approximately one trading year.

    start_date, end_date:
        Optional date restrictions.

    save:
        Save results to CSV.

    plot:
        Save a rolling-correlation chart.
    """

    if ticker_1 not in returns.columns:
        raise ValueError(
            f"{ticker_1} not found in return data."
        )

    if ticker_2 not in returns.columns:
        raise ValueError(
            f"{ticker_2} not found in return data."
        )

    data = filter_period(
        returns[
            [ticker_1, ticker_2]
        ],
        start_date=start_date,
        end_date=end_date,
    )

    data = data.dropna()

    if len(data) < window:
        raise ValueError(
            f"Only {len(data)} observations are "
            f"available, but window={window}."
        )

    # Full-period correlation
    full_correlation = (
        data[ticker_1]
        .corr(
            data[ticker_2]
        )
    )

    # Rolling correlation
    rolling_correlation = (
        data[ticker_1]
        .rolling(window)
        .corr(
            data[ticker_2]
        )
    )

    result = rolling_correlation.to_frame(
        name="rolling_correlation"
    )

    result["ticker_1"] = ticker_1
    result["ticker_2"] = ticker_2

    result["window"] = window

    result = result[
        [
            "ticker_1",
            "ticker_2",
            "window",
            "rolling_correlation",
        ]
    ]

    # Summary statistics
    valid = (
        rolling_correlation
        .dropna()
    )

    summary = {
        "ticker_1": ticker_1,
        "ticker_2": ticker_2,
        "start_date":
            data.index.min(),
        "end_date":
            data.index.max(),
        "observations":
            len(data),
        "window":
            window,
        "full_period_correlation":
            full_correlation,
        "mean_rolling_correlation":
            valid.mean(),
        "median_rolling_correlation":
            valid.median(),
        "min_rolling_correlation":
            valid.min(),
        "max_rolling_correlation":
            valid.max(),
        "rolling_correlation_std":
            valid.std(),
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    safe_name = (
        f"{ticker_1}_{ticker_2}"
        f"_{window}d"
    )

    if save:

        result.to_csv(
            RESULTS_DIR
            / f"rolling_corr_{safe_name}.csv"
        )

        summary_df.to_csv(
            RESULTS_DIR
            / f"rolling_corr_{safe_name}_summary.csv",
            index=False,
        )

    if plot:

        fig, ax = plt.subplots(
            figsize=(13, 6)
        )

        ax.plot(
            result.index,
            result[
                "rolling_correlation"
            ],
            linewidth=1.2,
        )

        ax.axhline(
            full_correlation,
            linestyle="--",
            linewidth=1,
            label=(
                "Full-period correlation"
            ),
        )

        ax.axhline(
            0,
            linewidth=0.8,
        )

        ax.set_title(
            f"{ticker_1} vs {ticker_2} "
            f"Rolling Correlation "
            f"({window} Trading Days)"
        )

        ax.set_ylabel(
            "Correlation"
        )

        ax.set_xlabel(
            "Date"
        )

        ax.legend()

        fig.tight_layout()

        if save:
            fig.savefig(
                RESULTS_DIR
                / f"rolling_corr_{safe_name}.png",
                dpi=200,
            )

        plt.close(fig)

    print("\n")
    print("=" * 70)
    print(
        f"ROLLING CORRELATION: "
        f"{ticker_1} vs {ticker_2}"
    )
    print("=" * 70)

    print(
        f"Period: "
        f"{data.index.min().date()} "
        f"to "
        f"{data.index.max().date()}"
    )

    print(
        f"Observations: {len(data)}"
    )

    print(
        f"Window: {window} trading days"
    )

    print(
        f"Full-period correlation: "
        f"{full_correlation:.3f}"
    )

    print(
        f"Mean rolling correlation: "
        f"{valid.mean():.3f}"
    )

    print(
        f"Minimum rolling correlation: "
        f"{valid.min():.3f}"
    )

    print(
        f"Maximum rolling correlation: "
        f"{valid.max():.3f}"
    )

    return result, summary_df


def compare_multiple_pairs(
    returns,
    pairs,
    window=252,
    start_date=None,
    end_date=None,
):
    """
    Run rolling correlation analysis for multiple pairs.
    """

    summaries = []

    for ticker_1, ticker_2 in pairs:

        _, summary = analyze_pair(
            returns,
            ticker_1,
            ticker_2,
            window=window,
            start_date=start_date,
            end_date=end_date,
            save=True,
            plot=True,
        )

        summaries.append(summary)

    combined = pd.concat(
        summaries,
        ignore_index=True,
    )

    combined.to_csv(
        RESULTS_DIR
        / f"rolling_correlation_pair_summary_{window}d.csv",
        index=False,
    )

    return combined