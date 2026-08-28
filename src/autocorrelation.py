import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import acf

from .config import (
    MAX_ACF_LAG,
    RESULTS_DIR,
    ROLLING_ACF_LAG,
    ROLLING_WINDOW,
)


def full_sample_acf(returns):
    """
    Calculate autocorrelation at lags 1 through
    MAX_ACF_LAG for every stock.
    """

    rows = []

    for ticker in returns.columns:

        series = returns[ticker].dropna()

        if len(series) <= MAX_ACF_LAG:
            continue

        values = acf(
            series,
            nlags=MAX_ACF_LAG,
            fft=True,
            missing="drop"
        )

        # Approximate 95% threshold:
        # +/- 1.96 / sqrt(T)
        threshold = (
            1.96 / np.sqrt(len(series))
        )

        for lag in range(
            1,
            MAX_ACF_LAG + 1
        ):

            value = values[lag]

            rows.append(
                {
                    "ticker": ticker,
                    "lag": lag,
                    "autocorrelation": value,
                    "observations":
                        len(series),
                    "approx_95_threshold":
                        threshold,
                    "significant":
                        abs(value) > threshold,
                }
            )

    results = pd.DataFrame(rows)

    results.to_csv(
        RESULTS_DIR /
        "full_sample_autocorrelation.csv",
        index=False
    )

    return results


def rolling_autocorrelation(
    returns,
    window=ROLLING_WINDOW,
    lag=ROLLING_ACF_LAG
):
    """
    Calculate rolling autocorrelation.

    Default:
        252 trading-day window
        lag = 1
    """

    output = pd.DataFrame(
        index=returns.index
    )

    for ticker in returns.columns:

        series = returns[ticker]

        shifted = series.shift(lag)

        rolling_corr = (
            series
            .rolling(window)
            .corr(shifted)
        )

        output[ticker] = rolling_corr

    output.to_csv(
        RESULTS_DIR /
        "rolling_autocorrelation.csv"
    )

    return output


def autocorrelation_summary(
    rolling_acf,
    window=ROLLING_WINDOW
):
    """
    Classify each stock's rolling lag-1 autocorrelation
    into three buckets:

        Strong Positive
        Strong Negative
        Everything Else

    The approximate 95% threshold is:
        +/- 1.96 / sqrt(window)

    Returns one row per stock with the percentage of
    valid windows spent in each regime.
    """

    threshold = 1.96 / np.sqrt(window)

    rows = []

    for ticker in rolling_acf.columns:

        series = rolling_acf[ticker].dropna()

        if series.empty:
            continue

        strong_positive = series > threshold
        strong_negative = series < -threshold
        everything_else = (
            ~strong_positive
            & ~strong_negative
        )

        rows.append(
            {
                "ticker": ticker,
                "mean_rolling_acf": series.mean(),
                "max_rolling_acf": series.max(),
                "min_rolling_acf": series.min(),
                "pct_strong_positive":
                    strong_positive.mean(),
                "pct_strong_negative":
                    strong_negative.mean(),
                "pct_everything_else":
                    everything_else.mean(),
                "observations":
                    len(series),
                "threshold":
                    threshold,
            }
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        "pct_strong_positive",
        ascending=False
    )

    summary.to_csv(
        RESULTS_DIR /
        "autocorrelation_summary.csv",
        index=False
    )

    return summary

def autocorrelation_regime_series(
    rolling_acf,
    window=ROLLING_WINDOW
):
    """
    Convert rolling autocorrelation values into
    three categorical regimes for every stock/date.

    Regimes:
        Strong Positive
        Strong Negative
        Everything Else
    """

    threshold = 1.96 / np.sqrt(window)

    # Start with Everything Else for valid observations.
    regimes = pd.DataFrame(
        "Everything Else",
        index=rolling_acf.index,
        columns=rolling_acf.columns,
        dtype="object",
    )

    # Preserve missing observations as NaN.
    regimes = regimes.where(
        rolling_acf.notna()
    )

    # Assign strong positive observations.
    regimes = regimes.mask(
        rolling_acf > threshold,
        "Strong Positive"
    )

    # Assign strong negative observations.
    regimes = regimes.mask(
        rolling_acf < -threshold,
        "Strong Negative"
    )

    regimes.to_csv(
        RESULTS_DIR /
        "autocorrelation_regimes.csv"
    )

    return regimes


def autocorrelation_regime_date_summary(
    regimes
):
    """
    Summarize autocorrelation regimes across stocks
    for every date.

    This lets us identify dates where many stocks
    simultaneously exhibit strong positive or negative
    autocorrelation.
    """

    rows = []

    for date, row in regimes.iterrows():

        valid = row.dropna()

        if valid.empty:
            continue

        strong_positive = (
            valid == "Strong Positive"
        ).sum()

        strong_negative = (
            valid == "Strong Negative"
        ).sum()

        everything_else = (
            valid == "Everything Else"
        ).sum()

        total = len(valid)

        rows.append(
            {
                "date": date,
                "strong_positive_stocks":
                    strong_positive,
                "strong_negative_stocks":
                    strong_negative,
                "everything_else_stocks":
                    everything_else,
                "total_stocks":
                    total,
                "pct_strong_positive":
                    strong_positive / total,
                "pct_strong_negative":
                    strong_negative / total,
                "pct_everything_else":
                    everything_else / total,
            }
        )

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    summary.to_csv(
        RESULTS_DIR /
        "autocorrelation_regime_by_date.csv",
        index=False
    )

    return summary

def strongest_autocorrelation_regime_dates(
    regime_date_summary,
    n=10
):
    """
    Return the strongest dates for positive and negative
    cross-stock autocorrelation regimes.
    """

    if regime_date_summary.empty:
        return pd.DataFrame()

    positive = (
        regime_date_summary
        .sort_values(
            "strong_positive_stocks",
            ascending=False
        )
        .head(n)
        .copy()
    )

    positive["dominant_regime"] = (
        "Strong Positive"
    )

    negative = (
        regime_date_summary
        .sort_values(
            "strong_negative_stocks",
            ascending=False
        )
        .head(n)
        .copy()
    )

    negative["dominant_regime"] = (
        "Strong Negative"
    )

    result = pd.concat(
        [positive, negative],
        ignore_index=True,
    )

    result = (
        result
        .sort_values(
            ["dominant_regime", "strong_positive_stocks"],
            ascending=[True, False],
        )
    )

    result.to_csv(
        RESULTS_DIR /
        "strongest_autocorrelation_regime_dates.csv",
        index=False
    )

    return result

def plot_rolling_acf(rolling_acf):
    """
    Plot rolling lag-1 autocorrelation for
    every stock.
    """

    fig, ax = plt.subplots(
        figsize=(14, 9)
    )

    for ticker in rolling_acf.columns:

        ax.plot(
            rolling_acf.index,
            rolling_acf[ticker],
            alpha=0.55,
            linewidth=0.8,
            label=ticker
        )

    threshold = (
        1.96 /
        np.sqrt(ROLLING_WINDOW)
    )

    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1
    )

    ax.axhline(
        -threshold,
        linestyle="--",
        linewidth=1
    )

    ax.axhline(
        0,
        linewidth=0.8
    )

    ax.set_title(
        "Rolling Lag-1 Autocorrelation "
        f"({ROLLING_WINDOW}-Day Window)"
    )

    ax.set_ylabel("Autocorrelation")

    ax.legend(
        ncol=4,
        fontsize=7
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR /
        "rolling_autocorrelation.png",
        dpi=200
    )

    plt.close(fig)


def compare_cluster_autocorrelation(
    rolling_acf,
    clusters
):
    """
    Calculate average rolling autocorrelation
    for each cluster.

    This helps answer Thomas's question:
    do stocks grouped together experience
    autocorrelation at similar times?
    """

    cluster_series = {}

    for cluster_id in sorted(
        clusters["cluster"].unique()
    ):

        members = clusters.loc[
            clusters["cluster"]
            == cluster_id,
            "ticker"
        ].tolist()

        available = [
            ticker
            for ticker in members
            if ticker in rolling_acf.columns
        ]

        if not available:
            continue

        cluster_series[
            f"Cluster_{cluster_id}"
        ] = (
            rolling_acf[available]
            .mean(axis=1)
        )

    result = pd.DataFrame(
        cluster_series
    )

    result.to_csv(
        RESULTS_DIR /
        "cluster_rolling_autocorrelation.csv"
    )

    return result

def plot_autocorrelation_regime_timeline(
    rolling_acf,
    window=ROLLING_WINDOW,
):
    """
    Plot the number of stocks in each autocorrelation
    regime through time.

    Regimes:
        Strong Positive: ACF > +1.96 / sqrt(window)
        Strong Negative: ACF < -1.96 / sqrt(window)
        Everything Else: between the two thresholds

    Only stocks with a valid rolling ACF observation
    on a given date are counted.
    """

    threshold = 1.96 / np.sqrt(window)

    valid = rolling_acf.notna()

    strong_positive = (
        (rolling_acf > threshold) & valid
    )

    strong_negative = (
        (rolling_acf < -threshold) & valid
    )

    everything_else = (
        (rolling_acf >= -threshold)
        & (rolling_acf <= threshold)
        & valid
    )

    regime_counts = pd.DataFrame(
        {
            "Strong Positive":
                strong_positive.sum(axis=1),
            "Strong Negative":
                strong_negative.sum(axis=1),
            "Everything Else":
                everything_else.sum(axis=1),
            "Available Stocks":
                valid.sum(axis=1),
        },
        index=rolling_acf.index,
    )

    # Remove dates before any rolling ACF exists.
    regime_counts = regime_counts.loc[
        regime_counts["Available Stocks"] > 0
    ].copy()

    regime_counts.index.name = "date"

    regime_counts.to_csv(
        RESULTS_DIR
        / "autocorrelation_regime_timeline.csv"
    )

    # --------------------------------------------------
    # Main chart
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(16, 8)
    )

    ax.plot(
        regime_counts.index,
        regime_counts["Strong Positive"],
        label="Strong Positive",
        linewidth=1.2,
    )

    ax.plot(
        regime_counts.index,
        regime_counts["Strong Negative"],
        label="Strong Negative",
        linewidth=1.2,
    )

    ax.plot(
        regime_counts.index,
        regime_counts["Everything Else"],
        label="Everything Else",
        linewidth=1.0,
        alpha=0.65,
    )

    ax.set_title(
        "Autocorrelation Regimes Across Energy Stocks\n"
        f"{window}-Day Rolling Lag-1 Autocorrelation"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Stocks")

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR
        / "autocorrelation_regime_timeline.png",
        dpi=200,
    )

    plt.close(fig)

    return regime_counts

def autocorrelation_regime_summary(regimes):
    """
    Summarize the percentage of observations spent by each stock
    in the three autocorrelation regimes:

        Strong Positive
        Strong Negative
        Everything Else

    Parameters
    ----------
    regimes : pandas.DataFrame
        DataFrame with dates as the index, stocks as columns, and
        regime labels as values.

    Returns
    -------
    pandas.DataFrame
        One row per ticker with the percentage of observations
        in each regime.
    """

    rows = []

    for ticker in regimes.columns:

        series = regimes[ticker].dropna()

        if series.empty:
            continue

        total = len(series)

        strong_positive = (
            series == "Strong Positive"
        ).sum()

        strong_negative = (
            series == "Strong Negative"
        ).sum()

        everything_else = (
            series == "Everything Else"
        ).sum()

        rows.append(
            {
                "ticker": ticker,
                "pct_strong_positive":
                    strong_positive / total,
                "pct_strong_negative":
                    strong_negative / total,
                "pct_everything_else":
                    everything_else / total,
            }
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        [
            "pct_strong_positive",
            "pct_strong_negative",
        ],
        ascending=[
            False,
            False,
        ],
    )

    summary.to_csv(
        RESULTS_DIR /
        "autocorrelation_regime_summary.csv",
        index=False,
    )

    return summary

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    RESULTS_DIR,
    ROLLING_WINDOW,
)


def build_autocorrelation_regime_series(
    rolling_acf,
    window=ROLLING_WINDOW,
):
    """
    Classify each stock/date into one of three regimes:

        Strong Positive
        Strong Negative
        Everything Else

    Threshold:
        +/- 1.96 / sqrt(window)

    Returns a DataFrame with the same index/columns as rolling_acf.
    """

    threshold = 1.96 / np.sqrt(window)

    regimes = pd.DataFrame(
        "Everything Else",
        index=rolling_acf.index,
        columns=rolling_acf.columns,
    )

    strong_positive = rolling_acf > threshold
    strong_negative = rolling_acf < -threshold

    regimes = regimes.mask(
        strong_positive,
        "Strong Positive",
    )

    regimes = regimes.mask(
        strong_negative,
        "Strong Negative",
    )

    # Missing rolling windows should remain missing.
    regimes = regimes.where(
        rolling_acf.notna()
    )

    return regimes


def autocorrelation_regime_breadth(
    rolling_acf,
    window=ROLLING_WINDOW,
):
    """
    Calculate the number and percentage of stocks in each
    autocorrelation regime on every date.
    """

    threshold = 1.96 / np.sqrt(window)

    available = rolling_acf.notna().sum(axis=1)

    strong_positive = (
        rolling_acf > threshold
    ).sum(axis=1)

    strong_negative = (
        rolling_acf < -threshold
    ).sum(axis=1)

    everything_else = (
        available
        - strong_positive
        - strong_negative
    )

    result = pd.DataFrame(
        {
            "strong_positive_stocks":
                strong_positive,

            "strong_negative_stocks":
                strong_negative,

            "everything_else_stocks":
                everything_else,

            "available_stocks":
                available,
        },
        index=rolling_acf.index,
    )

    result["pct_strong_positive"] = (
        result["strong_positive_stocks"]
        / result["available_stocks"]
    )

    result["pct_strong_negative"] = (
        result["strong_negative_stocks"]
        / result["available_stocks"]
    )

    result["pct_everything_else"] = (
        result["everything_else_stocks"]
        / result["available_stocks"]
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result.to_csv(
        RESULTS_DIR /
        "autocorrelation_regime_breadth.csv"
    )

    return result


def plot_autocorrelation_regime_breadth(
    breadth,
):
    """
    Plot the percentage of available stocks in each
    autocorrelation regime.

    This is the main autocorrelation visualization.
    """

    fig, ax = plt.subplots(
        figsize=(15, 8)
    )

    dates = breadth.index

    ax.stackplot(
        dates,
        breadth["pct_strong_positive"] * 100,
        breadth["pct_strong_negative"] * 100,
        breadth["pct_everything_else"] * 100,
        labels=[
            "Strong Positive",
            "Strong Negative",
            "Everything Else",
        ],
        alpha=0.8,
    )

    ax.set_title(
        "Autocorrelation Regimes Across Energy Stocks"
    )

    ax.set_ylabel(
        "Percentage of Available Stocks"
    )

    ax.set_xlabel("Date")

    ax.set_ylim(0, 100)

    ax.legend(
        loc="upper left"
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR /
        "autocorrelation_regime_breadth.png",
        dpi=200,
    )

    plt.close(fig)


def plot_autocorrelation_positive_negative_breadth(
    breadth,
):
    """
    Plot only strong positive and strong negative
    autocorrelation breadth.

    Positive regime is plotted above zero.
    Negative regime is plotted below zero.
    """

    fig, ax = plt.subplots(
        figsize=(15, 8)
    )

    dates = breadth.index

    positive = (
        breadth["pct_strong_positive"] * 100
    )

    negative = (
        -breadth["pct_strong_negative"] * 100
    )

    ax.plot(
        dates,
        positive,
        linewidth=1.2,
        label="Strong Positive",
    )

    ax.plot(
        dates,
        negative,
        linewidth=1.2,
        label="Strong Negative",
    )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.set_title(
        "Strong Autocorrelation Breadth Across Energy Stocks"
    )

    ax.set_ylabel(
        "Percentage of Available Stocks"
    )

    ax.set_xlabel("Date")

    ax.legend(
        loc="upper left"
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR /
        "autocorrelation_positive_negative_breadth.png",
        dpi=200,
    )

    plt.close(fig)


def cluster_autocorrelation_participation(
    rolling_acf,
    clusters,
    top_dates=10,
    window=ROLLING_WINDOW,
):
    """
    For the strongest autocorrelation dates, calculate what
    proportion of each historical cluster is participating.

    A stock participates when its rolling autocorrelation is
    significantly positive or significantly negative.
    """

    threshold = 1.96 / np.sqrt(window)

    # Count participating stocks on each date.
    participation = (
        rolling_acf.abs() > threshold
    )

    total_participation = (
        participation
        .sum(axis=1)
        .sort_values(ascending=False)
    )

    selected_dates = (
        total_participation
        .head(top_dates)
        .index
        .tolist()
    )

    cluster_ids = (
        sorted(
            clusters["cluster"].unique()
        )
    )

    rows = []

    for cluster_id in cluster_ids:

        members = clusters.loc[
            clusters["cluster"] == cluster_id,
            "ticker",
        ].tolist()

        available_members = [
            ticker
            for ticker in members
            if ticker in participation.columns
        ]

        if not available_members:
            continue

        cluster_size = len(
            available_members
        )

        for date in selected_dates:

            participating = participation.loc[
                date,
                available_members,
            ].sum()

            rows.append(
                {
                    "date": date,
                    "cluster": cluster_id,
                    "cluster_size": cluster_size,
                    "participating_stocks":
                        participating,
                    "participation_pct":
                        participating / cluster_size,
                }
            )

    result = pd.DataFrame(rows)

    result.to_csv(
        RESULTS_DIR /
        "autocorrelation_cluster_participation.csv",
        index=False,
    )

    return result


def plot_autocorrelation_cluster_participation(
    participation,
):
    """
    Heatmap of cluster participation on the strongest
    autocorrelation dates.
    """

    if participation.empty:
        return

    pivot = participation.pivot(
        index="cluster",
        columns="date",
        values="participation_pct",
    )

    pivot = pivot.sort_index()

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    image = ax.imshow(
        pivot.values * 100,
        aspect="auto",
    )

    ax.set_title(
        "Cluster Participation During Strong Autocorrelation Dates"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Historical Cluster")

    ax.set_xticks(
        range(len(pivot.columns))
    )

    ax.set_xticklabels(
        [
            pd.Timestamp(date).strftime("%Y-%m-%d")
            for date in pivot.columns
        ],
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        range(len(pivot.index))
    )

    ax.set_yticklabels(
        [
            f"Cluster {cluster}"
            for cluster in pivot.index
        ]
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Participating Stocks (%)",
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR /
        "autocorrelation_cluster_participation.png",
        dpi=200,
    )

    plt.close(fig)


def plot_autocorrelation_regime_timeline(
    rolling_acf,
    clusters=None,
    window=ROLLING_WINDOW,
):
    """
    Create the main autocorrelation visualizations.

    Returns the regime breadth DataFrame.
    """

    breadth = autocorrelation_regime_breadth(
        rolling_acf,
        window=window,
    )

    plot_autocorrelation_regime_breadth(
        breadth
    )

    plot_autocorrelation_positive_negative_breadth(
        breadth
    )

    if clusters is not None:

        participation = (
            cluster_autocorrelation_participation(
                rolling_acf,
                clusters,
                top_dates=10,
                window=window,
            )
        )

        plot_autocorrelation_cluster_participation(
            participation
        )

    return breadth

def plot_autocorrelation_regime_heatmap(
    regime_series,
    filename="autocorrelation_regime_heatmap.png",
):
    """
    Plot each stock's rolling autocorrelation regime through time.

    Regimes:
        Strong Positive  =  1
        Everything Else  =  0
        Strong Negative  = -1

    Rows are stocks and columns are dates.
    """

    from matplotlib.colors import BoundaryNorm, ListedColormap

    # Convert text regimes to numeric values
    regime_map = {
        "Strong Negative": -1,
        "Everything Else": 0,
        "Strong Positive": 1,
    }

    numeric = regime_series.replace(regime_map)

    # Make sure everything is numeric
    numeric = numeric.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # --------------------------------------------------
    # Order stocks by their historical cluster
    # if desired later. For now preserve column order.
    # --------------------------------------------------

    numeric = numeric.T

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(16, 12)
    )

    cmap = ListedColormap(
        [
            "#2166ac",  # negative
            "#f2f2f2",  # neutral
            "#b2182b",  # positive
        ]
    )

    norm = BoundaryNorm(
        [-1.5, -0.5, 0.5, 1.5],
        cmap.N,
    )

    image = ax.imshow(
        numeric.values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )

    # --------------------------------------------------
    # Y axis: stock names
    # --------------------------------------------------

    ax.set_yticks(
        np.arange(len(numeric.index))
    )

    ax.set_yticklabels(
        numeric.index,
        fontsize=8,
    )

    # --------------------------------------------------
    # X axis: dates
    # --------------------------------------------------

    dates = pd.to_datetime(
        numeric.columns
    )

    # Approximately 12 labels across the chart
    number_of_ticks = min(
        12,
        len(dates),
    )

    tick_positions = np.linspace(
        0,
        len(dates) - 1,
        number_of_ticks,
        dtype=int,
    )

    ax.set_xticks(
        tick_positions
    )

    ax.set_xticklabels(
        [
            dates[i].strftime("%Y-%m")
            for i in tick_positions
        ],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------

    ax.set_title(
        "Rolling Lag-1 Autocorrelation Regimes by Stock",
        fontsize=14,
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Stock")

    # --------------------------------------------------
    # Colorbar
    # --------------------------------------------------

    colorbar = fig.colorbar(
        image,
        ax=ax,
        ticks=[-1, 0, 1],
        fraction=0.025,
        pad=0.02,
    )

    colorbar.ax.set_yticklabels(
        [
            "Strong Negative",
            "Everything Else",
            "Strong Positive",
        ]
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / filename,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    return numeric

def plot_autocorrelation_breadth(regime_breadth):
    """
    Plot the number of stocks experiencing strong positive
    and strong negative rolling autocorrelation on each date.

    regime_breadth is expected to contain:
        strong_positive_stocks
        strong_negative_stocks
    """

    import matplotlib.pyplot as plt

    required_columns = [
        "strong_positive_stocks",
        "strong_negative_stocks",
    ]

    missing = [
        column
        for column in required_columns
        if column not in regime_breadth.columns
    ]

    if missing:
        raise ValueError(
            "regime_breadth is missing required columns: "
            + ", ".join(missing)
        )

    data = regime_breadth[
        required_columns
    ].copy()

    # Make sure the index is datetime.
    data.index = pd.to_datetime(data.index)

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    ax.plot(
        data.index,
        data["strong_positive_stocks"],
        linewidth=1.2,
        label="Strong Positive",
    )

    ax.plot(
        data.index,
        data["strong_negative_stocks"],
        linewidth=1.2,
        label="Strong Negative",
    )

    ax.set_title(
        "Cross-Stock Autocorrelation Breadth"
    )

    ax.set_ylabel(
        "Number of Stocks"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    output_path = (
        RESULTS_DIR /
        "autocorrelation_breadth_timeline.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nSaved autocorrelation breadth chart to: "
        f"{output_path}"
    )

    return data

def plot_autocorrelation_breadth_lines(
    rolling_acf,
    window=252,
    top_n=8,
):
    """Plot the number of stocks in strong positive/negative ACF regimes.

    Strong positive: rolling ACF > +1.96/sqrt(window)
    Strong negative: rolling ACF < -1.96/sqrt(window)

    Also marks the largest synchronised episodes and draws the
    3-stock threshold used by the episode analysis.
    """
    import matplotlib.pyplot as plt

    threshold = 1.96 / np.sqrt(window)

    positive_count = (rolling_acf > threshold).sum(axis=1)
    negative_count = (rolling_acf < -threshold).sum(axis=1)

    breadth = pd.DataFrame(
        {
            "strong_positive_stocks": positive_count,
            "strong_negative_stocks": negative_count,
        },
        index=rolling_acf.index,
    )

    # Only plot dates where at least one stock is available.
    breadth = breadth.loc[
        (breadth["strong_positive_stocks"] > 0)
        | (breadth["strong_negative_stocks"] > 0)
    ].copy()

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(
        breadth.index,
        breadth["strong_positive_stocks"],
        label="Strong Positive",
        linewidth=1.5,
    )

    ax.plot(
        breadth.index,
        breadth["strong_negative_stocks"],
        label="Strong Negative",
        linewidth=1.5,
    )

    # Three stocks is the minimum used to define a common episode.
    ax.axhline(
        3,
        linestyle="--",
        linewidth=1,
        label="3-stock common-episode threshold",
    )

    # Identify the largest positive and negative synchronised dates.
    candidates = []

    for date, row in breadth.iterrows():
        pos = int(row["strong_positive_stocks"])
        neg = int(row["strong_negative_stocks"])
        if pos > 0:
            candidates.append((pos, "positive", date))
        if neg > 0:
            candidates.append((neg, "negative", date))

    candidates.sort(key=lambda x: x[0], reverse=True)

    # Keep one annotation per date so the chart does not become cluttered.
    used_dates = set()
    annotations = []

    for count, direction, date in candidates:
        if date in used_dates:
            continue
        used_dates.add(date)
        annotations.append((count, direction, date))
        if len(annotations) >= top_n:
            break

    for count, direction, date in annotations:
        if direction == "positive":
            y = breadth.loc[date, "strong_positive_stocks"]
            text = f"{date:%d %b %Y}: {count} positive"
            va = "bottom"
            y_offset = 0.8
        else:
            y = breadth.loc[date, "strong_negative_stocks"]
            text = f"{date:%d %b %Y}: {count} negative"
            va = "bottom"
            y_offset = 0.8

        ax.annotate(
            text,
            xy=(date, y),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
            arrowprops={"arrowstyle": "-", "linewidth": 0.8},
        )

    ax.set_title(
        "Synchronised Autocorrelation Across Energy Stocks\n"
        f"252-Day Rolling Lag-1 ACF; |ACF| > {threshold:.4f}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Stocks")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    output_path = RESULTS_DIR / "autocorrelation_breadth_lines.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    breadth.to_csv(
        RESULTS_DIR / "autocorrelation_breadth_lines.csv"
    )

    print(
        f"\nSaved autocorrelation breadth plot to: {output_path}"
    )

    return breadth