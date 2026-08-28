import numpy as np
import pandas as pd

from .config import RESULTS_DIR, ROLLING_WINDOW


def build_significance_matrix(
    rolling_acf,
    window=ROLLING_WINDOW,
):
    """
    Convert rolling autocorrelation values into boolean matrices
    for significant, positive, and negative autocorrelation.

    Approximate two-sided 95% threshold:

        +/- 1.96 / sqrt(window)
    """

    threshold = (
        1.96 / np.sqrt(window)
    )

    significant = (
        rolling_acf.abs() > threshold
    )

    positive = (
        rolling_acf > threshold
    )

    negative = (
        rolling_acf < -threshold
    )

    return (
        significant,
        positive,
        negative,
        threshold,
    )


def build_daily_autocorrelation_counts(
    rolling_acf,
    window=ROLLING_WINDOW,
):
    """
    Count significant autocorrelation across the universe
    on every date.
    """

    (
        significant,
        positive,
        negative,
        threshold,
    ) = build_significance_matrix(
        rolling_acf,
        window,
    )

    available_stocks = (
        rolling_acf.notna().sum(axis=1)
    )

    significant_stocks = (
        significant.sum(axis=1)
    )

    positive_stocks = (
        positive.sum(axis=1)
    )

    negative_stocks = (
        negative.sum(axis=1)
    )

    result = pd.DataFrame(
        {
            "available_stocks":
                available_stocks,

            "significant_stocks":
                significant_stocks,

            "positive_stocks":
                positive_stocks,

            "negative_stocks":
                negative_stocks,

            "pct_stocks_significant":
                (
                    significant_stocks
                    / available_stocks
                ),

            "pct_stocks_positive":
                (
                    positive_stocks
                    / available_stocks
                ),

            "pct_stocks_negative":
                (
                    negative_stocks
                    / available_stocks
                ),
        },
        index=rolling_acf.index,
    )

    result["threshold"] = threshold

    result.to_csv(
        RESULTS_DIR
        / "daily_autocorrelation_counts.csv"
    )

    return result


def classify_breadth(
    significant_count,
):
    """
    Classify the breadth of an autocorrelation event.

    These categories are descriptive rather than statistical
    significance claims about the market as a whole.
    """

    if significant_count <= 2:
        return "Isolated"

    if significant_count <= 5:
        return "Small Common"

    if significant_count <= 10:
        return "Broad"

    return "Market-Wide"


def build_breadth_regimes(
    rolling_acf,
    window=ROLLING_WINDOW,
):
    """
    Add a qualitative breadth classification to every date.
    """

    counts = (
        build_daily_autocorrelation_counts(
            rolling_acf,
            window,
        )
    )

    result = counts.copy()

    result["breadth_regime"] = (
        result["significant_stocks"]
        .apply(
            classify_breadth
        )
    )

    result.to_csv(
        RESULTS_DIR
        / "autocorrelation_breadth_regimes.csv"
    )

    return result


def summarize_breadth_regimes(
    rolling_acf,
    window=ROLLING_WINDOW,
):
    """
    Summarize how frequently the universe falls into each
    autocorrelation breadth regime.
    """

    regimes = (
        build_breadth_regimes(
            rolling_acf,
            window,
        )
    )

    summary = (
        regimes
        .groupby(
            "breadth_regime"
        )
        .agg(
            days=(
                "breadth_regime",
                "size",
            ),
            mean_significant_stocks=(
                "significant_stocks",
                "mean",
            ),
            max_significant_stocks=(
                "significant_stocks",
                "max",
            ),
            mean_pct_stocks_significant=(
                "pct_stocks_significant",
                "mean",
            ),
        )
        .reset_index()
    )

    total_days = len(regimes)

    summary["pct_of_dates"] = (
        summary["days"]
        / total_days
    )

    regime_order = [
        "Isolated",
        "Small Common",
        "Broad",
        "Market-Wide",
    ]

    summary["breadth_regime"] = pd.Categorical(
        summary["breadth_regime"],
        categories=regime_order,
        ordered=True,
    )

    summary = (
        summary
        .sort_values(
            "breadth_regime"
        )
        .reset_index(drop=True)
    )

    summary.to_csv(
        RESULTS_DIR
        / "autocorrelation_breadth_summary.csv",
        index=False,
    )

    return summary


def identify_peak_dates(
    rolling_acf,
    window=ROLLING_WINDOW,
    top_n=20,
):
    """
    Find dates with the greatest cross-sectional number of
    significant autocorrelation readings.
    """

    counts = (
        build_breadth_regimes(
            rolling_acf,
            window,
        )
    )

    result = (
        counts
        .sort_values(
            [
                "significant_stocks",
                "pct_stocks_significant",
            ],
            ascending=False,
        )
        .head(top_n)
        .copy()
    )

    result.to_csv(
        RESULTS_DIR
        / "autocorrelation_peak_dates.csv"
    )

    return result


def get_peak_date_participants(
    rolling_acf,
    peak_dates,
    window=ROLLING_WINDOW,
):
    """
    For each peak date, list every participating stock,
    its ACF value, and its direction.
    """

    (
        significant,
        positive,
        negative,
        threshold,
    ) = build_significance_matrix(
        rolling_acf,
        window,
    )

    rows = []

    for date in peak_dates:

        if date not in rolling_acf.index:
            continue

        for ticker in rolling_acf.columns:

            value = rolling_acf.loc[
                date,
                ticker,
            ]

            if pd.isna(value):
                continue

            if not significant.loc[
                date,
                ticker,
            ]:
                continue

            if positive.loc[
                date,
                ticker,
            ]:

                direction = "Positive"

            elif negative.loc[
                date,
                ticker,
            ]:

                direction = "Negative"

            else:

                direction = "Significant"

            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "autocorrelation": value,
                    "direction": direction,
                    "threshold": threshold,
                }
            )

    result = pd.DataFrame(rows)

    if not result.empty:

        result = result.sort_values(
            [
                "date",
                "direction",
                "autocorrelation",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )

    result.to_csv(
        RESULTS_DIR
        / "autocorrelation_peak_participants.csv",
        index=False,
    )

    return result


def map_peak_participants_to_clusters(
    participants,
    clusters,
):
    """
    Attach the full-universe cluster assignment to every
    stock participating in a peak autocorrelation date.
    """

    cluster_map = (
        clusters[
            [
                "ticker",
                "cluster",
            ]
        ]
        .drop_duplicates(
            "ticker"
        )
        .set_index(
            "ticker"
        )["cluster"]
    )

    result = participants.copy()

    result["cluster"] = (
        result["ticker"]
        .map(cluster_map)
    )

    result.to_csv(
        RESULTS_DIR
        / "autocorrelation_peak_participants_with_clusters.csv",
        index=False,
    )

    return result


def summarize_peak_dates_by_cluster(
    participants_with_clusters,
):
    """
    Summarize how many participating stocks from each cluster
    appear on each peak date.
    """

    if participants_with_clusters.empty:

        result = pd.DataFrame(
            columns=[
                "date",
                "cluster",
                "participating_stocks",
                "positive_stocks",
                "negative_stocks",
            ]
        )

    else:

        grouped = (
            participants_with_clusters
            .groupby(
                [
                    "date",
                    "cluster",
                ]
            )
        )

        result = (
            grouped
            .agg(
                participating_stocks=(
                    "ticker",
                    "nunique",
                ),
                positive_stocks=(
                    "direction",
                    lambda x:
                        (
                            x == "Positive"
                        ).sum(),
                ),
                negative_stocks=(
                    "direction",
                    lambda x:
                        (
                            x == "Negative"
                        ).sum(),
                ),
            )
            .reset_index()
        )

        result = result.sort_values(
            [
                "date",
                "participating_stocks",
            ],
            ascending=[
                True,
                False,
            ],
        )

    result.to_csv(
        RESULTS_DIR
        / "autocorrelation_peak_cluster_summary.csv",
        index=False,
    )

    return result


def stock_broad_episode_exposure(
    rolling_acf,
    window=ROLLING_WINDOW,
    broad_threshold=11,
):
    """
    For each stock, determine how frequently its significant
    autocorrelation occurs during broad cross-sectional events.

    broad_threshold defaults to 11 significant stocks.
    """

    (
        significant,
        positive,
        negative,
        threshold,
    ) = build_significance_matrix(
        rolling_acf,
        window,
    )

    counts = (
        build_daily_autocorrelation_counts(
            rolling_acf,
            window,
        )
    )

    broad_dates = (
        counts["significant_stocks"]
        >= broad_threshold
    )

    rows = []

    for ticker in rolling_acf.columns:

        stock_significant = (
            significant[ticker]
            .fillna(False)
        )

        total_significant = (
            stock_significant.sum()
        )

        broad_significant = (
            stock_significant
            & broad_dates
        ).sum()

        isolated_significant = (
            stock_significant
            & (
                counts[
                    "significant_stocks"
                ]
                <= 2
            )
        ).sum()

        if total_significant == 0:

            pct_broad = np.nan
            pct_isolated = np.nan

        else:

            pct_broad = (
                broad_significant
                / total_significant
            )

            pct_isolated = (
                isolated_significant
                / total_significant
            )

        rows.append(
            {
                "ticker": ticker,
                "significant_dates":
                    total_significant,
                "broad_episode_dates":
                    broad_significant,
                "pct_significant_during_broad":
                    pct_broad,
                "isolated_episode_dates":
                    isolated_significant,
                "pct_significant_during_isolated":
                    pct_isolated,
            }
        )

    result = (
        pd.DataFrame(rows)
        .sort_values(
            "pct_significant_during_broad",
            ascending=False,
        )
    )

    result.to_csv(
        RESULTS_DIR
        / "stock_autocorrelation_episode_exposure.csv",
        index=False,
    )

    return result


def stock_overlap_matrix(
    rolling_acf,
    window=ROLLING_WINDOW,
):
    """
    Jaccard-style overlap in significant-autocorrelation dates.
    """

    (
        significant,
        positive,
        negative,
        threshold,
    ) = build_significance_matrix(
        rolling_acf,
        window,
    )

    tickers = list(
        rolling_acf.columns
    )

    matrix = pd.DataFrame(
        np.nan,
        index=tickers,
        columns=tickers,
    )

    for ticker_1 in tickers:

        for ticker_2 in tickers:

            s1 = (
                significant[ticker_1]
                .fillna(False)
            )

            s2 = (
                significant[ticker_2]
                .fillna(False)
            )

            both = (
                s1 & s2
            )

            either = (
                s1 | s2
            )

            denominator = (
                either.sum()
            )

            if denominator == 0:
                overlap = np.nan
            else:
                overlap = (
                    both.sum()
                    / denominator
                )

            matrix.loc[
                ticker_1,
                ticker_2,
            ] = overlap

    matrix.to_csv(
        RESULTS_DIR
        / "autocorrelation_overlap_matrix.csv"
    )

    return matrix


def directional_overlap_matrix(
    rolling_acf,
    window=ROLLING_WINDOW,
):
    """
    Among dates when both stocks are significant,
    measure how often their autocorrelation has the same sign.
    """

    (
        significant,
        positive,
        negative,
        threshold,
    ) = build_significance_matrix(
        rolling_acf,
        window,
    )

    tickers = list(
        rolling_acf.columns
    )

    matrix = pd.DataFrame(
        np.nan,
        index=tickers,
        columns=tickers,
    )

    for ticker_1 in tickers:

        for ticker_2 in tickers:

            same_positive = (
                positive[ticker_1].fillna(False)
                &
                positive[ticker_2].fillna(False)
            )

            same_negative = (
                negative[ticker_1].fillna(False)
                &
                negative[ticker_2].fillna(False)
            )

            both_significant = (
                significant[ticker_1].fillna(False)
                &
                significant[ticker_2].fillna(False)
            )

            denominator = (
                both_significant.sum()
            )

            if denominator == 0:

                value = np.nan

            else:

                value = (
                    (
                        same_positive
                        | same_negative
                    ).sum()
                    / denominator
                )

            matrix.loc[
                ticker_1,
                ticker_2,
            ] = value

    matrix.to_csv(
        RESULTS_DIR
        / "autocorrelation_same_direction_matrix.csv"
    )

    return matrix


def print_autocorrelation_episode_report(
    rolling_acf,
    clusters=None,
    window=ROLLING_WINDOW,
    minimum_stocks=3,
    top_n=10,
):
    """
    Run the complete cross-sectional autocorrelation episode analysis.
    """

    counts = (
        build_daily_autocorrelation_counts(
            rolling_acf,
            window,
        )
    )

    breadth = (
        build_breadth_regimes(
            rolling_acf,
            window,
        )
    )

    breadth_summary = (
        summarize_breadth_regimes(
            rolling_acf,
            window,
        )
    )

    peaks = (
        identify_peak_dates(
            rolling_acf,
            window,
            top_n=top_n,
        )
    )

    peak_dates = peaks.index.tolist()

    participants = (
        get_peak_date_participants(
            rolling_acf,
            peak_dates,
            window,
        )
    )

    if (
        clusters is not None
        and not participants.empty
    ):

        participants_with_clusters = (
            map_peak_participants_to_clusters(
                participants,
                clusters,
            )
        )

        cluster_summary = (
            summarize_peak_dates_by_cluster(
                participants_with_clusters,
            )
        )

    else:

        participants_with_clusters = (
            participants.copy()
        )

        cluster_summary = (
            pd.DataFrame()
        )

    stock_exposure = (
        stock_broad_episode_exposure(
            rolling_acf,
            window,
            broad_threshold=11,
        )
    )

    overlap = (
        stock_overlap_matrix(
            rolling_acf,
            window,
        )
    )

    directional = (
        directional_overlap_matrix(
            rolling_acf,
            window,
        )
    )

    print("\n")
    print("=" * 70)
    print(
        "AUTOCORRELATION EPISODE ANALYSIS"
    )
    print("=" * 70)

    threshold = (
        1.96 / np.sqrt(window)
    )

    print(
        f"Window: {window} trading days"
    )

    print(
        f"Significance threshold: "
        f"+/- {threshold:.4f}"
    )

    print(
        f"Minimum stocks for episode: "
        f"{minimum_stocks}"
    )

    print("\n")
    print(
        "AUTOCORRELATION BREADTH"
    )

    print(
        breadth_summary.to_string(
            index=False
        )
    )

    print("\n")
    print(
        "STRONGEST PEAK DATES"
    )

    print(
        peaks[
            [
                "significant_stocks",
                "positive_stocks",
                "negative_stocks",
                "pct_stocks_significant",
                "breadth_regime",
            ]
        ]
        .head(top_n)
        .to_string()
    )

    print("\n")
    print(
        "PEAK-DATE PARTICIPANTS"
    )

    for date in peak_dates[:top_n]:

        date_participants = (
            participants_with_clusters.loc[
                participants_with_clusters[
                    "date"
                ] == date
            ]
        )

        if date_participants.empty:
            continue

        print(
            f"\n{date.date()}:"
        )

        for _, row in (
            date_participants.iterrows()
        ):

            cluster_text = ""

            if (
                "cluster"
                in row.index
                and pd.notna(
                    row["cluster"]
                )
            ):

                cluster_text = (
                    f" | Cluster "
                    f"{int(row['cluster'])}"
                )

            print(
                f"  {row['ticker']}: "
                f"{row['autocorrelation']:.3f}"
                f" ({row['direction']})"
                f"{cluster_text}"
            )

    print("\n")
    print(
        "STOCK EXPOSURE TO BROAD EPISODES"
    )

    print(
        stock_exposure[
            [
                "ticker",
                "significant_dates",
                "broad_episode_dates",
                "pct_significant_during_broad",
                "pct_significant_during_isolated",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    if not cluster_summary.empty:

        print("\n")
        print(
            "PEAK-DATE PARTICIPATION "
            "BY CLUSTER"
        )

        print(
            cluster_summary.head(30)
            .to_string(
                index=False
            )
        )

    return {
        "daily_counts": counts,
        "breadth": breadth,
        "breadth_summary":
            breadth_summary,
        "peak_dates": peaks,
        "participants":
            participants_with_clusters,
        "cluster_summary":
            cluster_summary,
        "stock_exposure":
            stock_exposure,
        "overlap_matrix":
            overlap,
        "directional_overlap_matrix":
            directional,
    }