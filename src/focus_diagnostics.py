# src/focus_diagnostics.py

from pathlib import Path

import pandas as pd


FOCUS_STOCKS = [
    "XOM",
    "CVX",
    "OXY",
    "TS",
    "TTE",
    "FTI",
    "CCJ",
]


def build_focus_cluster_table(
    historical_clusters,
    recent_cluster_results,
    output_dir,
):
    """
    Compare focus-stock cluster membership across:
        - historical automatic clustering
        - recent 252-day clustering for selected k values

    Parameters
    ----------
    historical_clusters : pd.DataFrame
        Must contain columns:
            ticker
            cluster

    recent_cluster_results : dict
        Mapping:
            k -> DataFrame
        Each DataFrame must contain:
            ticker
            cluster

    output_dir : Path
        Results directory.

    Returns
    -------
    pd.DataFrame
    """

    output_dir = Path(output_dir)

    historical = (
        historical_clusters[
            historical_clusters["ticker"].isin(FOCUS_STOCKS)
        ][["ticker", "cluster"]]
        .rename(columns={"cluster": "historical_cluster"})
        .copy()
    )

    historical = historical.sort_values("ticker")

    result = historical.set_index("ticker")

    for k, clusters in sorted(
        recent_cluster_results.items()
    ):
        subset = (
            clusters[
                clusters["ticker"].isin(FOCUS_STOCKS)
            ][["ticker", "cluster"]]
            .rename(
                columns={
                    "cluster":
                        f"recent_k{k}_cluster"
                }
            )
            .set_index("ticker")
        )

        result = result.join(
            subset,
            how="left"
        )

    result = result.reset_index()

    result.to_csv(
        output_dir /
        "focus_stock_cluster_diagnostics.csv",
        index=False,
    )

    return result


def build_focus_autocorrelation_table(
    regime_summary,
    episode_exposure,
    output_dir,
):
    """
    Combine stock-level autocorrelation regime statistics
    with exposure to broad autocorrelation episodes.
    """

    output_dir = Path(output_dir)

    regime = regime_summary[
        regime_summary["ticker"].isin(FOCUS_STOCKS)
    ].copy()

    exposure = episode_exposure[
        episode_exposure["ticker"].isin(FOCUS_STOCKS)
    ].copy()

    columns = [
        "ticker",
        "pct_strong_positive",
        "pct_strong_negative",
        "pct_everything_else",
    ]

    regime = regime[
        [
            c for c in columns
            if c in regime.columns
        ]
    ]

    merged = regime.merge(
        exposure,
        on="ticker",
        how="left",
        suffixes=(
            "",
            "_episode"
        ),
    )

    merged = merged.sort_values("ticker")

    merged.to_csv(
        output_dir /
        "focus_stock_autocorrelation_diagnostics.csv",
        index=False,
    )

    return merged


def print_focus_diagnostics(
    cluster_table,
    autocorrelation_table,
):
    """
    Print a compact research-oriented summary
    for the focus stocks.
    """

    print("\n")
    print("=" * 70)
    print("FOCUS STOCK DIAGNOSTICS")
    print("=" * 70)

    print("\nCluster membership:")

    print(
        cluster_table.to_string(
            index=False
        )
    )

    print("\nAutocorrelation behaviour:")

    print(
        autocorrelation_table.to_string(
            index=False
        )
    )

    print("\nInterpretation:")

    for _, row in autocorrelation_table.iterrows():

        ticker = row["ticker"]

        positive = row.get(
            "pct_strong_positive",
            float("nan")
        )

        negative = row.get(
            "pct_strong_negative",
            float("nan")
        )

        broad = row.get(
            "pct_significant_during_broad",
            float("nan")
        )

        print(
            f"{ticker}: "
            f"positive-regime={positive:.1%}, "
            f"negative-regime={negative:.1%}, "
            f"broad-episode participation="
            f"{broad:.1%}"
        )