import numpy as np
import pandas as pd

from .config import RESULTS_DIR


def correlation_matrix(returns):
    """
    Pairwise return correlations.

    Pandas uses all overlapping observations
    available for each pair.
    """

    corr = returns.corr()

    corr.to_csv(
        RESULTS_DIR / "correlation_matrix.csv"
    )

    return corr


def pairwise_statistics(returns):
    """
    Generate summary statistics for every
    possible stock pair.
    """

    tickers = list(returns.columns)

    rows = []

    for i in range(len(tickers)):

        for j in range(i + 1, len(tickers)):

            a = tickers[i]
            b = tickers[j]

            pair = returns[[a, b]].dropna()

            if len(pair) < 30:
                continue

            correlation = pair[a].corr(pair[b])

            vol_a = pair[a].std()
            vol_b = pair[b].std()

            # Difference in volatility
            volatility_difference = abs(
                vol_a - vol_b
            )

            # How differently do the stocks move
            # on the same day?
            tracking_difference = (
                pair[a] - pair[b]
            ).std()

            rows.append(
                {
                    "stock_1": a,
                    "stock_2": b,
                    "observations": len(pair),
                    "correlation": correlation,
                    "volatility_1": vol_a,
                    "volatility_2": vol_b,
                    "volatility_difference":
                        volatility_difference,
                    "return_difference_std":
                        tracking_difference,
                }
            )

    results = pd.DataFrame(rows)

    results = results.sort_values(
        "correlation",
        ascending=False
    )

    results.to_csv(
        RESULTS_DIR / "pairwise_similarity.csv",
        index=False
    )

    return results


def print_requested_pairs(pair_stats):
    """
    Thomas specifically mentioned:
        XOM vs CVX
        COP vs VLO
    """

    requested = [
        ("XOM", "CVX"),
        ("COP", "VLO"),
    ]

    print("\n")
    print("=" * 70)
    print("THOMAS'S REQUESTED PAIRS")
    print("=" * 70)

    for a, b in requested:

        match = pair_stats[
            (
                (pair_stats["stock_1"] == a)
                &
                (pair_stats["stock_2"] == b)
            )
            |
            (
                (pair_stats["stock_1"] == b)
                &
                (pair_stats["stock_2"] == a)
            )
        ]

        if match.empty:
            print(
                f"\nNo result for {a}/{b}"
            )
            continue

        row = match.iloc[0]

        print(f"\n{a} vs {b}")
        print(
            f"Correlation: "
            f"{row['correlation']:.3f}"
        )
        print(
            f"Overlapping observations: "
            f"{int(row['observations'])}"
        )
        print(
            "Return-difference volatility: "
            f"{row['return_difference_std']:.4f}"
        )

def subset_correlation_matrix(
    returns,
    tickers,
    name,
):
    """
    Calculate and save a correlation matrix for a
    selected group of stocks.
    """

    available = [
        ticker
        for ticker in tickers
        if ticker in returns.columns
    ]

    data = returns[
        available
    ].dropna()

    matrix = data.corr()

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    matrix.to_csv(
        RESULTS_DIR
        / f"{safe_name}_correlation_matrix.csv"
    )

    print("\n")
    print("=" * 70)
    print(
        f"{name.upper()} CORRELATION MATRIX"
    )
    print("=" * 70)

    print(
        f"Period: "
        f"{data.index.min().date()} "
        f"to {data.index.max().date()}"
    )

    print(
        f"Observations: {len(data)}"
    )

    print("\n")

    print(
        matrix.round(3).to_string()
    )

    return matrix