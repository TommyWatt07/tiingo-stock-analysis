import argparse
import pandas as pd

from .autocorrelation_episodes import (
    print_autocorrelation_episode_report,
)

from .focus_diagnostics import (
    build_focus_autocorrelation_table,
    build_focus_cluster_table,
    print_focus_diagnostics,
)

from .autocorrelation import (
    autocorrelation_summary,
    autocorrelation_regime_series,
    compare_cluster_autocorrelation,
    full_sample_acf,
    plot_rolling_acf,
    rolling_autocorrelation,
    plot_autocorrelation_regime_timeline,
    plot_autocorrelation_regime_heatmap,
    plot_autocorrelation_breadth,
    plot_autocorrelation_breadth_lines,
)

from .clustering import (
    cluster_volatility_report,
    compare_cluster_memberships,
    create_dendrogram,
    print_clusters,
    run_fixed_k,
    run_kmeans,
    run_pca,
    run_subset_clustering,
)

from .config import RESULTS_DIR

from .pca_replicator import (
    run_pca_replication,
)

from .prepare import (
    build_price_matrix,
    build_return_matrix,
    create_history_summary,
    get_common_returns,
)

from .research_config import (
    CANADA_EXCLUDING_CCJ,
    CANADA_TICKERS,
    INTEGRATED_GROUP_TICKERS,
    OILFIELD_SERVICES_TS_TICKERS,
    OUTLIERS_TO_EXCLUDE,
    REST_OF_WORLD_EXCLUDING_CCJ,
    REST_OF_WORLD_TICKERS,
    ROLLING_CORRELATION_PAIRS,
    ROLLING_CORRELATION_WINDOW,
    RECENT_CLUSTER_WINDOW,
    SUPERMAJOR_TICKERS,
    TTE_TS_DIAGNOSTIC_PAIRS,
)

from .rolling_analysis import (
    compare_multiple_pairs,
)

from .similarity import (
    correlation_matrix,
    pairwise_statistics,
    print_requested_pairs,
    subset_correlation_matrix,
)

from .stock_replicator import (
    run_stock_replication,
)

from .tiingo import download_all


def main(force=False):

    print("=" * 70)
    print("TIINGO STOCK TIME-SERIES ANALYSIS")
    print("=" * 70)

    # ==================================================
    # 1. DOWNLOAD / LOAD DATA
    # ==================================================

    print("\nDownloading/loading Tiingo data...")

    histories = download_all(
        force=force
    )

    print(
        f"\nLoaded {len(histories)} stocks."
    )

    # ==================================================
    # 2. PREPARE PRICE AND RETURN DATA
    # ==================================================

    print("\nPreparing price matrix...")

    prices = build_price_matrix(
        histories
    )

    history_summary = (
        create_history_summary(
            prices
        )
    )

    print("\nAvailable histories:")

    print(
        history_summary.to_string(
            index=False
        )
    )

    returns = build_return_matrix(
        prices
    )

    common_returns = (
        get_common_returns(
            returns
        )
    )

    print(
        "\nCommon clustering period:"
    )

    print(
        common_returns.index.min(),
        "to",
        common_returns.index.max(),
    )

    print(
        "Observations:",
        len(common_returns),
    )

    # ==================================================
    # 3. PAIRWISE SIMILARITY
    # ==================================================

    print(
        "\nCalculating stock similarities..."
    )

    correlation_matrix(
        returns
    )

    pair_stats = (
        pairwise_statistics(
            returns
        )
    )

    print_requested_pairs(
        pair_stats
    )

    print("\n")
    print("=" * 70)
    print(
        "TOP 10 MOST CORRELATED PAIRS"
    )
    print("=" * 70)

    print(
        pair_stats[
            [
                "stock_1",
                "stock_2",
                "correlation",
                "observations",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # ==================================================
    # 4. CORRELATION MATRICES
    # ==================================================

    print("\n")
    print("=" * 70)
    print(
        "SUPERMAJOR CORRELATION MATRIX"
    )
    print("=" * 70)

    subset_correlation_matrix(
        returns,
        SUPERMAJOR_TICKERS,
        "Supermajors",
    )

    # --------------------------------------------------
    # International / integrated correlation matrix
    # --------------------------------------------------

    integrated_corr_tickers = [
        "SHEL",
        "BP",
        "TTE",
        "EQNR",
        "E",
        "TS",
    ]

    subset_correlation_matrix(
        returns,
        integrated_corr_tickers,
        "SHEL BP TTE EQNR E TS",
    )

    # ==================================================
    # 5. AUTOMATIC FULL-UNIVERSE CLUSTERING
    # ==================================================

    print(
        "\nRunning automatic clustering..."
    )

    (
        clusters,
        X_scaled,
        scores,
    ) = run_kmeans(
        common_returns
    )

    print_clusters(
        clusters
    )

    cluster_volatility_report(
        returns,
        clusters,
    )

    print("\nSilhouette scores:")

    print(
        scores.to_string(
            index=False
        )
    )

    # ==================================================
    # 6. PCA VISUALIZATION
    # ==================================================

    run_pca(
        common_returns,
        X_scaled,
        clusters,
    )

    # ==================================================
    # 7. HIERARCHICAL CLUSTERING
    # ==================================================

    create_dendrogram(
        common_returns,
        X_scaled,
    )

    # ==================================================
    # 8. RECENT 252-DAY CLUSTERING
    # ==================================================

    recent_252_returns = (
        common_returns
        .tail(RECENT_CLUSTER_WINDOW)
        .copy()
    )

    (
        recent_clusters,
        recent_scores,
        recent_k,
    ) = run_subset_clustering(
        recent_252_returns,
        recent_252_returns.columns.tolist(),
        "Recent 252 Days",
    )

    # ==================================================
    # 8A. RECENT 252-DAY FIXED-K CLUSTERING
    # ==================================================

    print("\n")
    print("=" * 70)
    print(
        "RECENT 252-DAY FIXED-K COMPARISON"
    )
    print("=" * 70)

    recent_fixed_clusters = {}
    recent_fixed_scores = {}

    for k in [5, 8, 10]:

        print("\n")
        print("=" * 70)
        print(f"RECENT 252 DAYS: K = {k}")
        print("=" * 70)

        fixed_clusters, fixed_score = run_fixed_k(
            recent_252_returns,
            k=k,
        )

        recent_fixed_clusters[k] = fixed_clusters
        recent_fixed_scores[k] = fixed_score

    focus_cluster_table = build_focus_cluster_table(
        historical_clusters=clusters,
        recent_cluster_results=recent_fixed_clusters,
        output_dir=RESULTS_DIR,
    )

    # ==================================================
    # FOCUS-STOCK CLUSTER DIAGNOSTICS
    # ==================================================

    focus_cluster_table = build_focus_cluster_table(
        historical_clusters=clusters,
        recent_cluster_results=recent_fixed_clusters,
        output_dir=RESULTS_DIR,
    )

    print("\n")
    print("=" * 70)
    print("FOCUS STOCK CLUSTER DIAGNOSTICS")
    print("=" * 70)

    print(
        focus_cluster_table.to_string(
            index=False
        )
    )

    # ==================================================
    # 8B. HISTORICAL VS RECENT CLUSTER COMPARISON
    # ==================================================

    print("\n")
    print("=" * 70)
    print(
        "HISTORICAL VS RECENT CLUSTER STABILITY"
    )
    print("=" * 70)

    cluster_comparison = (
        compare_cluster_memberships(
            clusters,
            recent_clusters,
        )
    )

    # ==================================================
    # 9. GEOGRAPHIC / SUBSET CLUSTERING
    # ==================================================

    run_subset_clustering(
        common_returns,
        CANADA_TICKERS,
        "Canada",
    )

    run_subset_clustering(
        common_returns,
        REST_OF_WORLD_TICKERS,
        "Rest of World",
    )

    run_subset_clustering(
        common_returns,
        CANADA_EXCLUDING_CCJ,
        "Canada Excluding CCJ",
    )

    run_subset_clustering(
        common_returns,
        REST_OF_WORLD_EXCLUDING_CCJ,
        "Rest of World Excluding CCJ",
    )

    # ==================================================
    # 10. FULL UNIVERSE EXCLUDING CCJ AND TPL
    # ==================================================

    full_universe_ex_outliers = [
        ticker
        for ticker in common_returns.columns
        if ticker not in OUTLIERS_TO_EXCLUDE
    ]

    run_subset_clustering(
        common_returns,
        full_universe_ex_outliers,
        "Full Universe Excluding CCJ and TPL",
    )

    # ==================================================
    # 11. FOCUSED DIAGNOSTIC CLUSTERS
    # ==================================================

    run_subset_clustering(
        common_returns,
        INTEGRATED_GROUP_TICKERS,
        "Integrated Group Diagnostic",
    )

    run_subset_clustering(
        common_returns,
        OILFIELD_SERVICES_TS_TICKERS,
        "Oilfield Services and TS Diagnostic",
    )

    # ==================================================
    # 12. AUTOCORRELATION
    # ==================================================

    print(
        "\nRunning autocorrelation analysis..."
    )

    # --------------------------------------------------
    # Full-sample autocorrelation
    # --------------------------------------------------

    full_sample_acf(
        returns
    )

    # --------------------------------------------------
    # Rolling autocorrelation
    # --------------------------------------------------

    rolling_acf = (
        rolling_autocorrelation(
            returns
        )
    )

    print("\nGenerating synchronised autocorrelation breadth plot...")

    regime_breadth = plot_autocorrelation_breadth_lines(
        rolling_acf,
        window=ROLLING_CORRELATION_WINDOW,
        top_n=8,
    )

    # --------------------------------------------------
    # Stock-level autocorrelation summary
    # --------------------------------------------------

    acf_summary = (
        autocorrelation_summary(
            rolling_acf
        )
    )

    print("\n")
    print("=" * 70)
    print(
        "AUTOCORRELATION SUMMARY"
    )
    print("=" * 70)

    print(
        acf_summary.head(10)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Rolling autocorrelation chart
    # --------------------------------------------------

    plot_rolling_acf(
        rolling_acf
    )

    # --------------------------------------------------
    # Cluster-level autocorrelation comparison
    # --------------------------------------------------

    compare_cluster_autocorrelation(
        rolling_acf,
        clusters
    )

    # --------------------------------------------------
    # Autocorrelation regime analysis
    # --------------------------------------------------

    print(
        "\nGenerating autocorrelation "
        "regime visualizations..."
    )

    regime_series = (
        autocorrelation_regime_series(
            rolling_acf
        )
    )

    plot_autocorrelation_regime_heatmap(
        regime_series
    )

    regime_breadth = (
        plot_autocorrelation_regime_timeline(
            rolling_acf,
            clusters=clusters,
        )
    )

    plot_autocorrelation_breadth(
        regime_breadth
    )

    print(
        "\nAutocorrelation regime breadth:"
    )

    print(
        regime_breadth[
            [
                "strong_positive_stocks",
                "strong_negative_stocks",
                "everything_else_stocks",
                "available_stocks",
                "pct_strong_positive",
                "pct_strong_negative",
                "pct_everything_else",
            ]
        ]
        .tail(10)
        .to_string()
    )

    # ==================================================
    # 13. AUTOCORRELATION EPISODES
    # ==================================================

    print_autocorrelation_episode_report(
        rolling_acf,
        clusters=clusters,
        window=ROLLING_CORRELATION_WINDOW,
        minimum_stocks=3,
        top_n=10,
    )

    # ==================================================
    # 14. ROLLING CORRELATIONS
    # ==================================================

    print("\n")
    print("=" * 70)
    print(
        "ROLLING CORRELATION ANALYSIS"
    )
    print("=" * 70)

    rolling_summary = (
        compare_multiple_pairs(
            returns,
            ROLLING_CORRELATION_PAIRS,
            window=ROLLING_CORRELATION_WINDOW,
        )
    )

    print(
        "\nRolling correlation summary:"
    )

    print(
        rolling_summary[
            [
                "ticker_1",
                "ticker_2",
                "full_period_correlation",
                "mean_rolling_correlation",
                "min_rolling_correlation",
                "max_rolling_correlation",
            ]
        ].to_string(
            index=False
        )
    )

    # ==================================================
    # 15. TTE / TS DIAGNOSTIC CORRELATIONS
    # ==================================================

    print("\n")
    print("=" * 70)
    print(
        "TTE / TS DIAGNOSTIC CORRELATIONS"
    )
    print("=" * 70)

    tte_ts_summary = (
        compare_multiple_pairs(
            returns,
            TTE_TS_DIAGNOSTIC_PAIRS,
            window=ROLLING_CORRELATION_WINDOW,
        )
    )

    print(
        "\nTTE / TS diagnostic "
        "correlation summary:"
    )

    print(
        tte_ts_summary[
            [
                "ticker_1",
                "ticker_2",
                "full_period_correlation",
                "mean_rolling_correlation",
                "min_rolling_correlation",
                "max_rolling_correlation",
            ]
        ].to_string(
            index=False
        )
    )

    # ==================================================
    # 16. XOM / CVX NON-PCA REPLICATION
    # ==================================================

    print("\n")
    print("=" * 70)
    print(
        "NON-PCA STOCK REPLICATION"
    )
    print("=" * 70)

    xom_replication = (
        run_stock_replication(
            common_returns,
            target="XOM",
        )
    )

    cvx_replication = (
        run_stock_replication(
            common_returns,
            target="CVX",
        )
    )

    # ==================================================
    # 17. XOM / CVX PCA REPLICATION
    # ==================================================

    print("\n")
    print("=" * 70)
    print(
        "PCA STOCK REPLICATION"
    )
    print("=" * 70)

    xom_pca = (
        run_pca_replication(
            common_returns,
            target="XOM",
        )
    )

    cvx_pca = (
        run_pca_replication(
            common_returns,
            target="CVX",
        )
    )

    # ==================================================
    # 18. REPLICATION METHOD COMPARISON
    # ==================================================

    comparison_rows = []

    analyses = [
        (
            "XOM",
            "Correlation/Lasso",
            xom_replication,
        ),
        (
            "CVX",
            "Correlation/Lasso",
            cvx_replication,
        ),
        (
            "XOM",
            "PCA",
            xom_pca,
        ),
        (
            "CVX",
            "PCA",
            cvx_pca,
        ),
    ]

    for target, method, result in analyses:

        overall = result["overall"]

        comparison_rows.append(
            {
                "target": target,
                "method": method,
                "stocks":
                    overall["stocks"],
                "test_correlation":
                    overall[
                        "test_correlation"
                    ],
                "test_r_squared":
                    overall[
                        "test_r_squared"
                    ],
                "test_tracking_error":
                    overall[
                        "test_tracking_error"
                    ],
                "target_volatility":
                    overall[
                        "test_target_volatility"
                    ],
                "basket_volatility":
                    overall[
                        "test_basket_volatility"
                    ],
                "volatility_reduction":
                    overall[
                        "test_volatility_reduction"
                    ],
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison.to_csv(
        RESULTS_DIR
        / "xom_vs_cvx_replication_comparison.csv",
        index=False,
    )

    print("\n")
    print("=" * 70)
    print(
        "XOM vs CVX / METHOD COMPARISON"
    )
    print("=" * 70)

    print(
        comparison.to_string(
            index=False
        )
    )

    # ==================================================
    # 19. COMPLETE
    # ==================================================

    print("\n")
    print("=" * 70)
    print(
        "ANALYSIS COMPLETE"
    )
    print("=" * 70)

    print(
        "\nCheck the results/ folder "
        "for CSV files and charts."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run Tiingo stock "
            "time-series analysis."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Redownload Tiingo data instead "
            "of using cached CSV files."
        ),
    )

    args = parser.parse_args()

    main(
        force=args.force
    )