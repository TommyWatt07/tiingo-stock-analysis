import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import (
    dendrogram,
    linkage,
)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import (
    MAX_CLUSTERS,
    MIN_CLUSTERS,
    RESULTS_DIR,
)


def prepare_features(common_returns):
    """
    Convert the return matrix into:

        one row per stock
        one feature per trading day

    Then standardize the features before clustering.
    """

    X = common_returns.T

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X, X_scaled


def choose_number_of_clusters(X_scaled):
    """
    Test every feasible number of clusters.

    With N stocks, silhouette scoring is feasible
    for k = 2 through N - 1.

    The selected k is the one with the highest
    silhouette score.
    """

    n_stocks = len(X_scaled)

    min_k = MIN_CLUSTERS

    if MAX_CLUSTERS is None:
        max_k = n_stocks - 1
    else:
        max_k = min(
            MAX_CLUSTERS,
            n_stocks - 1
        )

    if min_k > max_k:
        raise ValueError(
            "Not enough stocks to test clustering."
        )

    rows = []

    for k in range(
        min_k,
        max_k + 1
    ):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=100
        )

        labels = model.fit_predict(
            X_scaled
        )

        score = silhouette_score(
            X_scaled,
            labels
        )

        rows.append(
            {
                "k": k,
                "silhouette_score": score,
            }
        )

    scores = pd.DataFrame(rows)

    scores.to_csv(
        RESULTS_DIR /
        "cluster_silhouette_scores.csv",
        index=False
    )

    best_row = scores.loc[
        scores["silhouette_score"].idxmax()
    ]

    best_k = int(
        best_row["k"]
    )

    return best_k, scores


def run_kmeans(common_returns):
    """
    Automatically choose the best k and run K-means.
    """

    X, X_scaled = prepare_features(
        common_returns
    )

    best_k, scores = (
        choose_number_of_clusters(
            X_scaled
        )
    )

    print(
        f"\nBest cluster count "
        f"according to silhouette score: "
        f"{best_k}"
    )

    model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=100
    )

    labels = model.fit_predict(
        X_scaled
    )

    clusters = pd.DataFrame(
        {
            "ticker": X.index,
            "cluster": labels + 1,
        }
    )

    clusters = clusters.sort_values(
        ["cluster", "ticker"]
    )

    clusters.to_csv(
        RESULTS_DIR /
        "cluster_assignments.csv",
        index=False
    )

    return clusters, X_scaled, scores


def run_fixed_k(common_returns, k):
    """
    Run K-means using a specific user-requested k.

    This is useful for questions such as:
        "What happens if we force k = 8?"
    """

    X, X_scaled = prepare_features(
        common_returns
    )

    if k < 2:
        raise ValueError(
            "k must be at least 2."
        )

    if k >= len(X):
        raise ValueError(
            "k must be less than the number "
            "of stocks."
        )

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=100
    )

    labels = model.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    results = pd.DataFrame(
        {
            "ticker": X.index,
            "cluster": labels + 1,
        }
    )

    results = results.sort_values(
        ["cluster", "ticker"]
    )

    results.to_csv(
        RESULTS_DIR /
        f"clusters_k_{k}.csv",
        index=False
    )

    print("\n")
    print("=" * 70)
    print(
        f"FORCED K = {k} CLUSTER SOLUTION"
    )
    print("=" * 70)

    for cluster_id in sorted(
        results["cluster"].unique()
    ):

        members = results.loc[
            results["cluster"] == cluster_id,
            "ticker"
        ].tolist()

        print(
            f"Cluster {cluster_id}: "
            + ", ".join(members)
        )

    print(
        f"\nSilhouette score: "
        f"{score:.4f}"
    )

    return results, score


def force_five_cluster_analysis(
    common_returns
):
    """
    Run exactly five clusters.

    Kept because the five-cluster solution is
    useful for comparison with the original idea.
    """

    results, score = run_fixed_k(
        common_returns,
        k=5
    )

    print("\n")
    print("=" * 70)
    print("FIVE-CLUSTER COMPARISON")
    print("=" * 70)

    print(
        f"Silhouette score: "
        f"{score:.4f}"
    )

    return results, score


def run_pca(
    common_returns,
    X_scaled,
    clusters
):
    """
    Reduce the full return-history vectors to
    two dimensions for visualization.

    PCA is used for visualization, not as the
    clustering algorithm.
    """

    pca = PCA(
        n_components=2
    )

    coordinates = pca.fit_transform(
        X_scaled
    )

    pca_results = pd.DataFrame(
        coordinates,
        columns=[
            "PC1",
            "PC2"
        ]
    )

    pca_results["ticker"] = (
        common_returns.columns
    )

    pca_results = pca_results.merge(
        clusters,
        on="ticker"
    )

    pca_results.to_csv(
        RESULTS_DIR /
        "pca_coordinates.csv",
        index=False
    )

    print(
        "\nPCA explained variance:"
    )

    print(
        f"PC1: "
        f"{pca.explained_variance_ratio_[0]:.2%}"
    )

    print(
        f"PC2: "
        f"{pca.explained_variance_ratio_[1]:.2%}"
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.scatter(
        pca_results["PC1"],
        pca_results["PC2"],
        c=pca_results["cluster"],
        s=80
    )

    for _, row in (
        pca_results.iterrows()
    ):

        ax.annotate(
            row["ticker"],
            (
                row["PC1"],
                row["PC2"]
            ),
            xytext=(5, 4),
            textcoords="offset points"
        )

    ax.set_title(
        "PCA of Stock Return Behaviour"
    )

    ax.set_xlabel(
        "Principal Component 1"
    )

    ax.set_ylabel(
        "Principal Component 2"
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR /
        "pca_clusters.png",
        dpi=200
    )

    plt.close(fig)

    return pca_results, pca


def create_dendrogram(
    common_returns,
    X_scaled
):
    """
    Hierarchical clustering provides an
    independent visual cross-check.
    """

    linked = linkage(
        X_scaled,
        method="ward"
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    dendrogram(
        linked,
        labels=list(
            common_returns.columns
        ),
        leaf_rotation=90,
        ax=ax
    )

    ax.set_title(
        "Hierarchical Clustering of Stocks"
    )

    ax.set_ylabel(
        "Distance"
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR /
        "dendrogram.png",
        dpi=200
    )

    plt.close(fig)


def print_clusters(clusters):
    """
    Print cluster membership.
    """

    print("\n")
    print("=" * 70)
    print("STOCK CLUSTERS")
    print("=" * 70)

    for cluster_id in sorted(
        clusters["cluster"].unique()
    ):

        members = clusters.loc[
            clusters["cluster"] == cluster_id,
            "ticker"
        ].tolist()

        print(
            f"\nCluster {cluster_id}: "
            + ", ".join(members)
        )


def split_group_in_two(
    common_returns,
    tickers,
    group_name
):
    """
    Take a specified group and force it into
    exactly two subgroups.

    This can be used for:
        Oil
        Midstream
        Gas
        any other subset
    """

    available = [
        ticker
        for ticker in tickers
        if ticker in common_returns.columns
    ]

    if len(available) < 2:
        raise ValueError(
            f"Not enough securities available "
            f"for {group_name}."
        )

    group_returns = (
        common_returns[available]
    )

    X, X_scaled = prepare_features(
        group_returns
    )

    model = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=100
    )

    labels = model.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    results = pd.DataFrame(
        {
            "ticker": X.index,
            "subgroup": labels + 1,
        }
    )

    results = results.sort_values(
        [
            "subgroup",
            "ticker"
        ]
    )

    filename = (
        group_name.lower()
        .replace(" ", "_")
        + "_two_group_split.csv"
    )

    results.to_csv(
        RESULTS_DIR / filename,
        index=False
    )

    print("\n")
    print("=" * 70)
    print(
        f"{group_name.upper()} SPLIT INTO TWO"
    )
    print("=" * 70)

    for subgroup in sorted(
        results["subgroup"].unique()
    ):

        members = results.loc[
            results["subgroup"] == subgroup,
            "ticker"
        ].tolist()

        print(
            f"Subgroup {subgroup}: "
            + ", ".join(members)
        )

    print(
        f"\nSilhouette score: "
        f"{score:.4f}"
    )

    return results, score

def cluster_volatility_report(
    returns,
    clusters,
    annualization_factor=252,
):
    """
    Calculate annualized volatility for every stock
    and summarize volatility within each cluster.

    Volatility is:

        daily standard deviation * sqrt(252)
    """

    volatility = (
        returns.std()
        * np.sqrt(
            annualization_factor
        )
    )

    report = clusters.copy()

    report["annualized_volatility"] = (
        report["ticker"]
        .map(volatility)
    )

    report = report.sort_values(
        [
            "cluster",
            "annualized_volatility",
        ],
        ascending=[
            True,
            False,
        ],
    )

    report.to_csv(
        RESULTS_DIR
        / "cluster_stock_volatility.csv",
        index=False,
    )

    # Cluster-level summary
    summary = (
        report
        .groupby("cluster")
        ["annualized_volatility"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
                "std",
            ]
        )
        .reset_index()
    )

    summary.to_csv(
        RESULTS_DIR
        / "cluster_volatility_summary.csv",
        index=False,
    )

    print("\n")
    print("=" * 70)
    print(
        "STOCK VOLATILITY WITHIN CLUSTERS"
    )
    print("=" * 70)

    for cluster_id in sorted(
        report["cluster"].unique()
    ):

        members = report.loc[
            report["cluster"] == cluster_id
        ]

        print(
            f"\nCluster {cluster_id}"
        )

        for _, row in members.iterrows():

            print(
                f"  {row['ticker']}: "
                f"{row['annualized_volatility']:.2%}"
            )

    print("\n")
    print("=" * 70)
    print(
        "CLUSTER VOLATILITY SUMMARY"
    )
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    return report, summary

def run_subset_clustering(
    common_returns,
    tickers,
    group_name,
):
    """
    Run automatic K-means clustering on a specified
    subset of stocks.

    The optimal number of clusters is selected using
    the silhouette score rather than being forced.
    """

    available = [
        ticker
        for ticker in tickers
        if ticker in common_returns.columns
    ]

    missing = [
        ticker
        for ticker in tickers
        if ticker not in common_returns.columns
    ]

    if missing:
        print(
            f"\nWarning - missing from {group_name}: "
            + ", ".join(missing)
        )

    if len(available) < 3:
        raise ValueError(
            f"{group_name} needs at least "
            "3 available stocks."
        )

    subset_returns = common_returns[
        available
    ].dropna()

    X, X_scaled = prepare_features(
        subset_returns
    )

    n_stocks = len(X)

    rows = []

    # Test every valid k:
    # 2 through N - 1
    for k in range(
        2,
        n_stocks
    ):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=100,
        )

        labels = model.fit_predict(
            X_scaled
        )

        score = silhouette_score(
            X_scaled,
            labels
        )

        rows.append(
            {
                "k": k,
                "silhouette_score": score,
            }
        )

    scores = pd.DataFrame(rows)

    best_row = scores.loc[
        scores[
            "silhouette_score"
        ].idxmax()
    ]

    best_k = int(
        best_row["k"]
    )

    best_score = float(
        best_row["silhouette_score"]
    )

    # Fit final model
    model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=100,
    )

    labels = model.fit_predict(
        X_scaled
    )

    clusters = pd.DataFrame(
        {
            "ticker": X.index,
            "cluster": labels + 1,
        }
    )

    clusters = clusters.sort_values(
        [
            "cluster",
            "ticker",
        ]
    )

    safe_name = (
        group_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    clusters.to_csv(
        RESULTS_DIR
        / f"{safe_name}_clusters.csv",
        index=False,
    )

    scores.to_csv(
        RESULTS_DIR
        / f"{safe_name}_silhouette_scores.csv",
        index=False,
    )

    print("\n")
    print("=" * 70)
    print(
        f"{group_name.upper()} "
        "AUTOMATIC CLUSTERING"
    )
    print("=" * 70)

    print(
        f"Stocks: {n_stocks}"
    )

    print(
        f"Observations: "
        f"{len(subset_returns)}"
    )

    print(
        f"Optimal k: {best_k}"
    )

    print(
        f"Best silhouette score: "
        f"{best_score:.4f}"
    )

    print(
        "\nSilhouette scores:"
    )

    print(
        scores.to_string(
            index=False
        )
    )

    print(
        "\nClusters:"
    )

    for cluster_id in sorted(
        clusters["cluster"].unique()
    ):

        members = clusters.loc[
            clusters["cluster"]
            == cluster_id,
            "ticker",
        ].tolist()

        print(
            f"\nCluster {cluster_id}: "
            + ", ".join(members)
        )

    return (
        clusters,
        scores,
        best_k,
    )

def compare_cluster_memberships(
    historical_clusters,
    recent_clusters,
):
    """
    Compare historical and recent cluster structures.

    Cluster labels themselves are arbitrary, so Cluster 1 in one
    run is not assumed to correspond to Cluster 1 in another run.

    Instead, every recent cluster is matched to the historical
    cluster with the highest Jaccard similarity:

        intersection / union

    Outputs:
        historical_recent_cluster_matches.csv
        historical_vs_recent_clusters.csv
    """

    # --------------------------------------------------
    # Build membership dictionaries
    # --------------------------------------------------

    historical_groups = {
        cluster_id: set(
            historical_clusters.loc[
                historical_clusters["cluster"]
                == cluster_id,
                "ticker",
            ]
        )
        for cluster_id in sorted(
            historical_clusters[
                "cluster"
            ].unique()
        )
    }

    recent_groups = {
        cluster_id: set(
            recent_clusters.loc[
                recent_clusters["cluster"]
                == cluster_id,
                "ticker",
            ]
        )
        for cluster_id in sorted(
            recent_clusters[
                "cluster"
            ].unique()
        )
    }

    # --------------------------------------------------
    # Compare every historical/recent cluster pair
    # --------------------------------------------------

    pair_rows = []

    for historical_id, historical_members in (
        historical_groups.items()
    ):

        for recent_id, recent_members in (
            recent_groups.items()
        ):

            intersection = (
                historical_members
                & recent_members
            )

            union = (
                historical_members
                | recent_members
            )

            jaccard = (
                len(intersection)
                / len(union)
                if union
                else 0.0
            )

            historical_retention = (
                len(intersection)
                / len(historical_members)
                if historical_members
                else 0.0
            )

            recent_purity = (
                len(intersection)
                / len(recent_members)
                if recent_members
                else 0.0
            )

            pair_rows.append(
                {
                    "historical_cluster":
                        historical_id,
                    "recent_cluster":
                        recent_id,
                    "historical_size":
                        len(historical_members),
                    "recent_size":
                        len(recent_members),
                    "shared_stocks":
                        len(intersection),
                    "jaccard_similarity":
                        jaccard,
                    "historical_retention":
                        historical_retention,
                    "recent_purity":
                        recent_purity,
                    "shared_tickers":
                        "|".join(
                            sorted(intersection)
                        ),
                }
            )

    pair_comparison = pd.DataFrame(
        pair_rows
    )

    pair_comparison.to_csv(
        RESULTS_DIR
        / "historical_recent_cluster_pairwise_overlap.csv",
        index=False,
    )

    # --------------------------------------------------
    # Match each recent cluster to its closest
    # historical cluster
    # --------------------------------------------------

    match_rows = []

    for recent_id in sorted(
        recent_groups
    ):

        candidates = (
            pair_comparison.loc[
                pair_comparison[
                    "recent_cluster"
                ] == recent_id
            ]
            .sort_values(
                [
                    "jaccard_similarity",
                    "shared_stocks",
                    "historical_retention",
                ],
                ascending=False,
            )
        )

        best = candidates.iloc[0]

        match_rows.append(
            best.to_dict()
        )

    matches = pd.DataFrame(
        match_rows
    )

    matches.to_csv(
        RESULTS_DIR
        / "historical_recent_cluster_matches.csv",
        index=False,
    )

    # --------------------------------------------------
    # Stock-level comparison
    # --------------------------------------------------

    historical_assignments = (
        historical_clusters[
            [
                "ticker",
                "cluster",
            ]
        ]
        .rename(
            columns={
                "cluster":
                    "historical_cluster"
            }
        )
    )

    recent_assignments = (
        recent_clusters[
            [
                "ticker",
                "cluster",
            ]
        ]
        .rename(
            columns={
                "cluster":
                    "recent_cluster"
            }
        )
    )

    stock_comparison = (
        historical_assignments
        .merge(
            recent_assignments,
            on="ticker",
            how="outer",
        )
    )

    match_lookup = (
        matches
        .set_index(
            "recent_cluster"
        )
    )

    stock_comparison[
        "matched_historical_cluster"
    ] = (
        stock_comparison[
            "recent_cluster"
        ]
        .map(
            match_lookup[
                "historical_cluster"
            ]
        )
    )

    stock_comparison[
        "cluster_jaccard_similarity"
    ] = (
        stock_comparison[
            "recent_cluster"
        ]
        .map(
            match_lookup[
                "jaccard_similarity"
            ]
        )
    )

    stock_comparison[
        "cluster_historical_retention"
    ] = (
        stock_comparison[
            "recent_cluster"
        ]
        .map(
            match_lookup[
                "historical_retention"
            ]
        )
    )

    stock_comparison[
        "cluster_recent_purity"
    ] = (
        stock_comparison[
            "recent_cluster"
        ]
        .map(
            match_lookup[
                "recent_purity"
            ]
        )
    )

    # A stock is considered retained if its recent cluster
    # is best matched to its own historical cluster.

    stock_comparison["retained_group"] = (
        stock_comparison[
            "historical_cluster"
        ]
        == stock_comparison[
            "matched_historical_cluster"
        ]
    )

    stock_comparison = (
        stock_comparison
        .sort_values(
            [
                "retained_group",
                "cluster_jaccard_similarity",
                "ticker",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    stock_comparison.to_csv(
        RESULTS_DIR
        / "historical_vs_recent_clusters.csv",
        index=False,
    )

    # --------------------------------------------------
    # Print concise report
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "HISTORICAL VS RECENT CLUSTER STABILITY"
    )
    print("=" * 70)

    print(
        "\nRECENT CLUSTER MATCHES:"
    )

    display_matches = matches[
        [
            "recent_cluster",
            "historical_cluster",
            "recent_size",
            "historical_size",
            "shared_stocks",
            "jaccard_similarity",
            "historical_retention",
            "recent_purity",
            "shared_tickers",
        ]
    ].copy()

    print(
        display_matches.to_string(
            index=False
        )
    )

    retained = (
        stock_comparison[
            "retained_group"
        ].sum()
    )

    changed = (
        (~stock_comparison[
            "retained_group"
        ]).sum()
    )

    print("\n")
    print(
        f"Stocks retained in matched "
        f"historical group: {retained}"
    )

    print(
        f"Stocks assigned to a different "
        f"matched group: {changed}"
    )

    # --------------------------------------------------
    # Print stocks that changed
    # --------------------------------------------------

    changed_stocks = (
        stock_comparison.loc[
            ~stock_comparison[
                "retained_group"
            ]
        ]
    )

    if not changed_stocks.empty:

        print("\n")
        print(
            "STOCKS WITH CHANGED GROUP MEMBERSHIP"
        )

        print(
            changed_stocks[
                [
                    "ticker",
                    "historical_cluster",
                    "recent_cluster",
                    "matched_historical_cluster",
                    "cluster_jaccard_similarity",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------
    # Key stocks Thomas has discussed
    # --------------------------------------------------

    focus_tickers = [
        "OXY",
        "TS",
        "TTE",
        "XOM",
        "CVX",
        "FTI",
        "CCJ",
    ]

    focus = (
        stock_comparison.loc[
            stock_comparison[
                "ticker"
            ].isin(
                focus_tickers
            )
        ]
        .sort_values(
            "ticker"
        )
    )

    print("\n")
    print(
        "FOCUS STOCKS"
    )

    print(
        focus[
            [
                "ticker",
                "historical_cluster",
                "recent_cluster",
                "matched_historical_cluster",
                "cluster_jaccard_similarity",
                "retained_group",
            ]
        ].to_string(
            index=False
        )
    )

    return {
        "cluster_matches": matches,
        "pairwise_overlap":
            pair_comparison,
        "stock_comparison":
            stock_comparison,
    }