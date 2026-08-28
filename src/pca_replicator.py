import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import RESULTS_DIR
from .stock_replicator import (
    SUPERMAJORS,
    chronological_split,
    calculate_metrics,
)


DEFAULT_COMPONENT_VARIANCE = 0.90
DEFAULT_CANDIDATE_POOL_SIZE = 15

MIN_STOCKS = 2
MAX_STOCKS = 4


def eligible_candidates(
    returns,
    target,
):
    """
    Exclude the target and all five supermajors.
    """

    return [
        ticker
        for ticker in returns.columns
        if ticker != target
        and ticker not in SUPERMAJORS
    ]


def fit_pca_profile(
    train_returns,
    target,
    variance_threshold=DEFAULT_COMPONENT_VARIANCE,
):
    """
    Fit PCA using only eligible stocks from the training
    period.

    The target stock is then projected onto the same
    PCA factor space.
    """

    candidates = eligible_candidates(
        train_returns,
        target,
    )

    data = train_returns[
        [target] + candidates
    ].dropna()

    if data.empty:
        raise ValueError(
            f"No PCA training data available for {target}."
        )

    X = data[candidates]
    y = data[target]

    stock_scaler = StandardScaler()

    X_scaled = stock_scaler.fit_transform(
        X
    )

    target_scaler = StandardScaler()

    y_scaled = target_scaler.fit_transform(
        y.to_numpy().reshape(-1, 1)
    ).ravel()

    pca = PCA(
        n_components=variance_threshold
    )

    factor_scores = pca.fit_transform(
        X_scaled
    )

    # Target factor exposure
    target_exposure, _, _, _ = (
        np.linalg.lstsq(
            factor_scores,
            y_scaled,
            rcond=None,
        )
    )

    # Each stock's exposure to the factors.
    # For standardized PCA:
    # X_scaled ≈ scores @ components_
    stock_exposures = pd.DataFrame(
        pca.components_.T,
        index=candidates,
        columns=[
            f"PC{i + 1}"
            for i in range(
                pca.n_components_
            )
        ],
    )

    target_exposure = pd.Series(
        target_exposure,
        index=stock_exposures.columns,
        name=target,
    )

    explained = pd.DataFrame(
        {
            "component": [
                f"PC{i + 1}"
                for i in range(
                    pca.n_components_
                )
            ],
            "explained_variance":
                pca.explained_variance_ratio_,
        }
    )

    explained["cumulative_variance"] = (
        explained[
            "explained_variance"
        ].cumsum()
    )

    explained.to_csv(
        RESULTS_DIR
        / f"{target.lower()}_pca_explained_variance.csv",
        index=False,
    )

    target_exposure.to_frame(
        "target_exposure"
    ).to_csv(
        RESULTS_DIR
        / f"{target.lower()}_pca_target_exposure.csv"
    )

    stock_exposures.to_csv(
        RESULTS_DIR
        / f"{target.lower()}_pca_stock_exposures.csv"
    )

    return (
        pca,
        factor_scores,
        stock_exposures,
        target_exposure,
        explained,
    )


def rank_pca_candidates(
    stock_exposures,
    target_exposure,
    pool_size=DEFAULT_CANDIDATE_POOL_SIZE,
):
    """
    Rank stocks according to how close their PCA factor
    exposure is to the target's factor exposure.
    """

    distances = {}

    target_vector = (
        target_exposure.to_numpy()
    )

    target_norm = (
        np.linalg.norm(
            target_vector
        )
    )

    for ticker in stock_exposures.index:

        vector = (
            stock_exposures.loc[
                ticker
            ].to_numpy()
        )

        distance = np.linalg.norm(
            vector
            - target_vector
        )

        # Also use cosine similarity.
        norm = np.linalg.norm(
            vector
        )

        if (
            norm > 0
            and target_norm > 0
        ):

            cosine_similarity = (
                np.dot(
                    vector,
                    target_vector,
                )
                / (
                    norm
                    * target_norm
                )
            )

        else:
            cosine_similarity = 0.0

        distances[ticker] = {
            "factor_distance": distance,
            "cosine_similarity":
                cosine_similarity,
        }

    ranking = (
        pd.DataFrame(distances)
        .T
    )

    ranking = ranking.sort_values(
        [
            "factor_distance",
            "cosine_similarity",
        ],
        ascending=[
            True,
            False,
        ],
    )

    pool = ranking.head(
        pool_size
    ).index.tolist()

    ranking.to_csv(
        RESULTS_DIR
        / "pca_candidate_ranking.csv"
    )

    return pool, ranking


def optimize_factor_weights(
    stock_exposures,
    target_exposure,
    selected_stocks,
):
    """
    Find long-only weights summing to 1 that make the
    selected stocks' weighted PCA factor exposures as
    close as possible to the target's exposure.
    """

    selected_stocks = list(
        selected_stocks
    )

    B = (
        stock_exposures.loc[
            selected_stocks
        ].to_numpy().T
    )

    target = (
        target_exposure
        .to_numpy()
    )

    n = len(selected_stocks)

    initial = np.ones(n) / n

    def objective(weights):
        difference = (
            B @ weights
            - target
        )

        return np.sum(
            difference ** 2
        )

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights:
                np.sum(weights) - 1,
        }
    ]

    bounds = [
        (0.0, 1.0)
        for _ in range(n)
    ]

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "PCA factor-weight optimization failed: "
            + result.message
        )

    return pd.Series(
        result.x,
        index=selected_stocks,
        name="weight",
    )


def evaluate_pca_basket(
    train,
    validation,
    test,
    target,
    selected_stocks,
    stock_exposures,
    target_exposure,
):
    """
    Weights are determined entirely from PCA factor
    exposure on the training data.

    Validation is then used to select baskets.
    Test remains untouched.
    """

    weights = optimize_factor_weights(
        stock_exposures,
        target_exposure,
        selected_stocks,
    )

    validation_data = validation[
        [target] + list(selected_stocks)
    ].dropna()

    validation_portfolio = (
        validation_data[
            list(selected_stocks)
        ]
        .mul(
            weights,
            axis=1,
        )
        .sum(axis=1)
    )

    validation_metrics = calculate_metrics(
        validation_data[target],
        validation_portfolio,
    )

    test_data = test[
        [target] + list(selected_stocks)
    ].dropna()

    test_portfolio = (
        test_data[
            list(selected_stocks)
        ]
        .mul(
            weights,
            axis=1,
        )
        .sum(axis=1)
    )

    test_metrics = calculate_metrics(
        test_data[target],
        test_portfolio,
    )

    return (
        weights,
        validation_metrics,
        test_metrics,
    )


def run_pca_replication(
    returns,
    target,
    candidate_pool_size=DEFAULT_CANDIDATE_POOL_SIZE,
):
    """
    Run the PCA-based replication experiment for
    an arbitrary target stock.
    """

    train, validation, test = (
        chronological_split(
            returns
        )
    )

    print("\n")
    print("=" * 70)
    print(
        f"{target} PCA REPLICATION"
    )
    print("=" * 70)

    print(
        f"Training: "
        f"{train.index.min().date()} "
        f"to {train.index.max().date()}"
    )

    print(
        f"Validation: "
        f"{validation.index.min().date()} "
        f"to {validation.index.max().date()}"
    )

    print(
        f"Test: "
        f"{test.index.min().date()} "
        f"to {test.index.max().date()}"
    )

    (
        pca,
        factor_scores,
        stock_exposures,
        target_exposure,
        explained,
    ) = fit_pca_profile(
        train,
        target,
    )

    print(
        f"\nPCA components used: "
        f"{pca.n_components_}"
    )

    print(
        f"Variance explained: "
        f"{explained['cumulative_variance'].iloc[-1]:.2%}"
    )

    candidate_pool, ranking = (
        rank_pca_candidates(
            stock_exposures,
            target_exposure,
            candidate_pool_size,
        )
    )

    print(
        "\nPCA candidate pool:"
    )

    print(
        ", ".join(candidate_pool)
    )

    all_rows = []
    all_weights = []
    best_rows = []

    for k in range(
        MIN_STOCKS,
        MAX_STOCKS + 1,
    ):

        combinations = list(
            itertools.combinations(
                candidate_pool,
                k,
            )
        )

        print(
            f"\nSearching PCA "
            f"{target} {k}-stock baskets "
            f"({len(combinations)} combinations)..."
        )

        for combination in combinations:

            (
                weights,
                validation_metrics,
                test_metrics,
            ) = evaluate_pca_basket(
                train,
                validation,
                test,
                target,
                combination,
                stock_exposures,
                target_exposure,
            )

            stocks_string = "|".join(
                combination
            )

            row = {
                "target": target,
                "n_stocks": k,
                "stocks": stocks_string,

                "validation_correlation":
                    validation_metrics[
                        "correlation"
                    ],

                "validation_r_squared":
                    validation_metrics[
                        "r_squared"
                    ],

                "validation_tracking_error":
                    validation_metrics[
                        "tracking_error"
                    ],

                "test_correlation":
                    test_metrics[
                        "correlation"
                    ],

                "test_r_squared":
                    test_metrics[
                        "r_squared"
                    ],

                "test_tracking_error":
                    test_metrics[
                        "tracking_error"
                    ],

                "test_rmse":
                    test_metrics[
                        "rmse"
                    ],

                "test_target_volatility":
                    test_metrics[
                        "annualized_target_volatility"
                    ],

                "test_basket_volatility":
                    test_metrics[
                        "annualized_basket_volatility"
                    ],

                "test_volatility_reduction":
                    test_metrics[
                        "volatility_reduction"
                    ],

                "test_target_final_wealth":
                    test_metrics[
                        "target_final_wealth"
                    ],

                "test_basket_final_wealth":
                    test_metrics[
                        "basket_final_wealth"
                    ],

                "test_target_max_drawdown":
                    test_metrics[
                        "target_max_drawdown"
                    ],

                "test_basket_max_drawdown":
                    test_metrics[
                        "basket_max_drawdown"
                    ],
            }

            all_rows.append(row)

            for ticker in combination:

                all_weights.append(
                    {
                        "target": target,
                        "n_stocks": k,
                        "stocks":
                            stocks_string,
                        "ticker": ticker,
                        "weight":
                            weights[ticker],
                    }
                )

        k_results = [
            row
            for row in all_rows
            if row["n_stocks"] == k
        ]

        best = min(
            k_results,
            key=lambda row:
                row[
                    "validation_tracking_error"
                ],
        )

        best_rows.append(best)

    all_results = pd.DataFrame(
        all_rows
    )

    all_weights = pd.DataFrame(
        all_weights
    )

    best_by_k = pd.DataFrame(
        best_rows
    )

    all_results.to_csv(
        RESULTS_DIR
        / f"{target.lower()}_pca_all_baskets.csv",
        index=False,
    )

    all_weights.to_csv(
        RESULTS_DIR
        / f"{target.lower()}_pca_all_weights.csv",
        index=False,
    )

    best_by_k.to_csv(
        RESULTS_DIR
        / f"{target.lower()}_pca_best_by_k.csv",
        index=False,
    )

    print("\n")
    print("=" * 70)
    print(
        f"BEST {target} PCA BASKET BY SIZE"
    )
    print("=" * 70)

    for _, row in (
        best_by_k.iterrows()
    ):

        print(
            f"\n{int(row['n_stocks'])} stocks:"
        )

        print(
            f"  {row['stocks']}"
        )

        print(
            f"  Test correlation: "
            f"{row['test_correlation']:.4f}"
        )

        print(
            f"  Test R-squared: "
            f"{row['test_r_squared']:.4f}"
        )

        print(
            f"  Test tracking error: "
            f"{row['test_tracking_error']:.6f}"
        )

        print(
            f"  Test volatility: "
            f"{row['test_basket_volatility']:.2%}"
        )

        print(
            f"  Volatility reduction: "
            f"{row['test_volatility_reduction']:.2%}"
        )

    overall = best_by_k.loc[
        best_by_k[
            "validation_tracking_error"
        ].idxmin()
    ]

    selected_stocks = tuple(
        overall["stocks"].split("|")
    )

    weights = all_weights[
        all_weights["stocks"]
        == overall["stocks"]
    ].set_index(
        "ticker"
    )["weight"]

    print("\n")
    print("=" * 70)
    print(
        f"BEST OVERALL {target} "
        "PCA REPLICATION"
    )
    print("=" * 70)

    for ticker in selected_stocks:

        print(
            f"{ticker}: "
            f"{weights[ticker]:.2%}"
        )

    print(
        f"\nTest correlation: "
        f"{overall['test_correlation']:.4f}"
    )

    print(
        f"Test R-squared: "
        f"{overall['test_r_squared']:.4f}"
    )

    print(
        f"Test tracking error: "
        f"{overall['test_tracking_error']:.6f}"
    )

    print(
        f"Target volatility: "
        f"{overall['test_target_volatility']:.2%}"
    )

    print(
        f"Basket volatility: "
        f"{overall['test_basket_volatility']:.2%}"
    )

    print(
        f"Volatility reduction: "
        f"{overall['test_volatility_reduction']:.2%}"
    )

    return {
        "target": target,
        "candidate_pool": candidate_pool,
        "ranking": ranking,
        "all_results": all_results,
        "all_weights": all_weights,
        "best_by_k": best_by_k,
        "overall": overall,
        "weights": weights,
        "pca": pca,
        "explained": explained,
    }