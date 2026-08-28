import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from .config import RESULTS_DIR


SUPERMAJORS = {
    "XOM",
    "CVX",
    "SHEL",
    "BP",
    "TTE",
}

DEFAULT_TEST_FRACTION = 0.20
DEFAULT_VALIDATION_FRACTION = 0.20

DEFAULT_CANDIDATE_POOL_SIZE = 15

MIN_STOCKS = 2
MAX_STOCKS = 4


def chronological_split(
    returns,
    validation_fraction=DEFAULT_VALIDATION_FRACTION,
    test_fraction=DEFAULT_TEST_FRACTION,
):
    """
    Split the time series chronologically into:

        training
        validation
        test

    No random shuffling is used.
    """

    if (
        validation_fraction <= 0
        or test_fraction <= 0
        or validation_fraction + test_fraction >= 1
    ):
        raise ValueError(
            "Validation and test fractions must be "
            "positive and sum to less than 1."
        )

    n = len(returns)

    train_end = int(
        n * (
            1
            - validation_fraction
            - test_fraction
        )
    )

    validation_end = int(
        n * (
            1
            - test_fraction
        )
    )

    train = returns.iloc[
        :train_end
    ].copy()

    validation = returns.iloc[
        train_end:validation_end
    ].copy()

    test = returns.iloc[
        validation_end:
    ].copy()

    return train, validation, test


def eligible_candidates(
    returns,
    target,
):
    """
    Return candidate stocks excluding the target
    and all five supermajors.
    """

    candidates = []

    for ticker in returns.columns:

        if ticker == target:
            continue

        if ticker in SUPERMAJORS:
            continue

        candidates.append(ticker)

    return candidates


def rank_candidates(
    train_returns,
    target,
    pool_size=DEFAULT_CANDIDATE_POOL_SIZE,
):
    """
    Rank eligible securities using:

        1. positive Lasso score
        2. absolute correlation with target

    The top candidates form a manageable pool from which
    all 2-, 3- and 4-stock combinations are tested.
    """

    candidates = eligible_candidates(
        train_returns,
        target,
    )

    if len(candidates) < pool_size:
        pool_size = len(candidates)

    data = train_returns[
        [target] + candidates
    ].dropna()

    if data.empty:
        raise ValueError(
            "No overlapping training observations "
            f"for {target}."
        )

    X = data[candidates]
    y = data[target]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(
        alpha=0.001,
        positive=True,
        max_iter=100000,
    )

    lasso.fit(
        X_scaled,
        y,
    )

    lasso_scores = pd.Series(
        np.abs(lasso.coef_),
        index=candidates,
    )

    correlations = (
        data[candidates]
        .corrwith(y)
        .abs()
    )

    ranking = pd.DataFrame(
        {
            "lasso_score": lasso_scores,
            "abs_correlation": correlations,
        }
    )

    ranking["lasso_rank"] = (
        ranking["lasso_score"]
        .rank(
            ascending=False,
            method="average",
        )
    )

    ranking["correlation_rank"] = (
        ranking["abs_correlation"]
        .rank(
            ascending=False,
            method="average",
        )
    )

    ranking["combined_rank"] = (
        ranking["lasso_rank"]
        + ranking["correlation_rank"]
    )

    ranking = ranking.sort_values(
        "combined_rank"
    )

    pool = ranking.head(
        pool_size
    ).index.tolist()

    ranking.to_csv(
        RESULTS_DIR
        / f"{target.lower()}_candidate_ranking.csv"
    )

    return pool, ranking


def optimize_weights(
    returns,
    target,
    selected_stocks,
):
    """
    Find long-only weights that sum to 100% and
    minimize squared tracking error on the supplied
    training data.
    """

    selected_stocks = list(
        selected_stocks
    )

    data = returns[
        [target] + selected_stocks
    ].dropna()

    if data.empty:
        raise ValueError(
            "No overlapping observations for "
            f"{target} and {selected_stocks}."
        )

    y = data[target].to_numpy(
        dtype=float
    )

    X = data[selected_stocks].to_numpy(
        dtype=float
    )

    n = len(selected_stocks)

    initial = np.ones(n) / n

    def objective(weights):
        portfolio = X @ weights

        error = (
            y
            - portfolio
        )

        return np.sum(
            error ** 2
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
            "Weight optimization failed: "
            + result.message
        )

    return pd.Series(
        result.x,
        index=selected_stocks,
        name="weight",
    )


def calculate_metrics(
    actual,
    portfolio,
):
    """
    Calculate replication and risk measures.
    """

    data = pd.concat(
        [
            actual.rename("actual"),
            portfolio.rename("portfolio"),
        ],
        axis=1,
    ).dropna()

    actual_values = data[
        "actual"
    ]

    portfolio_values = data[
        "portfolio"
    ]

    error = (
        actual_values
        - portfolio_values
    )

    mse = np.mean(
        error ** 2
    )

    rmse = np.sqrt(mse)

    tracking_error = (
        error.std()
    )

    correlation = (
        actual_values.corr(
            portfolio_values
        )
    )

    ss_res = np.sum(
        error ** 2
    )

    ss_tot = np.sum(
        (
            actual_values
            - actual_values.mean()
        ) ** 2
    )

    r_squared = (
        1
        - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    actual_volatility = (
        actual_values.std()
        * np.sqrt(252)
    )

    portfolio_volatility = (
        portfolio_values.std()
        * np.sqrt(252)
    )

    volatility_reduction = (
        1
        - portfolio_volatility
        / actual_volatility
    )

    actual_wealth = (
        100
        * np.exp(
            actual_values.cumsum()
        )
    )

    portfolio_wealth = (
        100
        * np.exp(
            portfolio_values.cumsum()
        )
    )

    actual_drawdown = (
        actual_wealth
        / actual_wealth.cummax()
        - 1
    )

    portfolio_drawdown = (
        portfolio_wealth
        / portfolio_wealth.cummax()
        - 1
    )

    return {
        "correlation": correlation,
        "r_squared": r_squared,
        "rmse": rmse,
        "tracking_error": tracking_error,
        "annualized_target_volatility":
            actual_volatility,
        "annualized_basket_volatility":
            portfolio_volatility,
        "volatility_reduction":
            volatility_reduction,
        "target_final_wealth":
            actual_wealth.iloc[-1],
        "basket_final_wealth":
            portfolio_wealth.iloc[-1],
        "target_max_drawdown":
            actual_drawdown.min(),
        "basket_max_drawdown":
            portfolio_drawdown.min(),
    }


def evaluate_basket(
    train,
    validation,
    test,
    target,
    selected_stocks,
):
    """
    Estimate weights using training data.

    Validation data is used only to choose between
    candidate baskets.

    Test data remains completely untouched until the
    final evaluation.
    """

    weights = optimize_weights(
        train,
        target,
        selected_stocks,
    )

    # Validation
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

    # Test
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


def search_baskets(
    train,
    validation,
    test,
    target,
    candidate_pool,
    min_stocks=MIN_STOCKS,
    max_stocks=MAX_STOCKS,
):
    """
    Search every combination within the candidate pool.

    The winning basket for each k is chosen using
    VALIDATION tracking error, never test data.
    """

    all_rows = []
    all_weights = []

    best_rows = []

    for k in range(
        min_stocks,
        max_stocks + 1,
    ):

        combinations = list(
            itertools.combinations(
                candidate_pool,
                k,
            )
        )

        print(
            f"\nSearching "
            f"{target} {k}-stock baskets "
            f"({len(combinations)} combinations)..."
        )

        for combination in combinations:

            (
                weights,
                validation_metrics,
                test_metrics,
            ) = evaluate_basket(
                train,
                validation,
                test,
                target,
                combination,
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

        # --------------------------------------------------
        # Select winner using VALIDATION tracking error
        # --------------------------------------------------

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
        / f"{target.lower()}_replication_all_baskets.csv",
        index=False,
    )

    all_weights.to_csv(
        RESULTS_DIR
        / f"{target.lower()}_replication_all_weights.csv",
        index=False,
    )

    best_by_k.to_csv(
        RESULTS_DIR
        / f"{target.lower()}_replication_best_by_k.csv",
        index=False,
    )

    return (
        all_results,
        all_weights,
        best_by_k,
    )


def save_comparison_plot(
    test_returns,
    target,
    stocks,
    weights,
):
    """
    Compare $100 invested in target versus
    the replication basket on the test period.
    """

    stocks = list(stocks)

    data = test_returns[
        [target] + stocks
    ].dropna()

    basket_returns = (
        data[stocks]
        .mul(
            weights,
            axis=1,
        )
        .sum(axis=1)
    )

    target_wealth = (
        100
        * np.exp(
            data[target].cumsum()
        )
    )

    basket_wealth = (
        100
        * np.exp(
            basket_returns.cumsum()
        )
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    ax.plot(
        target_wealth.index,
        target_wealth,
        label=f"100% {target}",
        linewidth=2,
    )

    ax.plot(
        basket_wealth.index,
        basket_wealth,
        label="Replication basket",
        linewidth=2,
    )

    ax.set_title(
        f"{target} vs Replication Basket "
        "$100 Test-Period Comparison"
    )

    ax.set_ylabel(
        "Portfolio value ($)"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR
        / f"{target.lower()}_replication_comparison.png",
        dpi=200,
    )

    plt.close(fig)


def run_stock_replication(
    returns,
    target,
    candidate_pool_size=DEFAULT_CANDIDATE_POOL_SIZE,
):
    """
    Run the complete non-PCA replication analysis
    for an arbitrary target stock.

    XOM and CVX can both be targets.
    """

    train, validation, test = (
        chronological_split(
            returns
        )
    )

    print("\n")
    print("=" * 70)
    print(
        f"{target} CORRELATION/LASSO "
        "REPLICATION"
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

    candidate_pool, ranking = (
        rank_candidates(
            train,
            target,
            pool_size=candidate_pool_size,
        )
    )

    print(
        "\nCandidate pool:"
    )

    print(
        ", ".join(candidate_pool)
    )

    (
        all_results,
        all_weights,
        best_by_k,
    ) = search_baskets(
        train,
        validation,
        test,
        target,
        candidate_pool,
    )

    print("\n")
    print("=" * 70)
    print(
        f"BEST {target} BASKET BY SIZE"
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
            f"  Target volatility: "
            f"{row['test_target_volatility']:.2%}"
        )

        print(
            f"  Volatility reduction: "
            f"{row['test_volatility_reduction']:.2%}"
        )

    # Overall winner: smallest validation tracking error
    overall = best_by_k.loc[
        best_by_k[
            "validation_tracking_error"
        ].idxmin()
    ]

    selected_stocks = tuple(
        overall["stocks"].split("|")
    )

    target_weights = all_weights[
        all_weights["stocks"]
        == overall["stocks"]
    ].set_index(
        "ticker"
    )["weight"]

    save_comparison_plot(
        test,
        target,
        selected_stocks,
        target_weights,
    )

    print("\n")
    print("=" * 70)
    print(
        f"BEST OVERALL {target} "
        "REPLICATION"
    )
    print("=" * 70)

    for ticker in selected_stocks:

        print(
            f"{ticker}: "
            f"{target_weights[ticker]:.2%}"
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

    print(
        f"Target final $100: "
        f"${overall['test_target_final_wealth']:.2f}"
    )

    print(
        f"Basket final $100: "
        f"${overall['test_basket_final_wealth']:.2f}"
    )

    print(
        f"Target max drawdown: "
        f"{overall['test_target_max_drawdown']:.2%}"
    )

    print(
        f"Basket max drawdown: "
        f"{overall['test_basket_max_drawdown']:.2%}"
    )

    return {
        "target": target,
        "candidate_pool": candidate_pool,
        "ranking": ranking,
        "all_results": all_results,
        "all_weights": all_weights,
        "best_by_k": best_by_k,
        "overall": overall,
        "weights": target_weights,
    }