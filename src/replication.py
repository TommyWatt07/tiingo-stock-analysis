import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from .config import RESULTS_DIR


def chronological_train_test_split(
    returns,
    test_fraction=0.20
):
    """
    Split the return data chronologically.

    No shuffling is used because this is time-series data.
    """

    if not 0 < test_fraction < 1:
        raise ValueError(
            "test_fraction must be between 0 and 1."
        )

    n = len(returns)

    split_index = int(
        n * (1 - test_fraction)
    )

    train = returns.iloc[:split_index].copy()
    test = returns.iloc[split_index:].copy()

    return train, test


def select_candidates_with_lasso(
    train_returns,
    target,
    n_stocks,
    alpha=0.001
):
    """
    Use positive Lasso on the training data to identify
    a small set of candidate stocks.

    Lasso is only used for stock selection.
    Final portfolio weights are estimated separately
    with a long-only sum-to-one constraint.
    """

    target_series = train_returns[target]

    candidates = [
        col
        for col in train_returns.columns
        if col != target
    ]

    X = train_returns[candidates].copy()
    y = target_series.copy()

    # Remove rows with missing data.
    data = pd.concat(
        [X, y.rename(target)],
        axis=1
    ).dropna()

    X = data[candidates]
    y = data[target]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = Lasso(
        alpha=alpha,
        positive=True,
        max_iter=100000
    )

    model.fit(
        X_scaled,
        y
    )

    coefficients = pd.Series(
        model.coef_,
        index=candidates
    )

    coefficients = coefficients[
        coefficients > 1e-10
    ]

    # If Lasso selects fewer stocks than requested,
    # use the highest absolute coefficients from all
    # candidates to fill the set.
    if len(coefficients) < n_stocks:

        all_coefficients = pd.Series(
            model.coef_,
            index=candidates
        )

        selected = (
            all_coefficients
            .abs()
            .sort_values(
                ascending=False
            )
            .head(n_stocks)
            .index
            .tolist()
        )

    else:

        selected = (
            coefficients
            .abs()
            .sort_values(
                ascending=False
            )
            .head(n_stocks)
            .index
            .tolist()
        )

    return selected


def optimize_long_only_weights(
    train_returns,
    target,
    selected_stocks
):
    """
    Estimate long-only weights for the selected stocks.

    Constraints:
        weight >= 0
        sum(weight) = 1

    Objective:
        minimize squared tracking error
    """

    data = train_returns[
        [target] + selected_stocks
    ].dropna()

    y = data[target].to_numpy(
        dtype=float
    )

    X = data[selected_stocks].to_numpy(
        dtype=float
    )

    n = len(selected_stocks)

    initial_weights = np.ones(n) / n

    def objective(weights):
        portfolio = X @ weights

        errors = y - portfolio

        return np.sum(
            errors ** 2
        )

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights:
                np.sum(weights) - 1
        }
    ]

    bounds = [
        (0.0, 1.0)
        for _ in range(n)
    ]

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-12,
        }
    )

    if not result.success:
        raise RuntimeError(
            "Portfolio optimization failed: "
            + result.message
        )

    weights = pd.Series(
        result.x,
        index=selected_stocks,
        name="weight"
    )

    return weights


def evaluate_replication(
    actual,
    replicated
):
    """
    Calculate portfolio replication statistics.
    """

    data = pd.concat(
        [
            actual.rename("actual"),
            replicated.rename("replicated"),
        ],
        axis=1
    ).dropna()

    actual_values = data["actual"]
    replicated_values = data["replicated"]

    errors = (
        actual_values
        - replicated_values
    )

    mse = np.mean(
        errors ** 2
    )

    rmse = np.sqrt(mse)

    mae = np.mean(
        np.abs(errors)
    )

    correlation = (
        actual_values.corr(
            replicated_values
        )
    )

    ss_res = np.sum(
        errors ** 2
    )

    ss_tot = np.sum(
        (
            actual_values
            - actual_values.mean()
        ) ** 2
    )

    r_squared = (
        1 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    tracking_error = errors.std()

    actual_volatility = (
        actual_values.std()
    )

    replicated_volatility = (
        replicated_values.std()
    )

    volatility_difference = abs(
        actual_volatility
        - replicated_volatility
    )

    return {
        "observations": len(data),
        "correlation": correlation,
        "r_squared": r_squared,
        "rmse": rmse,
        "mae": mae,
        "tracking_error": tracking_error,
        "actual_volatility":
            actual_volatility,
        "replicated_volatility":
            replicated_volatility,
        "volatility_difference":
            volatility_difference,
    }


def build_replication_portfolio(
    train_returns,
    target,
    n_stocks,
    alpha=0.001
):
    """
    Select stocks using Lasso and then estimate
    long-only sum-to-one portfolio weights.
    """

    selected = select_candidates_with_lasso(
        train_returns=train_returns,
        target=target,
        n_stocks=n_stocks,
        alpha=alpha
    )

    weights = optimize_long_only_weights(
        train_returns=train_returns,
        target=target,
        selected_stocks=selected
    )

    return selected, weights


def run_xom_replication(
    returns,
    target="XOM",
    max_stocks=5,
    test_fraction=0.20,
    alpha=0.001
):
    """
    Attempt to replicate the target stock using
    1 through max_stocks of the other securities.

    Stock selection and portfolio optimization are
    performed only on the training sample.

    Results are evaluated on both training and test
    samples.
    """

    if target not in returns.columns:
        raise ValueError(
            f"{target} is not in the return matrix."
        )

    train, test = (
        chronological_train_test_split(
            returns,
            test_fraction=test_fraction
        )
    )

    summary_rows = []
    weight_rows = []

    all_test_replicated = {}

    for k in range(
        1,
        max_stocks + 1
    ):

        print("\n")
        print("=" * 70)
        print(
            f"{target} REPLICATION "
            f"USING {k} STOCKS"
        )
        print("=" * 70)

        selected, weights = (
            build_replication_portfolio(
                train_returns=train,
                target=target,
                n_stocks=k,
                alpha=alpha
            )
        )

        print(
            "Selected stocks:"
        )

        for ticker in selected:

            print(
                f"  {ticker}: "
                f"{weights[ticker]:.4%}"
            )

        # --------------------------------------------------
        # Training portfolio
        # --------------------------------------------------

        train_data = train[
            [target] + selected
        ].dropna()

        train_replicated = (
            train_data[selected]
            .mul(
                weights,
                axis=1
            )
            .sum(axis=1)
        )

        train_metrics = (
            evaluate_replication(
                train_data[target],
                train_replicated
            )
        )

        # --------------------------------------------------
        # Test portfolio
        # --------------------------------------------------

        test_data = test[
            [target] + selected
        ].dropna()

        test_replicated = (
            test_data[selected]
            .mul(
                weights,
                axis=1
            )
            .sum(axis=1)
        )

        test_metrics = (
            evaluate_replication(
                test_data[target],
                test_replicated
            )
        )

        all_test_replicated[k] = (
            test_replicated
        )

        summary_rows.append(
            {
                "target": target,
                "n_stocks": k,

                "train_correlation":
                    train_metrics[
                        "correlation"
                    ],

                "train_r_squared":
                    train_metrics[
                        "r_squared"
                    ],

                "train_rmse":
                    train_metrics[
                        "rmse"
                    ],

                "train_tracking_error":
                    train_metrics[
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

                "test_rmse":
                    test_metrics[
                        "rmse"
                    ],

                "test_mae":
                    test_metrics[
                        "mae"
                    ],

                "test_tracking_error":
                    test_metrics[
                        "tracking_error"
                    ],

                "test_actual_volatility":
                    test_metrics[
                        "actual_volatility"
                    ],

                "test_replicated_volatility":
                    test_metrics[
                        "replicated_volatility"
                    ],

                "test_volatility_difference":
                    test_metrics[
                        "volatility_difference"
                    ],
            }
        )

        for ticker in selected:

            weight_rows.append(
                {
                    "target": target,
                    "n_stocks": k,
                    "ticker": ticker,
                    "weight": weights[ticker],
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    weights = pd.DataFrame(
        weight_rows
    )

    summary.to_csv(
        RESULTS_DIR /
        f"{target.lower()}_replication_summary.csv",
        index=False
    )

    weights.to_csv(
        RESULTS_DIR /
        f"{target.lower()}_replication_weights.csv",
        index=False
    )

    # --------------------------------------------------
    # Find best test portfolio by correlation
    # --------------------------------------------------

    best_row = summary.loc[
        summary["test_correlation"].idxmax()
    ]

    best_k = int(
        best_row["n_stocks"]
    )

    best_replicated = (
        all_test_replicated[best_k]
    )

    # --------------------------------------------------
    # Actual vs replicated test plot
    # --------------------------------------------------

    best_test_data = test[
        [target]
    ].join(
        best_replicated.rename(
            "replicated"
        ),
        how="inner"
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    ax.plot(
        best_test_data.index,
        best_test_data[target].cumsum(),
        label=f"{target} cumulative returns"
    )

    ax.plot(
        best_test_data.index,
        best_test_data["replicated"].cumsum(),
        label=(
            f"{best_k}-stock replication "
            "cumulative returns"
        )
    )

    ax.set_title(
        f"{target}: Actual vs "
        f"{best_k}-Stock Replication"
    )

    ax.set_ylabel(
        "Cumulative log return"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR /
        f"{target.lower()}_actual_vs_replicated.png",
        dpi=200
    )

    plt.close(fig)

    print("\n")
    print("=" * 70)
    print(
        f"BEST {target} "
        "REPLICATION ON TEST DATA"
    )
    print("=" * 70)

    print(
        f"Number of stocks: {best_k}"
    )

    print(
        f"Test correlation: "
        f"{best_row['test_correlation']:.4f}"
    )

    print(
        f"Test R-squared: "
        f"{best_row['test_r_squared']:.4f}"
    )

    print(
        f"Test tracking error: "
        f"{best_row['test_tracking_error']:.6f}"
    )

    print(
        f"Test RMSE: "
        f"{best_row['test_rmse']:.6f}"
    )

    return summary, weights