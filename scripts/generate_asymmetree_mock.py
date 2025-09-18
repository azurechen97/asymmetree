import argparse
import math
from typing import Tuple

import numpy as np
import pandas as pd


def _make_categorical(n: int, seed: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    # Uneven category distributions and some rare categories
    cat_a = rng.choice(
        ["A", "B", "C", "D", "E"], size=n, p=[0.45, 0.25, 0.15, 0.10, 0.05]
    )
    cat_b = rng.choice(["X1", "X2", "X3"], size=n, p=[0.7, 0.25, 0.05])
    # High-cardinality feature (e.g., zip codes or hashed IDs)
    hi_cards = [f"Z{z:04d}" for z in range(500)]
    # Zipf-like skew: probability ~ 1/(k+1)
    weights = np.array([1.0 / (k + 1) for k in range(len(hi_cards))], dtype=float)
    weights /= weights.sum()
    cat_c = rng.choice(hi_cards, size=n, p=weights)
    return (
        pd.Series(cat_a, name="cat_a"),
        pd.Series(cat_b, name="cat_b"),
        pd.Series(cat_c, name="cat_c"),
    )


def _make_numeric(
    n: int, seed: int
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed + 1)
    num_a = rng.normal(loc=0.0, scale=1.0, size=n)
    num_b = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    # Bimodal distribution to stress tree splits
    mode = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    num_c = rng.normal(loc=np.where(mode == 0, -2.0, 3.5), scale=0.8, size=n)
    # Correlated feature
    num_d = 0.6 * num_a + 0.4 * rng.normal(size=n)
    return (
        pd.Series(num_a, name="num_a"),
        pd.Series(num_b, name="num_b"),
        pd.Series(num_c, name="num_c"),
        pd.Series(num_d, name="num_d"),
    )


def _compute_logit(
    df: pd.DataFrame,
    base_log_odds: float,
) -> np.ndarray:
    # Start from base log-odds controlling prevalence
    n = len(df)
    logit = np.full(n, base_log_odds, dtype=float)

    # Strong, tree-friendly signals: crisp thresholds and categorical interactions
    num_a = df["num_a"].values
    num_b = df["num_b"].values
    num_c = df["num_c"].values
    num_d = df["num_d"].values

    cat_a = df["cat_a"].values
    cat_b = df["cat_b"].values
    cat_c = df["cat_c"].values

    # Positive risk rules (large positive contributions)
    logit += 3.5 * (num_c > 2.5)
    logit += 2.0 * (num_b > 4.5)
    logit += 3.0 * (num_b > 7.5)
    logit += 2.5 * (num_a > 1.8)
    logit += 2.2 * (num_d > 1.2)

    # Interactions that trees can split on
    logit += 3.8 * ((num_b > 5.5) & (num_a > 1.5))
    logit += 4.2 * ((cat_b == "X3") & (num_c > 2.2))
    logit += 3.0 * ((np.isin(cat_a, ["D", "E"])) & (num_d > 0.8))
    logit += 2.5 * (cat_b == "X2")
    logit += 3.2 * (cat_a == "E")

    # Targeted uplift for some high-cardinality values
    hot_z = {f"Z{z:04d}" for z in [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]}
    logit += 3.5 * np.isin(cat_c, list(hot_z))

    # Negative risk (suppressors) to create contrastive structure
    logit += -2.0 * (num_a < -1.5)
    logit += -2.5 * (num_c < -1.0)
    logit += -1.2 * ((cat_a == "A") & (num_b < 1.0))
    logit += -2.0 * ((cat_b == "X1") & (num_d < -0.5))

    return logit


def _calibrate_base_log_odds(
    n: int,
    target_rate: float,
    seed: int,
    max_iter: int = 30,
    tol: float = 1e-4,
) -> float:
    # Binary search over base_log_odds to hit desired prevalence
    # Start with wide odds range
    lo, hi = -20.0, -2.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        df = _synthesize_once(n=n, seed=seed, base_log_odds=mid, return_only_df=True)
        probs = 1.0 / (1.0 + np.exp(-_compute_logit(df, base_log_odds=mid)))
        rate = float(np.mean(probs))
        if abs(rate - target_rate) < tol:
            return mid
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _synthesize_once(
    n: int,
    seed: int,
    base_log_odds: float,
    return_only_df: bool = False,
) -> pd.DataFrame:
    cat_a, cat_b, cat_c = _make_categorical(n, seed)
    num_a, num_b, num_c, num_d = _make_numeric(n, seed)
    df = pd.concat([cat_a, cat_b, cat_c, num_a, num_b, num_c, num_d], axis=1)
    if return_only_df:
        return df
    logit = _compute_logit(df, base_log_odds=base_log_odds)
    probs = 1.0 / (1.0 + np.exp(-logit))
    rng = np.random.default_rng(seed + 7)
    y = rng.binomial(1, probs)
    df["target"] = y.astype(int)
    return df


def generate_dataset(
    n_rows: int = 200_000,
    target_rate: float = 0.005,
    seed: int = 7,
) -> pd.DataFrame:
    base_log_odds = _calibrate_base_log_odds(n_rows, target_rate, seed)
    df = _synthesize_once(n_rows, seed, base_log_odds)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate mock dataset for Asymmetree")
    parser.add_argument(
        "--out", type=str, default="data/asymmetree_mock.csv", help="Output CSV path"
    )
    parser.add_argument("--rows", type=int, default=200_000, help="Number of rows")
    parser.add_argument(
        "--rate",
        type=float,
        default=0.005,
        help="Target positive rate (e.g., 0.005 for 0.5%)",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    df = generate_dataset(n_rows=args.rows, target_rate=args.rate, seed=args.seed)

    # Place an id column first for convenience
    df.insert(0, "id", np.arange(len(df)))
    # Enforce dtypes explicitly
    df = df.astype(
        {
            "id": "int64",
            "cat_a": "category",
            "cat_b": "category",
            "cat_c": "category",
            "num_a": "float64",
            "num_b": "float64",
            "num_c": "float64",
            "num_d": "float64",
            "target": "int64",
        }
    )

    df.to_csv(args.out, index=False)
    print(
        f"Wrote {len(df):,} rows to {args.out}. Positives: {df['target'].sum():,} ({100*df['target'].mean():.3f}%)"
    )


if __name__ == "__main__":
    main()
