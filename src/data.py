"""Synthetic multi-asset return generator with crisis regimes."""

import numpy as np
import pandas as pd

from src import config


def generate_returns(
    n_assets: int = config.N_ASSETS,
    n_days: int = config.N_DAYS,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Generate correlated asset returns with regime switching."""
    rng = np.random.default_rng(seed)
    assets = [f"asset_{i}" for i in range(n_assets)]
    returns = np.zeros((n_days, n_assets))
    regime = np.zeros(n_days, dtype=int)

    for t in range(1, n_days):
        if regime[t - 1] == 1 and rng.random() < config.CRISIS_PERSIST:
            regime[t] = 1
        elif rng.random() < config.CRISIS_PROB:
            regime[t] = 1

        vol = config.VOL_NORMAL if regime[t] == 0 else config.VOL_CRISIS
        base_corr = config.CORR_NORMAL if regime[t] == 0 else config.CORR_CRISIS

        cov = np.full((n_assets, n_assets), base_corr)
        np.fill_diagonal(cov, 1.0)
        L = np.linalg.cholesky(cov)
        z = L @ rng.standard_normal(n_assets)
        returns[t] = vol * z

    df = pd.DataFrame(returns, columns=assets)
    df["regime"] = regime
    return df


def build_target(df: pd.DataFrame, horizon: int = config.HORIZON) -> pd.Series:
    """Binary label: 1 if any crisis day within next `horizon` days."""
    future_crisis = (
        df["regime"]
        .shift(-1)
        .rolling(horizon)
        .max()
        .shift(-(horizon - 1))
    )
    return (future_crisis == 1).astype(int)


def save_raw(df: pd.DataFrame, name: str = "synthetic_returns.csv") -> None:
    path = config.DATA_RAW / name
    df.to_csv(path, index=False)
    print(f"[data] saved raw returns -> {path}")