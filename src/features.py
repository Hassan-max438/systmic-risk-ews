"""Rolling correlation / clustering features that feed the classifier."""

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config


def _mean_off_diag(corr: np.ndarray) -> float:
    n = corr.shape[0]
    return (corr.sum() - n) / (n * (n - 1))


def _mean_pairwise_distance(corr: np.ndarray) -> float:
    dist = 1 - np.abs(corr)
    n = dist.shape[0]
    tri = dist[np.triu_indices(n, k=1)]
    return float(tri.mean())


def build_features(df: pd.DataFrame, windows=config.WINDOWS) -> pd.DataFrame:
    """Compute rolling correlation and clustering-distance features."""
    assets = [c for c in df.columns if c.startswith("asset_")]
    features = pd.DataFrame(index=df.index)

    for w in windows:
        avg_corr = []
        cluster_dist = []
        for i in tqdm(range(len(df)), desc=f"rolling w={w}"):
            if i < w:
                avg_corr.append(np.nan)
                cluster_dist.append(np.nan)
                continue
            sub = df[assets].iloc[i - w : i]
            corr = sub.corr().values
            avg_corr.append(_mean_off_diag(corr))
            cluster_dist.append(_mean_pairwise_distance(corr))

        features[f"avg_corr_{w}"] = avg_corr
        features[f"cluster_dist_{w}"] = cluster_dist

    features["sys_vol_30"] = df[assets].rolling(30).std().mean(axis=1)
    features["sys_vol_90"] = df[assets].rolling(90).std().mean(axis=1)

    return features


def assemble_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Features + target, drop NaN rows from rolling windows."""
    from src.data import build_target

    feats = build_features(df)
    feats["target"] = build_target(df)
    feats = feats.dropna()
    return feats