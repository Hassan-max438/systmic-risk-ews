"""Train supervised classifiers with time-series split and imbalance handling."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src import config


def time_split(
    X: pd.DataFrame, y: pd.Series, test_fraction: float = config.TEST_FRACTION
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split = int(len(X) * (1 - test_fraction))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def class_weights(y: pd.Series) -> Dict[int, float]:
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, w))


def get_models(class_weight: Dict[int, float]) -> Dict[str, object]:
    return {
        "LogReg": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight=class_weight, max_iter=1000, random_state=config.SEED
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            class_weight=class_weight, **config.RF_PARAMS
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=config.SEED,
        ),
    }


def walk_forward_pr_auc(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    from sklearn.metrics import average_precision_score

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr, va in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[va])[:, 1]
        scores.append(average_precision_score(y.iloc[va], p))
    return float(np.mean(scores)), float(np.std(scores))