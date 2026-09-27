"""Metrics, threshold tuning, feature importance, saving."""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from src import config


def report(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba > 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "report": classification_report(y_test, preds, digits=3),
        "confusion": confusion_matrix(y_test, preds),
        "proba": proba,
    }


def best_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thr[np.argmax(f1[:-1])])


def feature_importance(model, columns) -> pd.Series:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        imp = model.steps[-1][1].coef_[0]
    return pd.Series(imp, index=columns).sort_values(ascending=False)


def save_model(model, threshold: float, path=config.MODEL_PATH) -> None:
    joblib.dump({"model": model, "threshold": threshold}, path)
    print(f"[evaluate] saved model -> {path}")