"""End-to-end pipeline: data -> features -> train -> evaluate -> save."""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src import config
from src.data import generate_returns, save_raw
from src.evaluate import best_threshold, feature_importance, report, save_model
from src.features import assemble_dataset
from src.train import class_weights, get_models, time_split, walk_forward_pr_auc


def main():
    # 1) Data
    df = generate_returns()
    save_raw(df)
    print(f"[main] regime distribution:\n{df['regime'].value_counts(normalize=True)}")

    # 2) Features + target
    data = assemble_dataset(df)
    print(f"[main] dataset shape: {data.shape}")

    X = data.drop(columns="target")
    y = data["target"]
    print(f"[main] class balance:\n{y.value_counts(normalize=True)}")

    # 3) Split (time-series aware)
    X_tr, X_te, y_tr, y_te = time_split(X, y)
    print(f"[main] train {X_tr.shape} | test {X_te.shape}")

    cw = class_weights(y_tr)
    print(f"[main] class weights: {cw}")

    # 4) Train + evaluate each model
    models = get_models(cw)
    for name, m in models.items():
        print(f"\n=== {name} ===")
        m.fit(X_tr, y_tr)
        res = report(m, X_te, y_te)
        print(f"ROC-AUC: {res['roc_auc']:.4f}")
        print(f"PR-AUC : {res['pr_auc']:.4f}")
        print(res["report"])

        thr = best_threshold(y_te, res["proba"])
        preds = (res["proba"] > thr).astype(int)
        print(f"Tuned threshold = {thr:.4f}")
        print("At tuned threshold:")
        print(classification_report(y_te, preds, digits=3))

    # 5) Walk-forward CV on RandomForest
    print("\n=== Walk-forward CV (RandomForest) ===")
    rf = models["RandomForest"]
    mean_pr, std_pr = walk_forward_pr_auc(rf, X, y, n_splits=5)
    print(f"PR-AUC = {mean_pr:.4f} +/- {std_pr:.4f}")

    # 6) Final fit + save
    final = models["RandomForest"]
    final.fit(X_tr, y_tr)
    proba = final.predict_proba(X_te)[:, 1]
    thr = best_threshold(y_te, proba)

    print("\n=== Top feature importances ===")
    print(feature_importance(final, X.columns).head(10))

    save_model(final, thr)

    # 7) Inference example
    loaded = joblib.load(config.MODEL_PATH)
    alerts = (
        loaded["model"].predict_proba(X_te)[:, 1] > loaded["threshold"]
    ).astype(int)
    print(f"[main] alerts raised in test set: {alerts.sum()} / {len(alerts)}")

    
if __name__ == "__main__":
    main()