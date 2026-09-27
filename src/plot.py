import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""Generate a chart showing predictions vs. actual crises over time."""

import joblib
...

import joblib
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.data import generate_returns
from src.evaluate import best_threshold
from src.features import assemble_dataset
from src.train import class_weights, get_models, time_split


def main():
    # 1) Data + features
    df = generate_returns()
    data = assemble_dataset(df)
    X = data.drop(columns="target")
    y = data["target"]

    # 2) Split + train
    X_tr, X_te, y_tr, y_te = time_split(X, y)
    cw = class_weights(y_tr)
    models = get_models(cw)
    rf = models["RandomForest"]
    rf.fit(X_tr, y_tr)

    # 3) Predictions on test set
    proba = rf.predict_proba(X_te)[:, 1]
    thr = best_threshold(y_te, proba)

    # 4) Market index for context
    assets = [c for c in df.columns if c.startswith("asset_")]
    market = df[assets].mean(axis=1)
    test_index = data.index[-len(X_te):]
    market_test = market.loc[test_index]

    # 5) Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1]})

    axes[0].plot(market_test.values, color="black", lw=1,
                 label="Market index (avg return)")
    crisis_days = np.where(y_te.values == 1)[0]
    for i in crisis_days:
        axes[0].axvspan(i - 0.5, i + 0.5, color="red", alpha=0.08)
    axes[0].set_ylabel("Avg return")
    axes[0].set_title("Market index — red shaded = actual crisis within next 10 days")
    axes[0].legend(loc="upper left")

    axes[1].plot(proba, color="crimson", lw=1.2, label="Predicted P(crisis)")
    axes[1].axhline(thr, color="gray", linestyle="--",
                    label=f"Alert threshold = {thr:.3f}")
    axes[1].fill_between(range(len(proba)), 0, proba,
                         where=proba > thr, color="red", alpha=0.25,
                         label="Alert zone")
    axes[1].set_ylabel("P(crisis)")
    axes[1].set_title("Model's predicted crisis probability")
    axes[1].legend(loc="upper left")

    alerts = (proba > thr).astype(int)
    axes[2].step(range(len(alerts)), alerts, where="mid",
                 color="darkred", label="Alert (1 = raised)")
    axes[2].step(range(len(y_te)), y_te.values, where="mid",
                 color="black", alpha=0.5, label="Actual crisis (1 = yes)")
    axes[2].set_ylabel("0 / 1")
    axes[2].set_xlabel("Test-set time index")
    axes[2].set_title("Alerts vs actual crises")
    axes[2].legend(loc="upper left")

    plt.tight_layout()
    out = config.ROOT / "prediction_chart.png"
    plt.savefig(out, dpi=120)
    print(f"[plot] saved chart -> {out}")

  #  plt.show()


if __name__ == "__main__":
    main()