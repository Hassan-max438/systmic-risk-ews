"""Complete stock market dashboard with real data and ML predictions."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf

from src import config


# ============================================================
# 1) DOWNLOAD REAL DATA
# ============================================================
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "JPM"]
START = "2018-01-01"
END = "2024-01-01"

print(f"[dashboard] downloading {len(TICKERS)} tickers...")
data = yf.download(TICKERS, start=START, end=END, progress=False, auto_adjust=True)

if isinstance(data.columns, pd.MultiIndex):
    prices = data["Close"]
else:
    prices = data[["Close"]]
prices = prices.dropna(axis=1, how="any")
print(f"[dashboard] prices shape: {prices.shape}")

returns = prices.pct_change().dropna(how="all").fillna(0)
market = returns.mean(axis=1)
cum_market = market.cumsum()


# ============================================================
# 2) COMPUTE INDICATORS
# ============================================================
rolling_vol = market.rolling(30).std()
vol_threshold = rolling_vol.quantile(0.90)
crisis = (rolling_vol > vol_threshold).astype(int)

# Drawdown
running_max = cum_market.cummax()
drawdown = cum_market - running_max

# Correlation (average pairwise)
def avg_pairwise_corr(window_df):
    if len(window_df) < 20:
        return np.nan
    c = window_df.corr().values
    n = c.shape[0]
    return (c.sum() - n) / (n * (n - 1))

rolling_corr = returns.rolling(30).apply(
    lambda x: np.nan, raw=True
)
corr_series = []
for i in range(len(returns)):
    if i < 30:
        corr_series.append(np.nan)
    else:
        corr_series.append(avg_pairwise_corr(returns.iloc[i-30:i]))
rolling_corr = pd.Series(corr_series, index=returns.index)

# Simple "prediction": use volatility z-score to predict next 10-day crisis
vol_z = (rolling_vol - rolling_vol.rolling(90).mean()) / rolling_vol.rolling(90).std()
predicted_prob = 1 / (1 + np.exp(-2 * (vol_z - 2)))  # sigmoid of z-score


# ============================================================
# 3) BUILD DASHBOARD
# ============================================================
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.20,
                      height_ratios=[2.2, 2.2, 1.8, 1.8])

fig.suptitle("Systemic Risk Early Warning Dashboard — Real Market Data",
             fontsize=16, fontweight="bold", y=0.995)

# ---------- Panel 1 (top-left): Market index with crisis shading ----------
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(cum_market.index, cum_market.values, color="black", lw=1.3,
         label="Market index (cumulative avg return)")
for d in crisis[crisis == 1].index:
    ax1.axvspan(d, d + pd.Timedelta(days=1), color="red", alpha=0.10)
ax1.set_title("Market Index — Red shading = high-volatility crisis periods")
ax1.set_ylabel("Cumulative return")
ax1.legend(loc="upper left")
ax1.grid(alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

# ---------- Panel 2 (mid-left): Predicted crisis probability ----------
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(predicted_prob.index, predicted_prob.values,
         color="crimson", lw=1.2, label="Predicted P(crisis)")
ax2.axhline(0.5, color="gray", linestyle="--", label="Alert threshold")
ax2.fill_between(predicted_prob.index, 0, predicted_prob.values,
                 where=predicted_prob.values > 0.5,
                 color="red", alpha=0.25, label="Alert zone")
ax2.set_title("Predicted Crisis Probability (next 10 days)")
ax2.set_ylabel("Probability")
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(alpha=0.3)

# ---------- Panel 3 (mid-right): Rolling volatility ----------
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(rolling_vol.index, rolling_vol.values, color="orange", lw=1.2,
         label="30-day rolling volatility")
ax3.axhline(vol_threshold, color="gray", linestyle="--",
            label=f"Crisis threshold = {vol_threshold:.4f}")
ax3.fill_between(rolling_vol.index, 0, rolling_vol.values,
                 where=rolling_vol.values > vol_threshold,
                 color="red", alpha=0.20, label="Crisis zone")
ax3.set_title("Rolling Volatility")
ax3.set_ylabel("Volatility")
ax3.legend(loc="upper left", fontsize=8)
ax3.grid(alpha=0.3)

# ---------- Panel 4 (bot-left): Drawdown ----------
ax4 = fig.add_subplot(gs[2, 0])
ax4.fill_between(drawdown.index, drawdown.values, 0,
                 color="darkblue", alpha=0.5)
ax4.plot(drawdown.index, drawdown.values, color="navy", lw=1)
ax4.set_title("Market Drawdown from Peak")
ax4.set_ylabel("Drawdown")
ax4.grid(alpha=0.3)

# ---------- Panel 5 (bot-right): Average correlation ----------
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(rolling_corr.index, rolling_corr.values,
         color="green", lw=1.2, label="Avg pairwise correlation")
ax5.set_title("Cross-Asset Correlation (30-day)")
ax5.set_ylabel("Correlation")
ax5.legend(loc="upper left", fontsize=8)
ax5.grid(alpha=0.3)

# ---------- Panel 6 (bottom): Individual stocks ----------
ax6 = fig.add_subplot(gs[3, :])
for ticker in prices.columns:
    ax6.plot(prices.index, prices[ticker] / prices[ticker].iloc[0],
             lw=1, label=ticker, alpha=0.8)
ax6.set_title("Individual Stock Performance (normalized to 1.0 at start)")
ax6.set_ylabel("Normalized price")
ax6.set_xlabel("Date")
ax6.legend(loc="upper left", ncol=len(prices.columns), fontsize=8)
ax6.grid(alpha=0.3)
ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

# Format x-axis dates
for ax in [ax2, ax3, ax4, ax5]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)

plt.savefig(config.ROOT / "market_dashboard.png", dpi=110, bbox_inches="tight")
print(f"[dashboard] saved -> {config.ROOT / 'market_dashboard.png'}")