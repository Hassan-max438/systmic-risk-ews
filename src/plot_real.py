"""Plot real stock market data with crisis periods shaded."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from src import config

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "JPM"]
START = "2015-01-01"
END = "2024-01-01"

print(f"[plot_real] downloading {len(TICKERS)} tickers from Yahoo Finance...")
data = yf.download(TICKERS, start=START, end=END, progress=False, auto_adjust=True)

if isinstance(data.columns, pd.MultiIndex):
    prices = data["Close"]
else:
    prices = data[["Close"]]

prices = prices.dropna(axis=1, how="any")
print(f"[plot_real] prices shape: {prices.shape}")
print(f"[plot_real] tickers kept: {list(prices.columns)}")

returns = prices.pct_change().dropna(how="all").fillna(0)

market = returns.mean(axis=1)
rolling_vol = market.rolling(30).std()
threshold = rolling_vol.quantile(0.90)
crisis = (rolling_vol > threshold).astype(int)

cum = market.cumsum()
running_max = cum.cummax()
drawdown = cum - running_max
dd_threshold = drawdown.quantile(0.10)

print(f"[plot_real] vol threshold: {threshold:.5f}")
print(f"[plot_real] crisis days: {crisis.sum()} / {len(crisis)}")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                         gridspec_kw={"height_ratios": [2, 2, 1]})

axes[0].plot(cum.index, cum.values, color="black", lw=1.2,
             label="Market index (cumulative avg return)")
for d in crisis[crisis == 1].index:
    axes[0].axvspan(d, d + pd.Timedelta(days=1), color="red", alpha=0.12)
axes[0].set_ylabel("Cumulative return")
axes[0].set_title(f"Real market: {', '.join(prices.columns)} ({START} to {END}) - red = crisis days")
axes[0].legend(loc="upper left")
axes[0].grid(alpha=0.3)

axes[1].plot(rolling_vol.index, rolling_vol.values, color="crimson", lw=1,
             label="30-day rolling volatility")
axes[1].axhline(threshold, color="gray", linestyle="--",
                label=f"Crisis threshold (top 10%) = {threshold:.4f}")
axes[1].fill_between(rolling_vol.index, 0, rolling_vol.values,
                     where=rolling_vol.values > threshold,
                     color="red", alpha=0.20, label="Crisis zone")
axes[1].set_ylabel("Volatility")
axes[1].set_title("Rolling volatility as crisis indicator")
axes[1].legend(loc="upper left")
axes[1].grid(alpha=0.3)

axes[2].plot(drawdown.index, drawdown.values, color="darkblue", lw=1,
             label="Drawdown from peak")
axes[2].axhline(dd_threshold, color="gray", linestyle="--",
                label=f"Worst 10% drawdown = {dd_threshold:.4f}")
axes[2].set_ylabel("Drawdown")
axes[2].set_xlabel("Date")
axes[2].set_title("Market drawdown")
axes[2].legend(loc="lower left")
axes[2].grid(alpha=0.3)

plt.tight_layout()
out = config.ROOT / "real_market_chart.png"
plt.savefig(out, dpi=120)
print(f"[plot_real] saved -> {out}")

csv_out = config.DATA_RAW / "real_prices.csv"
prices.to_csv(csv_out)
print(f"[plot_real] saved prices -> {csv_out}")