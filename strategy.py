import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

ALPHA_KEY = "5LNC4VWREO8TCP58"

def fetch_stock(ticker):
    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY&symbol={ticker}"
        f"&apikey={ALPHA_KEY}&outputsize=compact"
    )
    r = requests.get(url)
    data = r.json()
    print("ALPHA RESPONSE:", list(data.keys())[:3])
    time_series_key = None
    for key in data.keys():
        if "Daily" in key:
            time_series_key = key
            break
    if time_series_key is None:
        raise ValueError(f"Alpha Vantage returned error: {data}")
    ts = data[time_series_key]
    df = pd.DataFrame({
        "date": list(ts.keys()),
        "close": [float(v["4. close"]) for v in ts.values()]
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df.sort_index()

def fetch_data(ticker1, ticker2, period="6mo", interval="1d"):
    df1 = fetch_stock(ticker1)
    df2 = fetch_stock(ticker2)
    if df1.empty or df2.empty:
        raise ValueError("Alpha Vantage returned no data for one of the tickers")
    df = pd.DataFrame({
        "p1": df1["close"],
        "p2": df2["close"]
    }).dropna()
    if df.empty:
        raise ValueError("No overlapping price data between tickers")
    return df

def compute_hedge_ratio(df):
    x = df["p2"].values.reshape(-1, 1)
    y = df["p1"].values
    beta = (x.T @ y) / (x.T @ x)[0]
    return float(beta)

def build_features(df, beta, lookback=20, horizon=5):
    spread = df["p1"] - beta * df["p2"]
    s_mean = spread.rolling(lookback).mean()
    s_std = spread.rolling(lookback).std()
    z = (spread - s_mean) / s_std
    mom = spread.diff()
    vol = spread.rolling(lookback).std()
    corr = df["p1"].rolling(lookback).corr(df["p2"])
    future_reversion = []
    for i in range(len(spread)):
        if i + horizon >= len(spread):
            future_reversion.append(np.nan)
            continue
        current = spread.iloc[i]
        window = spread.iloc[i+1:i+1+horizon]
        reverts = window.sub(current).abs().min()
        label = 1 if reverts < s_std.iloc[i] else 0
        future_reversion.append(label)
    features = pd.DataFrame({
        "spread": spread,
        "zscore": z,
        "mom": mom,
        "vol": vol,
        "corr": corr,
        "label": future_reversion
    }).dropna()
    return features

def train_model(features):
    X = features[["spread", "zscore", "mom", "vol", "corr"]]
    y = features["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc, X

def compute_equity_stats(spread, signals):
    positions = signals.shift().fillna(0)
    ret = (-spread.diff().fillna(0)) * positions
    if ret.abs().sum() == 0:
        equity = pd.Series([0] * len(ret), index=ret.index)
    else:
        equity = ret.cumsum()
    sharpe = np.sqrt(252) * ret.mean() / ret.std() if ret.std() != 0 else 0
    max_dd = 0
    peak = equity.iloc[0] if len(equity) else 0
    for v in equity:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    stats = {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_return": float(equity.iloc[-1]) if len(equity) else 0.0
    }
    return equity, stats

def backtest_ml(df, beta, model, X, z_entry=1.0, prob_threshold=0.6):
    spread = df["p1"] - beta * df["p2"]
    s_mean = spread.rolling(20).mean()
    s_std = spread.rolling(20).std()
    z = (spread - s_mean) / s_std
    probs = model.predict_proba(X)[:, 1]
    signals = pd.Series(0, index=X.index)
    for i in range(len(X)):
        idx = X.index[i]
        if np.isnan(z.loc[idx]):
            continue
        if abs(z.loc[idx]) > z_entry and probs[i] > prob_threshold:
            signals.loc[idx] = -1 if z.loc[idx] > 0 else 1
    equity, stats = compute_equity_stats(spread.loc[signals.index], signals)
    return signals, equity, stats, z, spread

def backtest_zscore(df, beta, z_entry=1.0, lookback=20):
    spread = df["p1"] - beta * df["p2"]
    s_mean = spread.rolling(lookback).mean()
    s_std = spread.rolling(lookback).std()
    z = (spread - s_mean) / s_std
    signals = pd.Series(0, index=df.index)
    for idx in df.index:
        if np.isnan(z.loc[idx]):
            continue
        if abs(z.loc[idx]) > z_entry:
            signals.loc[idx] = -1 if z.loc[idx] > 0 else 1
    equity, stats = compute_equity_stats(spread, signals)
    return signals, equity, stats, z, spread

def plot_price(df, t1, t2):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df.index, df["p1"], label=t1)
    ax.plot(df.index, df["p2"], label=t2)
    ax.legend()
    ax.set_title("Prices")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img

def plot_equity(equity):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(equity.index, equity.values)
    ax.set_title("Equity Curve")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img

def plot_spread_z(spread, z):
    df = pd.DataFrame({"spread": spread, "z": z}).dropna()
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(df.index, df["spread"], label="Spread")
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["z"], linestyle="dashed", label="Z-score")
    ax1.set_title("Spread vs Z-score")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img

def run_strategy(ticker1, ticker2, period="6mo", interval="1d"):
    df = fetch_data(ticker1, ticker2, period, interval)
    beta = compute_hedge_ratio(df)
    features = build_features(df, beta)
    model, acc, X = train_model(features)
    signals, equity, stats, z, spread = backtest_ml(df, beta, model, X, z_entry=1.0, prob_threshold=0.6)
    price_img = plot_price(df, ticker1, ticker2)
    eq_img = plot_equity(equity)
    spread_img = plot_spread_z(spread, z)
    return {
        "model_accuracy": acc,
        "stats": stats,
        "price_img": price_img,
        "equity_img": eq_img,
        "spread_img": spread_img,
        "beta": beta
    }

def run_backtest(ticker1, ticker2, strategy_type="ml", z_entry=1.0, prob_threshold=0.6, lookback=20, horizon=5):
    df = fetch_data(ticker1, ticker2)
    beta = compute_hedge_ratio(df)
    model_acc = None
    if strategy_type == "ml":
        features = build_features(df, beta, lookback=lookback, horizon=horizon)
        model, model_acc, X = train_model(features)
        signals, equity, stats, z, spread = backtest_ml(df, beta, model, X, z_entry=z_entry, prob_threshold=prob_threshold)
    else:
        signals, equity, stats, z, spread = backtest_zscore(df, beta, z_entry=z_entry, lookback=lookback)
    n_trades = int((signals != 0).sum())
    out = {
        "ticker1": ticker1,
        "ticker2": ticker2,
        "sharpe": stats["sharpe"],
        "total_return": stats["total_return"],
        "max_drawdown": stats["max_drawdown"],
        "model_accuracy": model_acc,
        "n_trades": n_trades
    }
    return out