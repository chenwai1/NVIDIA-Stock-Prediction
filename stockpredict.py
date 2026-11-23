"""
EEM 349 - Stock price prediction: Ridge vs XGBoost, with and without PCA.

Usage:
    python stock_pipeline.py path/to/stock.csv

The CSV needs the columns: Date, Open, High, Low, Close, Volume.
If your column names differ (e.g. "Adj Close"), rename them in load_data().
Outputs (results.csv and PNG figures) go to the "outputs" folder.
"""
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from xgboost import plot_tree

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "stock.csv"
DATE_COL = "Date"
PREDICT = "return"      # "return" = predict next-day return (recommended), "price" = next-day close
TEST_SIZE = 0.20        # last 20% of the timeline is the test set
PCA_VARIANCE = 0.95     # keep components explaining 95% of the training variance
SEED = 42
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Data and features (every feature uses information up to day t only)
# ----------------------------------------------------------------------------
def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss)


def load_data(path):
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df.sort_values(DATE_COL).reset_index(drop=True)


def build_features(df):
    c = df["Close"]
    f = pd.DataFrame(index=df.index)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        f[col] = df[col]
    for n in (5, 10, 20):
        f[f"sma_{n}"] = c.rolling(n).mean()
        f[f"ema_{n}"] = c.ewm(span=n, adjust=False).mean()
    f["rsi_14"] = rsi(c)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    f["macd"] = ema12 - ema26
    f["macd_signal"] = f["macd"].ewm(span=9, adjust=False).mean()
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_upper"] = mid + 2 * sd
    f["bb_lower"] = mid - 2 * sd
    for k in (1, 2, 5):
        f[f"ret_{k}"] = c.pct_change(k)
    f["vol_10"] = c.pct_change().rolling(10).std()
    return f


df = load_data("nvidia_stock_prices.csv")
X_all = build_features(df)
data = X_all.copy()
data["date"] = df[DATE_COL]
data["next_close"] = df["Close"].shift(-1)       # target: close at t+1
data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

feature_cols = list(X_all.columns)
print(f"Rows: {len(data)} | Features: {len(feature_cols)}")

# ----------------------------------------------------------------------------
# 2. Chronological split (never shuffle time series)
# ----------------------------------------------------------------------------
split = int(len(data) * (1 - TEST_SIZE))
train, test = data.iloc[:split], data.iloc[split:]
X_train, X_test = train[feature_cols], test[feature_cols]
close_test = test["Close"].values
price_test = test["next_close"].values

if PREDICT == "return":
    y_train = train["next_close"] / train["Close"] - 1
else:
    y_train = train["next_close"]


def to_price(pred, close):
    """Convert model output back to a price so all models share the same metrics."""
    return close * (1 + pred) if PREDICT == "return" else pred


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


# Naive baseline: tomorrow's close = today's close
baseline = metrics(price_test, close_test)
print("Baseline (yesterday's close):", {k: round(v, 4) for k, v in baseline.items()})


# ----------------------------------------------------------------------------
# 3. One function runs any of the four configurations
#    Scaler and PCA live inside the Pipeline, so they are fitted on training
#    data only (inside every CV fold too) -> no data leakage.
# ----------------------------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=5)
searches, predictions, rows = {}, {}, []


def run_config(label, model, grid, use_pca):
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=PCA_VARIANCE, random_state=SEED)))
    steps.append(("model", model))
    pipe = Pipeline(steps)

    search = GridSearchCV(
        pipe,
        {f"model__{k}": v for k, v in grid.items()},
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    # Refit the best settings on the full training set and time that fit only
    best = clone(pipe).set_params(**search.best_params_)
    t0 = time.perf_counter()
    best.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    pred = to_price(best.predict(X_test), close_test)
    n_features = best.named_steps["pca"].n_components_ if use_pca else X_train.shape[1]
    rows.append({
        "Configuration": label,
        "No. of features": n_features,
        **metrics(price_test, pred),
        "Training time (s)": fit_time,
        "Best params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
    })
    searches[label], predictions[label] = search, pred
    print(f"Done: {label}")


ridge_grid = {"alpha": np.logspace(-3, 3, 13)}
xgb_grid = {
    "max_depth": [2, 3, 5],
    "learning_rate": [0.03, 0.1],
    "n_estimators": [200, 400],
}


def new_xgb():
    # n_jobs=1 here because GridSearchCV already runs folds in parallel
    return XGBRegressor(random_state=SEED, n_jobs=1, subsample=0.8, colsample_bytree=0.8)


run_config("Ridge - without reduction", Ridge(), ridge_grid, use_pca=False)
run_config("Ridge - with PCA", Ridge(), ridge_grid, use_pca=True)
run_config("XGBoost - without reduction", new_xgb(), xgb_grid, use_pca=False)
run_config("XGBoost - with PCA", new_xgb(), xgb_grid, use_pca=True)

results = pd.DataFrame(rows)
results.loc[len(results)] = {
    "Configuration": "Baseline (yesterday's close)", "No. of features": 0,
    **baseline, "Training time (s)": 0.0, "Best params": "-",
}
results.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)
print("\n", results.drop(columns="Best params").round(4).to_string(index=False))

# ----------------------------------------------------------------------------
# 4. Figures for the report
# ----------------------------------------------------------------------------
# (a) Actual vs predicted on the test period
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(test["date"], price_test, label="Actual", color="black", linewidth=1.4)
for label, style in [("Ridge - without reduction", "--"), ("XGBoost - without reduction", ":")]:
    ax.plot(test["date"], predictions[label], style, label=label, linewidth=1.2)
ax.set_title("Actual vs predicted next-day close (test set)")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "actual_vs_predicted.png"), dpi=150)
plt.close(fig)

# (b) PCA cumulative explained variance (fitted on training data only)
scaled = StandardScaler().fit_transform(X_train)
cum_var = np.cumsum(PCA().fit(scaled).explained_variance_ratio_)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, len(cum_var) + 1), cum_var, marker="o")
ax.axhline(PCA_VARIANCE, color="red", linestyle="--", label=f"{PCA_VARIANCE:.0%} threshold")
ax.set_xlabel("Number of components")
ax.set_ylabel("Cumulative explained variance")
ax.set_title("PCA cumulative variance (training data)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pca_variance.png"), dpi=150)
plt.close(fig)


# (c) Parameter sensitivity from the CV results
def sensitivity(ax, label, param, log=False):
    cv = pd.DataFrame(searches[label].cv_results_)
    cv["rmse"] = -cv["mean_test_score"]
    col = f"param_model__{param}"
    cv[col] = cv[col].astype(float)
    best_per_value = cv.groupby(col)["rmse"].min()
    ax.plot(best_per_value.index, best_per_value.values, marker="o")
    ax.set_xlabel(param)
    ax.set_ylabel("CV RMSE (target units)")
    ax.set_title(f"{label.split(' - ')[0]}: {param}")
    if log:
        ax.set_xscale("log")


fig, axes = plt.subplots(1, 3, figsize=(14, 4))
sensitivity(axes[0], "Ridge - without reduction", "alpha", log=True)
sensitivity(axes[1], "XGBoost - without reduction", "max_depth")
sensitivity(axes[2], "XGBoost - without reduction", "learning_rate")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "parameter_sensitivity.png"), dpi=150)
plt.close(fig)

# (d) Feature importance from XGBoost (without reduction)
xgb_best = searches["XGBoost - without reduction"].best_estimator_
importance = pd.Series(xgb_best.named_steps["model"].feature_importances_, index=feature_cols)
fig, ax = plt.subplots(figsize=(7, 5))
importance.sort_values().plot.barh(ax=ax)
ax.set_title("XGBoost feature importance")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "xgb_feature_importance.png"), dpi=150)
plt.close(fig)

print(f"\nSaved results.csv and figures to ./{OUT_DIR}/")

# (e) Visual decision tree from the best XGBoost model
xgb_model = xgb_best.named_steps["model"]
# Assign actual feature names to the booster so nodes show 'Close', 'rsi_14', etc. instead of 'f0', 'f1'
xgb_model.get_booster().feature_names = feature_cols

fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
plot_tree(
    xgb_model,
    tree_idx=0,        # Use num_trees=0 if using XGBoost < 3.0
    rankdir="TB",
    ax=ax
)
ax.set_title("XGBoost Regression Tree (Tree #0) - NVDA Stock Prediction", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "xgboost_nvda_tree.png"), dpi=300)
plt.close(fig)