pip install kaggle
"""
Predicting national life expectancy from development indicators.
AAE 718 - Project 04 (Models).

Data: WHO Life Expectancy dataset, pulled from Kaggle via the Kaggle API
(kumarajarshi/life-expectancy-who). No data files are stored in the repo.

"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

plt.rcParams.update({"figure.dpi":130,"savefig.dpi":150,"savefig.bbox":"tight",
    "font.size":11,"axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.25,"axes.titleweight":"bold","axes.titlesize":12})
A, B = "#1f6f8b", "#e1701a"
DATASET = "kumarajarshi/life-expectancy-who"
TARGET = "Life expectancy"


def fetch_data(dest="data"):

    os.makedirs(dest, exist_ok=True)
    csv_path = os.path.join(dest, "Life Expectancy Data.csv")
    if not os.path.exists(csv_path):
        # If a ~/.kaggle/access_token file exists, load it into the env var the
        # client expects, so the token lives in a git-ignored file, not in code.
        token_file = os.path.expanduser("~/.kaggle/access_token")
        if "KAGGLE_API_TOKEN" not in os.environ and os.path.exists(token_file):
            with open(token_file) as fh:
                os.environ["KAGGLE_API_TOKEN"] = fh.read().strip()
        import kaggle                       # reads KAGGLE_API_TOKEN or ~/.kaggle/kaggle.json
        kaggle.api.authenticate()
        print("Downloading dataset from Kaggle ...")
        kaggle.api.dataset_download_files(DATASET, path=dest, unzip=True)
    return pd.read_csv(csv_path)


def clean_data(df):
    """Tidy column names, encode status, drop rows with no target."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=[TARGET])
    df["Status"] = (df["Status"] == "Developed").astype(int)   # 1=Developed
    return df


def build_features(df):
    feat = [c for c in df.columns if c not in ("Country", TARGET)]
    return df[feat].copy(), df[TARGET].copy(), feat


def main():
    os.makedirs("images", exist_ok=True)
    df = clean_data(fetch_data())
    X, y, feat = build_features(df)
    print(f"rows: {len(df)}, features: {len(feat)}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    medians = X_tr.median(numeric_only=True)            # impute with TRAIN medians (no leakage)
    X_tr, X_te = X_tr.fillna(medians), X_te.fillna(medians)

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
    lr = LinearRegression().fit(X_tr, y_tr)
    print(f"Random forest   -- train R2 {rf.score(X_tr,y_tr):.3f}  test R2 {rf.score(X_te,y_te):.3f}"
          f"  test MAE {mean_absolute_error(y_te, rf.predict(X_te)):.2f} yrs")
    print(f"Linear baseline -- train R2 {lr.score(X_tr,y_tr):.3f}  test R2 {lr.score(X_te,y_te):.3f}")

    # Fig 1: predicted vs actual
    pred = rf.predict(X_te)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_te, pred, s=14, alpha=0.4, color=A)
    lo, hi = y_te.min(), y_te.max()
    ax.plot([lo,hi],[lo,hi], color=B, lw=1.8, ls="--", label="perfect prediction")
    ax.set_title(f"Predicted vs actual life expectancy (test R\u00b2={rf.score(X_te,y_te):.3f})")
    ax.set_xlabel("Actual life expectancy (years)"); ax.set_ylabel("Predicted (years)")
    ax.legend(fontsize=9, frameon=False)
    fig.savefig("images/fig1_pred_vs_actual.png"); plt.close(fig)

    # Fig 2: feature importance
    imp = pd.Series(rf.feature_importances_, index=feat).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(7.5,4.8))
    imp.plot.barh(ax=ax, color=A)
    ax.set_title("Top-10 feature importance"); ax.set_xlabel("Importance")
    fig.savefig("images/fig2_importance.png"); plt.close(fig)

    # Fig 3: schooling vs life expectancy
    fig, ax = plt.subplots(figsize=(7.5,5))
    for st, lab, col in [(1,"Developed",B), (0,"Developing",A)]:
        d = df[df["Status"]==st]
        ax.scatter(d["Schooling"], d[TARGET], s=12, alpha=0.4, color=col, label=lab)
    ax.set_title("Schooling vs life expectancy"); ax.set_xlabel("Years of schooling")
    ax.set_ylabel("Life expectancy (years)"); ax.legend(fontsize=9, frameon=False)
    fig.savefig("images/fig3_schooling.png"); plt.close(fig)

    print("Saved figures to images/.")


if __name__ == "__main__":
    main()
