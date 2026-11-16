"""West Midlands house price model on HM Land Registry Price Paid Data.

Trains on 2024-2025 sales, tests out-of-time on 2026 sales, and compares a
gradient-boosted model against a median-by-outcode-and-type baseline.
Run: python pipeline.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"
URL = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-{year}.csv"
COLS = ["id", "price", "date", "postcode", "type", "new_build", "tenure", "paon", "saon",
        "street", "locality", "town", "district", "county", "category", "status"]
KEEP = ["price", "date", "postcode", "type", "new_build", "tenure", "district", "county", "category"]
TRAIN_YEARS, TEST_YEAR = (2024, 2025), 2026
CAT = ["type", "new_build", "tenure", "district", "outcode"]
NUM = ["lat", "lon", "month_idx", "neighbour_logp"]
K = 15


def load_sales() -> pd.DataFrame:
    cache = DATA / "sales.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frames = []
    for year in (*TRAIN_YEARS, TEST_YEAR):
        print(f"downloading {year} ...", flush=True)
        df = pd.read_csv(URL.format(year=year), header=None, names=COLS, usecols=KEEP)
        frames.append(df[df.county == "WEST MIDLANDS"])
    sales = pd.concat(frames, ignore_index=True)
    DATA.mkdir(exist_ok=True)
    sales.to_parquet(cache)
    return sales


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standard-price residential sales only; fixed price bounds so the test set is not used to set them."""
    df = df[(df.category == "A") & (df.type != "O") & df.postcode.notna()
            & df.price.between(30_000, 2_000_000)].copy()
    df["date"] = pd.to_datetime(df.date)
    df["outcode"] = df.postcode.str.split().str[0]
    df["month_idx"] = (df.date.dt.year - 2024) * 12 + df.date.dt.month - 1
    return df


def postcode_coords(postcodes) -> pd.DataFrame:
    """Latitude and longitude per postcode from the postcodes.io bulk endpoint (100 per request)."""
    cache = DATA / "postcodes.csv"
    if cache.exists():
        return pd.read_csv(cache)
    todo = sorted(postcodes)
    rows = []
    for i in range(0, len(todo), 100):
        res = requests.post("https://api.postcodes.io/postcodes", json={"postcodes": todo[i:i + 100]}, timeout=60)
        for item in res.json()["result"]:
            r = item["result"] or {}
            rows.append({"postcode": item["query"], "lat": r.get("latitude"), "lon": r.get("longitude")})
    coords = pd.DataFrame(rows)
    coords.to_csv(cache, index=False)
    return coords


def xy(df: pd.DataFrame) -> np.ndarray:
    return np.c_[df.lat, df.lon * 0.61]  # cos(52.5 deg): makes a degree of longitude comparable to latitude


def neighbour_price(nn, train_logp: np.ndarray, points: np.ndarray, exclude_self: bool) -> np.ndarray:
    """Mean log price of the K nearest training sales. For training rows the sale itself is left out."""
    idx = nn.kneighbors(points, n_neighbors=K + exclude_self, return_distance=False)
    keep = np.ones(idx.shape, dtype=bool)
    if exclude_self:
        keep = idx != np.arange(len(points))[:, None]
        keep[keep.all(axis=1), -1] = False  # self not among the neighbours (ties): drop the farthest instead
    return (train_logp[idx] * keep).sum(axis=1) / K


def baseline_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Median training price for the same outcode and property type, falling back to type, then overall."""
    by_both = train.groupby(["outcode", "type"]).price.median().rename("b1")
    by_type = train.groupby("type").price.median().rename("b2")
    out = test.join(by_both, on=["outcode", "type"]).join(by_type, on="type")
    return out.b1.fillna(out.b2).fillna(train.price.median()).to_numpy()


def to_X(df: pd.DataFrame, cats: dict) -> pd.DataFrame:
    X = df[CAT + NUM].copy()
    for c in CAT:  # categories unseen in training become missing, which the model handles natively
        X[c] = pd.Categorical(X[c], categories=cats[c])
    return X


def score(y, p) -> dict:
    ape = np.abs(p - y) / y
    return {"mae_gbp": round(float(np.mean(np.abs(p - y)))),
            "median_ape_pct": round(float(np.median(ape)) * 100, 1),
            "within_20pct_share": round(float(np.mean(ape <= 0.2)) * 100, 1),
            "mean_error_gbp": round(float(np.mean(p - y)))}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = clean(load_sales())
    df = df.merge(postcode_coords(df.postcode.unique()), on="postcode", how="left").dropna(subset=["lat", "lon"])
    train = df[df.date.dt.year.isin(TRAIN_YEARS)].copy()
    test = df[df.date.dt.year == TEST_YEAR].copy()
    assert len(train) and len(test), "need both training and test years"

    nn = NearestNeighbors().fit(xy(train))
    train_logp = np.log(train.price.to_numpy())
    train["neighbour_logp"] = neighbour_price(nn, train_logp, xy(train), exclude_self=True)
    test["neighbour_logp"] = neighbour_price(nn, train_logp, xy(test), exclude_self=False)

    cats = {c: sorted(train[c].dropna().unique()) for c in CAT}
    model = HistGradientBoostingRegressor(categorical_features="from_dtype", max_iter=500,
                                          learning_rate=0.06, early_stopping=True, random_state=0)
    model.fit(to_X(train, cats), np.log(train.price))

    y = test.price.to_numpy()
    pred = np.exp(model.predict(to_X(test, cats)))
    base = baseline_predict(train, test)

    sample = test.sample(min(5000, len(test)), random_state=0)
    imp = permutation_importance(model, to_X(sample, cats), np.log(sample.price),
                                 n_repeats=5, random_state=0, scoring="neg_mean_absolute_error")
    importance = pd.Series(imp.importances_mean, index=CAT + NUM).sort_values()

    by_type = {t: {"n": int(m.sum()), "model": score(y[m], pred[m]), "baseline": score(y[m], base[m])}
               for t in sorted(test.type.unique()) for m in [(test.type == t).to_numpy()]}
    metrics = {
        "train_rows": len(train), "test_rows": len(test),
        "train_period": f"{train.date.min():%Y-%m-%d} to {train.date.max():%Y-%m-%d}",
        "test_period": f"{test.date.min():%Y-%m-%d} to {test.date.max():%Y-%m-%d}",
        "model": score(y, pred), "baseline": score(y, base), "by_property_type": by_type,
        "permutation_importance_log_mae": importance.round(4).to_dict(),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(y, pred, s=3, alpha=0.2)
    ax[0].plot([3e4, 2e6], [3e4, 2e6], color="black", lw=1)
    ax[0].set(xscale="log", yscale="log", xlabel="Actual price (GBP)", ylabel="Predicted price (GBP)",
              title=f"Out-of-time test, {TEST_YEAR} sales")
    importance.plot.barh(ax=ax[1])
    ax[1].set(xlabel="Increase in log-price MAE when shuffled", title="Permutation importance")
    fig.tight_layout()
    fig.savefig(REPORTS / "model_results.png", dpi=130)

    lookup = train.groupby("outcode").agg(district=("district", lambda s: s.mode()[0]),
                                          lat=("lat", "mean"), lon=("lon", "mean"))
    lookup["neighbour_logp"] = neighbour_price(nn, train_logp, xy(lookup), exclude_self=False)
    joblib.dump({"model": model, "cats": cats, "lookup": lookup,
                 "latest_month_idx": int(train.month_idx.max())}, DATA / "model.joblib")
    print(json.dumps({k: metrics[k] for k in ("train_rows", "test_rows", "test_period", "model", "baseline")}, indent=2))


if __name__ == "__main__":
    main()
