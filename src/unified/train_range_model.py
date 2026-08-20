"""Train ONE range model across all three telematics devices, from
data/processed/unified_epochs.csv.

Replaces the per-device models (src/intellicar, src/tata) with a single
pipeline whose features come from data/processed/unified_feature_ranking.csv
-- the top 8 the selection script picked -- plus `device` and `vehicle` as
categoricals. Those two carry what the 8 numeric features deliberately do
not: the fleet's battery size and vehicle class, which set the *level* of
range, while the numeric features explain variation around that level.

Target: implied_range_km = distance_km / soc_used * 100
Excluded on purpose: distance_km and soc_used (they define the target),
soc_start/soc_end (extrapolation risk -- see unified_features.py).

The script does not just fit a merged model, it checks whether merging was
the right call at all: for each device it also fits a model on that device's
rows alone, using the identical feature set and evaluation, and prints the
two side by side. Merging is only a win if the unified model is at least
competitive per fleet -- an average that hides a regression on the two small
fleets would not be.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from src.common.paths import (
    UNIFIED_EPOCHS_PATH,
    UNIFIED_FEATURE_RANKING_PATH,
    UNIFIED_MODEL_PATH,
    UNIFIED_VEHICLE_RANGE_PATH,
)

CATEGORICAL_FEATURES = ["device", "vehicle"]
TARGET = "implied_range_km"
RNG = 42


def load_selected_features():
    rank = pd.read_csv(UNIFIED_FEATURE_RANKING_PATH, index_col=0)
    feats = list(rank[rank["selected"]].sort_values("selection_order").index)
    if not feats:
        raise SystemExit("No features marked selected -- run select_features.py first.")
    return feats


def make_pipeline(model, numeric_features, categorical_features):
    pre = ColumnTransformer(
        [
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def candidates():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=5, random_state=RNG
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=RNG
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RNG
        ),
    }


def device_balanced_weights(df):
    """soc_used weighting (a bigger SOC drop is a more reliable measurement)
    scaled so each device contributes equally in total.

    Without the second factor the pooled fit is 85% Intellicar and simply
    learns that fleet, which is the main way a naive merge goes wrong."""
    w = df["soc_used"].astype(float).copy()
    per_device_total = df.groupby("device")["soc_used"].transform("sum")
    n_devices = df["device"].nunique()
    return w * (w.sum() / (n_devices * per_device_total))


def grouped_cv_predictions(pipeline, X, y, w, groups, n_splits=5):
    """Out-of-fold predictions with vehicles held out entirely -- the honest
    read on a vehicle the model has never seen. A random split leaks, since
    `vehicle` is itself a feature."""
    oof = np.full(len(X), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        pipeline.fit(X.iloc[tr], y.iloc[tr], model__sample_weight=w.iloc[tr])
        oof[te] = pipeline.predict(X.iloc[te])
    return oof


def main():
    numeric_features = load_selected_features()
    df = pd.read_csv(UNIFIED_EPOCHS_PATH)

    print(f"Unified table: {len(df)} epochs, {df['vehicle'].nunique()} vehicles, "
          f"{df['device'].nunique()} devices")
    print(f"Top-8 features: {numeric_features}")
    print(f"Categoricals:   {CATEGORICAL_FEATURES}\n")
    print(df.groupby("device").agg(
        epochs=(TARGET, "size"), vehicles=("vehicle", "nunique"),
        median_range=(TARGET, "median"),
    ).round(1).to_string())

    X = df[numeric_features + CATEGORICAL_FEATURES]
    y = df[TARGET]
    w = device_balanced_weights(df)
    groups = df["vehicle"]

    print("\n" + "=" * 66)
    print("Model selection (vehicle-disjoint 5-fold, MAE in km)")
    print("=" * 66)

    results = {}
    for name, model in candidates().items():
        pipe = make_pipeline(model, numeric_features, CATEGORICAL_FEATURES)
        oof = grouped_cv_predictions(pipe, X, y, w, groups)
        mae = mean_absolute_error(y, oof, sample_weight=w)
        r2 = r2_score(y, oof, sample_weight=w)
        per_dev = {
            d: mean_absolute_error(y[m], oof[m], sample_weight=w[m])
            for d, m in ((d, df["device"] == d) for d in sorted(df["device"].unique()))
        }
        results[name] = (mae, r2, per_dev)
        dev_str = "  ".join(f"{d}={v:.1f}" for d, v in per_dev.items())
        print(f"{name:20s} MAE={mae:6.1f}  R2={r2:6.3f}   [{dev_str}]")

    best_name = min(results, key=lambda n: results[n][0])
    print(f"\nBest: {best_name}")

    # --- is merging actually better than one model per device? -----------
    print("\n" + "=" * 66)
    print("Merged vs per-device, same features, same vehicle-disjoint CV")
    print("=" * 66)
    print(f"{'device':<12}{'epochs':>8}{'merged':>10}{'device-only':>14}{'winner':>10}")

    comparison = {}
    for dev in sorted(df["device"].unique()):
        mask = (df["device"] == dev).values
        merged_mae = results[best_name][2][dev]

        sub = df[mask]
        if sub["vehicle"].nunique() < 5:
            print(f"{dev:<12}{mask.sum():>8}{merged_mae:>10.1f}{'n/a':>14}{'-':>10}"
                  f"   (<5 vehicles, can't CV alone)")
            comparison[dev] = (merged_mae, None)
            continue

        pipe = make_pipeline(
            candidates()[best_name], numeric_features, ["vehicle"]
        )
        oof_d = grouped_cv_predictions(
            pipe, sub[numeric_features + ["vehicle"]], sub[TARGET],
            sub["soc_used"], sub["vehicle"],
            n_splits=min(5, sub["vehicle"].nunique()),
        )
        solo_mae = mean_absolute_error(sub[TARGET], oof_d, sample_weight=sub["soc_used"])
        winner = "merged" if merged_mae <= solo_mae else "separate"
        comparison[dev] = (merged_mae, solo_mae)
        print(f"{dev:<12}{mask.sum():>8}{merged_mae:>10.1f}{solo_mae:>14.1f}{winner:>10}")

    # --- fit final model on everything -----------------------------------
    best_pipeline = make_pipeline(
        candidates()[best_name], numeric_features, CATEGORICAL_FEATURES
    )
    best_pipeline.fit(X, y, model__sample_weight=w)

    UNIFIED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "numeric_features": numeric_features,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET,
            "model_name": best_name,
            "n_epochs_trained_on": len(df),
            "devices": sorted(df["device"].unique()),
            "vehicles": sorted(df["vehicle"].unique()),
            "cv_mae_by_device": results[best_name][2],
        },
        UNIFIED_MODEL_PATH,
    )
    print(f"\nSaved unified model to {UNIFIED_MODEL_PATH}")

    model = best_pipeline.named_steps["model"]
    feat_names = numeric_features + list(
        best_pipeline.named_steps["pre"]
        .named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    if hasattr(model, "feature_importances_"):
        imps = sorted(zip(feat_names, model.feature_importances_), key=lambda t: -t[1])
        print("\nTop feature importances (numeric features only):")
        for f, i in [t for t in imps if t[0] in numeric_features][:10]:
            print(f"  {f:22s} {i:.3f}")
        cat_total = sum(i for f, i in imps if f not in numeric_features)
        print(f"  [device+vehicle identity, combined] {cat_total:.3f}")

    # --- per-vehicle predicted full range --------------------------------
    rows = []
    for (dev, veh), grp in df.groupby(["device", "vehicle"]):
        row = {f: grp[f].median() for f in numeric_features}
        row["device"], row["vehicle"] = dev, veh
        rows.append(row)
    query = pd.DataFrame(rows)
    query["predicted_range_km"] = best_pipeline.predict(
        query[numeric_features + CATEGORICAL_FEATURES]
    ).round(1)

    obs = df.groupby(["device", "vehicle"]).agg(
        n_epochs=(TARGET, "size"), observed_median_range_km=(TARGET, "median")
    ).reset_index()
    out = query.merge(obs, on=["device", "vehicle"])[
        ["device", "vehicle", "n_epochs", "observed_median_range_km", "predicted_range_km"]
    ].round(1).sort_values(["device", "predicted_range_km"], ascending=[True, False])

    UNIFIED_VEHICLE_RANGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(UNIFIED_VEHICLE_RANGE_PATH, index=False)
    print(f"\nPer-vehicle range table -> {UNIFIED_VEHICLE_RANGE_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
