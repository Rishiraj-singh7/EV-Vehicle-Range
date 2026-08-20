"""Train a model that predicts a vehicle's full range (km at 100% SOC) from
per-epoch driving conditions, using data/processed/intellicar_epochs.csv
(built by build_epochs.py).

Candidates: Decision Tree, Random Forest, Gradient Boosting (all
tree-based, per the product decision), plus Linear Regression kept as a
sanity-check baseline. Whichever scores the lowest MAE on held-out data is
selected and saved -- see evaluate() below.

Target: implied_range_km = distance_km / soc_used * 100
Features: avg_speed, max_speed, avg_batt_temp, duration_hours,
          odometer_start (mileage/age proxy), vehicle (id)

distance_km and soc_used themselves are NOT features -- they define the
target, so including them would make the model trivially memorize the
label instead of learning how conditions (speed, temperature, vehicle age)
affect range. Epochs are weighted by soc_used during training/scoring so
epochs with a bigger, more reliable SOC drop count for more.

soc_start/soc_end are also excluded even though efficiency does vary by
SOC band in real EVs: almost none of the training epochs actually span
close to 100%->0% (data is nearly all partial discharges in the 25-95%
band), so a model that leans on absolute SOC level would be extrapolating
wildly right at the operating point we care about (predicting full
0-100% range) -- which is exactly what happened in an earlier version of
this script (it produced negative "ranges" for several vehicles).
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
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from src.common.paths import (
    INTELLICAR_EPOCHS_PATH,
    INTELLICAR_MODEL_PATH,
    INTELLICAR_VEHICLE_RANGE_PATH,
)

NUMERIC_FEATURES = [
    "avg_speed",
    "max_speed",
    "avg_batt_temp",
    "duration_hours",
    "odometer_start",
]
CATEGORICAL_FEATURES = ["vehicle"]
TARGET = "implied_range_km"


def load_data():
    df = pd.read_csv(INTELLICAR_EPOCHS_PATH)
    df["avg_batt_temp"] = df["avg_batt_temp"].fillna(df["avg_batt_temp"].median())
    return df


def make_pipeline(model):
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def evaluate(name, pipeline, X, y, w, groups):
    """Two views: random split (how well it fits this fleet's data) and
    group k-fold by vehicle (how well it'd generalize to an unseen vehicle,
    since 'vehicle' is a feature that a new vehicle wouldn't match)."""
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.25, random_state=42
    )
    pipeline.fit(X_train, y_train, model__sample_weight=w_train)
    pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, pred, sample_weight=w_test)
    r2 = r2_score(y_test, pred, sample_weight=w_test)

    gkf = GroupKFold(n_splits=5)
    group_maes = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        pipeline.fit(
            X.iloc[train_idx], y.iloc[train_idx], model__sample_weight=w.iloc[train_idx]
        )
        p = pipeline.predict(X.iloc[test_idx])
        group_maes.append(
            mean_absolute_error(y.iloc[test_idx], p, sample_weight=w.iloc[test_idx])
        )

    print(f"\n=== {name} ===")
    print(f"Random-split: MAE={mae:.1f} km, R^2={r2:.3f}")
    print(f"GroupKFold-by-vehicle: MAE={np.mean(group_maes):.1f} km (+/- {np.std(group_maes):.1f})")
    return mae, r2


def main():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    w = df["soc_used"]
    groups = df["vehicle"]

    print(f"Training on {len(df)} epochs across {df['vehicle'].nunique()} vehicles")
    print(f"Target range: min={y.min():.0f} km, median={y.median():.0f} km, max={y.max():.0f} km")

    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=5, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }

    results = {}
    pipelines = {}
    for name, model in candidates.items():
        pipeline = make_pipeline(model)
        results[name] = evaluate(name, pipeline, X, y, w, groups)
        pipelines[name] = pipeline

    best_name = min(results, key=lambda n: results[n][0])
    print(f"\nBest model by random-split MAE: {best_name}")

    # Refit best model on all data for saving/inference.
    best_pipeline = make_pipeline(candidates[best_name])
    best_pipeline.fit(X, y, model__sample_weight=w)

    INTELLICAR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET,
            "model_name": best_name,
            "n_epochs_trained_on": len(df),
        },
        INTELLICAR_MODEL_PATH,
    )
    print(f"Saved trained model to {INTELLICAR_MODEL_PATH}")

    # Feature importance / coefficients for the chosen model, where available.
    model = best_pipeline.named_steps["model"]
    feat_names = NUMERIC_FEATURES + list(
        best_pipeline.named_steps["pre"]
        .named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    if hasattr(model, "feature_importances_"):
        importances = sorted(zip(feat_names, model.feature_importances_), key=lambda t: -t[1])
        print("\nTop feature importances:")
        for f, imp in importances[:10]:
            print(f"  {f}: {imp:.3f}")
    elif hasattr(model, "coef_"):
        coefs = sorted(zip(feat_names, model.coef_), key=lambda t: -abs(t[1]))
        print("\nTop coefficients:")
        for f, c in coefs[:10]:
            print(f"  {f}: {c:.2f}")

    # Per-vehicle predicted range at "typical" conditions (median speed/temp
    # observed for that vehicle, full SOC window 100 -> 0).
    print("\nPredicted full range per vehicle (typical driving conditions):")
    rows = []
    for vehicle, grp in df.groupby("vehicle"):
        row = {
            "vehicle": vehicle,
            "avg_speed": grp["avg_speed"].median(),
            "max_speed": grp["max_speed"].median(),
            "avg_batt_temp": grp["avg_batt_temp"].median(),
            "duration_hours": grp["duration_hours"].median(),
            "odometer_start": grp["odometer_start"].median(),
        }
        rows.append(row)
    query = pd.DataFrame(rows)
    preds = best_pipeline.predict(query[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    query["predicted_range_km"] = preds
    query["n_epochs"] = df.groupby("vehicle").size().values
    query["observed_median_range_km"] = df.groupby("vehicle")[TARGET].median().values
    out = (
        query[["vehicle", "n_epochs", "observed_median_range_km", "predicted_range_km"]]
        .sort_values("predicted_range_km", ascending=False)
        .round(1)
    )
    print(out.to_string(index=False))

    INTELLICAR_VEHICLE_RANGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(INTELLICAR_VEHICLE_RANGE_PATH, index=False)
    print(f"\nSaved per-vehicle range table to {INTELLICAR_VEHICLE_RANGE_PATH}")


if __name__ == "__main__":
    main()
