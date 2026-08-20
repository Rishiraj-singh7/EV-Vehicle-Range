"""Train a range model dedicated to the Tata GPS fleet
(data/processed/tata_epochs.csv, built by build_epochs.py). Kept as a
separate model from the Intellicar one (different sensors, sampling rates,
and vehicle types -- see the discussion in chat for why).

Candidates: Decision Tree, Random Forest, Gradient Boosting, plus Linear
Regression as a baseline -- same selection approach as the Intellicar
trainer, see its docstring for the reasoning.

Target: implied_range_km = distance_km / soc_used * 100
Features: avg_speed, max_speed, pct_ac_on, avg_accel_xy, max_accel_xy,
          avg_altitude, altitude_range, driving_minutes, odometer_start,
          vehicle (id)

distance_km/soc_used are excluded as features for the same reason as the
other pipeline: they define the target, so using them directly would let
the model memorize the label instead of learning from conditions.
soc_start/soc_end are excluded too -- same extrapolation risk as before.
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

from src.common.paths import TATA_EPOCHS_PATH, TATA_MODEL_PATH, TATA_VEHICLE_RANGE_PATH

NUMERIC_FEATURES = [
    "avg_speed",
    "max_speed",
    "pct_ac_on",
    "avg_accel_xy",
    "max_accel_xy",
    "avg_altitude",
    "altitude_range",
    "driving_minutes",
    "odometer_start",
]
CATEGORICAL_FEATURES = ["vehicle"]
TARGET = "implied_range_km"


def make_pipeline(model):
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def evaluate(name, pipeline, X, y, w, groups):
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
    df = pd.read_csv(TATA_EPOCHS_PATH)
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
    for name, model in candidates.items():
        pipeline = make_pipeline(model)
        results[name] = evaluate(name, pipeline, X, y, w, groups)

    best_name = min(results, key=lambda n: results[n][0])
    print(f"\nBest model by random-split MAE: {best_name}")

    best_pipeline = make_pipeline(candidates[best_name])
    best_pipeline.fit(X, y, model__sample_weight=w)

    TATA_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET,
            "model_name": best_name,
            "n_epochs_trained_on": len(df),
        },
        TATA_MODEL_PATH,
    )
    print(f"Saved trained model to {TATA_MODEL_PATH}")

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

    print("\nPredicted full range per vehicle (typical driving conditions):")
    rows = []
    for vehicle, grp in df.groupby("vehicle"):
        rows.append(
            {
                "vehicle": vehicle,
                "avg_speed": grp["avg_speed"].median(),
                "max_speed": grp["max_speed"].median(),
                "pct_ac_on": grp["pct_ac_on"].median(),
                "avg_accel_xy": grp["avg_accel_xy"].median(),
                "max_accel_xy": grp["max_accel_xy"].median(),
                "avg_altitude": grp["avg_altitude"].median(),
                "altitude_range": grp["altitude_range"].median(),
                "driving_minutes": grp["driving_minutes"].median(),
                "odometer_start": grp["odometer_start"].median(),
            }
        )
    query = pd.DataFrame(rows)
    preds = best_pipeline.predict(query[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    query["predicted_range_km"] = preds.round(1)
    query["n_epochs"] = df.groupby("vehicle").size().values
    query["observed_median_range_km"] = df.groupby("vehicle")[TARGET].median().values.round(1)

    out = (
        query[["vehicle", "n_epochs", "observed_median_range_km", "predicted_range_km"]]
        .sort_values("predicted_range_km", ascending=False)
    )
    print(out.to_string(index=False))
    TATA_VEHICLE_RANGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TATA_VEHICLE_RANGE_PATH, index=False)
    print(f"\nSaved per-vehicle range table to {TATA_VEHICLE_RANGE_PATH}")


if __name__ == "__main__":
    main()
