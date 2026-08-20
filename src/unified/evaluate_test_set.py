"""Score the unified model against a ground-truth range table
(data/reference/test_actual_range.csv) and report where it is wrong.

Three views, because no single number answers "is this good enough":

1. **Regression metrics** (MAE / RMSE / MAPE / bias / R2). Range is a
   continuous quantity, so these are the metrics to actually tune against.
   `bias` is the one to read first: it separates "noisy but centred" from
   "systematically reading high/low", and only the latter is fixable by
   recalibration.

2. **Banded precision / recall.** Precision and recall are classification
   metrics and do not exist for a regression until predictions are bucketed,
   so vehicles are placed in Short / Medium / Long bands (cut on the
   *actual* range terciles) and the model is scored on whether it lands each
   vehicle in the right band. This answers "does it rank vehicles correctly".

3. **Operational threshold sweep.** The real dispatch question is "can this
   vehicle cover a route of length L?", so for each L the positive class is
   "range >= L":
       precision = of the vehicles the model cleared, how many really could
                   (low precision => vehicles stranded mid-route)
       recall    = of the vehicles that really could, how many it cleared
                   (low recall => vehicles idled unnecessarily)
   Which of the two matters more is an operational call, not a modelling one.

**In-sample vs held-out** is reported separately throughout, and only the
held-out block means anything for generalisation: a vehicle the model
trained on has its own one-hot column and is partly memorised.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
)

from src.common.paths import (
    PROCESSED_DIR,
    REFERENCE_DIR,
    UNIFIED_EPOCHS_PATH,
    UNIFIED_MODEL_PATH,
)

TEST_PATH = REFERENCE_DIR / "test_actual_range.csv"
RESULTS_PATH = PROCESSED_DIR / "test_set_predictions.csv"
BAND_LABELS = ["Short", "Medium", "Long"]


def regression_metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - actual
    return {
        "n": len(actual),
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE%": float(np.mean(np.abs(err / actual)) * 100),
        "bias": float(np.mean(err)),
        "R2": r2_score(actual, pred) if len(actual) > 1 else float("nan"),
        "corr": float(np.corrcoef(actual, pred)[0, 1]) if len(actual) > 1 else float("nan"),
        "within_10km%": float(np.mean(np.abs(err) <= 10) * 100),
        "within_20km%": float(np.mean(np.abs(err) <= 20) * 100),
    }


def print_metrics_block(title, m):
    print(f"\n{title}  (n={m['n']})")
    print(f"  MAE   {m['MAE']:7.1f} km      RMSE  {m['RMSE']:7.1f} km")
    print(f"  MAPE  {m['MAPE%']:7.1f} %       bias  {m['bias']:+7.1f} km")
    print(f"  R2    {m['R2']:7.3f}         corr  {m['corr']:7.3f}")
    print(f"  within +/-10 km: {m['within_10km%']:5.1f} %   within +/-20 km: {m['within_20km%']:5.1f} %")


def banded_report(df, title):
    """Bucket on the ACTUAL range terciles, then score the predictions
    against it. Cutting on actual (not predicted) keeps the class definition
    fixed and independent of the model."""
    edges = df["actual_range_km"].quantile([1 / 3, 2 / 3]).values
    y_true = np.digitize(df["actual_range_km"].values, edges)
    y_pred = np.digitize(df["predicted_range_km"].values, edges)

    print(f"\n{title}")
    print(f"  band edges (actual terciles): <{edges[0]:.0f} | {edges[0]:.0f}-{edges[1]:.0f} | >{edges[1]:.0f} km")
    labels = sorted(set(y_true) | set(y_pred))
    names = [BAND_LABELS[i] for i in labels]
    print(classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("  confusion matrix (rows=actual, cols=predicted):")
    print("        " + "".join(f"{n:>9}" for n in names))
    for n, row in zip(names, cm):
        print(f"  {n:>6}" + "".join(f"{v:>9}" for v in row))


def threshold_sweep(df, thresholds):
    print("\nPositive class = 'can cover a route of L km'")
    print(f"  {'L (km)':>7}{'actual+':>9}{'pred+':>7}{'precision':>11}{'recall':>8}{'F1':>7}"
          f"{'stranded':>10}{'idled':>7}")
    rows = []
    for L in thresholds:
        yt = (df["actual_range_km"] >= L).astype(int)
        yp = (df["predicted_range_km"] >= L).astype(int)
        if yt.nunique() < 2:
            continue
        p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
        stranded = int(((yp == 1) & (yt == 0)).sum())  # cleared but couldn't make it
        idled = int(((yp == 0) & (yt == 1)).sum())     # held back but could have
        print(f"  {L:>7}{yt.sum():>9}{yp.sum():>7}{p:>11.3f}{r:>8.3f}{f1:>7.3f}{stranded:>10}{idled:>7}")
        rows.append({"threshold_km": L, "precision": p, "recall": r, "f1": f1,
                     "false_cleared": stranded, "false_held": idled})
    return pd.DataFrame(rows)


def calibration_diagnostic(scored, epochs):
    """Separates three different things that all show up as "error":

      * random noise            -> nothing to fix but more/cleaner data
      * a constant offset       -> fixable by shifting the output
      * a target-definition gap -> not a model problem at all

    The last one is the reason this function compares the supplied actual
    range against the *raw telemetry's own* implied range as well as against
    the model. If the untouched data already sits below the supplied
    numbers, no amount of retuning will close that gap -- the two are
    measuring different things."""
    a = scored["actual_range_km"].values.astype(float)
    p = scored["predicted_range_km"].values.astype(float)

    print(f"  spread: actual std {a.std():.1f} km vs predicted std {p.std():.1f} km "
          f"(ratio {p.std() / a.std():.2f}; <1 means predictions are compressed)")
    held = scored[~scored["in_training"]]
    if len(held) > 1:
        ah, ph = held["actual_range_km"].values, held["predicted_range_km"].values
        print(f"  held out only: ratio {ph.std() / ah.std():.2f}, "
              f"predicted span {ph.min():.0f}-{ph.max():.0f} km vs actual {ah.min():.0f}-{ah.max():.0f} km")

    print("\n  Error by actual-range band -- a flat column here means a constant")
    print("  offset; a trending one means the model compresses the range.")
    band = pd.cut(scored["actual_range_km"], [0, 160, 180, 200, 220, 400])
    by_band = scored.groupby(band, observed=True).agg(
        n=("error_km", "size"), mean_error_km=("error_km", "mean")
    ).round(1)
    print(by_band.to_string())

    print("\n  What recalibration would buy (fit here for illustration -- fit it on a")
    print("  separate validation split before shipping, or it will flatter itself):")
    for label, sub in [("all scored", scored), ("held out", held)]:
        if len(sub) < 3:
            continue
        aa = sub["actual_range_km"].values.astype(float)
        pp = sub["predicted_range_km"].values.astype(float)
        slope, intercept = np.polyfit(pp, aa, 1)
        print(f"    {label:11s} raw {mean_absolute_error(aa, pp):5.1f} km"
              f" | +bias shift {mean_absolute_error(aa, pp - (pp - aa).mean()):5.1f} km"
              f" | +affine {mean_absolute_error(aa, slope * pp + intercept):5.1f} km"
              f"  (slope {slope:.2f}, intercept {intercept:+.0f})")

    # The decisive check: what does the untouched telemetry say, with no model?
    obs = epochs.groupby("vehicle")["implied_range_km"].median()
    j = scored[["vehicle", "actual_range_km", "error_km"]].join(
        obs.rename("observed_median_range_km"), on="vehicle"
    ).dropna()
    if len(j):
        raw_gap = (j["observed_median_range_km"] - j["actual_range_km"]).mean()
        print(f"\n  Raw telemetry (no model): observed median implied range sits "
              f"{raw_gap:+.1f} km\n  from the supplied actual range; the model sits "
              f"{scored['error_km'].mean():+.1f} km from it.")
        if raw_gap < -5:
            print("  => The training labels themselves run below the supplied ground truth,")
            print("     so this is mostly a target-definition gap, not a model defect.")


def main():
    test = pd.read_csv(TEST_PATH)
    epochs = pd.read_csv(UNIFIED_EPOCHS_PATH)
    bundle = joblib.load(UNIFIED_MODEL_PATH)
    pipeline = bundle["pipeline"]
    feats = bundle["numeric_features"]
    cats = bundle["categorical_features"]
    trained_vehicles = set(bundle["vehicles"])

    # Per-vehicle "typical conditions" = median of that vehicle's epochs --
    # the same query the per-vehicle range table is built from.
    present = epochs[epochs["vehicle"].isin(test["vehicle"])]
    if present.empty:
        raise SystemExit("No test vehicles have epoch data -- run build_epochs.py first.")

    grp = present.groupby(["device", "vehicle"])
    query = grp[feats].median().reset_index()
    query["n_epochs"] = grp.size().values
    query["predicted_range_km"] = pipeline.predict(query[feats + cats])

    df = test.merge(query, on="vehicle", how="left")
    df["in_training"] = df["vehicle"].isin(trained_vehicles)
    df["error_km"] = df["predicted_range_km"] - df["actual_range_km"]
    df["abs_error_km"] = df["error_km"].abs()

    scored = df.dropna(subset=["predicted_range_km"]).copy()
    unscorable = df[df["predicted_range_km"].isna()]

    print("=" * 72)
    print("TEST-SET EVALUATION -- unified model vs supplied actual ranges")
    print("=" * 72)
    print(f"Test vehicles supplied : {len(df)}")
    print(f"  scored               : {len(scored)}")
    print(f"  no telemetry, skipped: {len(unscorable)}")
    if len(unscorable):
        print("    " + ", ".join(unscorable["vehicle"]))
    print(f"\nOf the {len(scored)} scored:")
    print(f"  seen in training (in-sample, optimistic): {scored['in_training'].sum()}")
    print(f"  never seen (held out, the honest number): {(~scored['in_training']).sum()}")

    print("\n" + "=" * 72)
    print("1. REGRESSION METRICS  (the ones to tune against)")
    print("=" * 72)
    print_metrics_block("ALL SCORED", regression_metrics(scored["actual_range_km"], scored["predicted_range_km"]))
    for flag, label in [(True, "IN-SAMPLE (trained on these vehicles)"),
                        (False, "HELD OUT (never seen -- the real test)")]:
        sub = scored[scored["in_training"] == flag]
        if len(sub) > 1:
            print_metrics_block(label, regression_metrics(sub["actual_range_km"], sub["predicted_range_km"]))
    for dev, g in scored.groupby("device"):
        if len(g) > 1:
            print_metrics_block(f"device: {dev}", regression_metrics(g["actual_range_km"], g["predicted_range_km"]))

    print("\n" + "=" * 72)
    print("2. BANDED PRECISION / RECALL  (regression bucketed into 3 bands)")
    print("=" * 72)
    banded_report(scored, "ALL SCORED")
    held = scored[~scored["in_training"]]
    if len(held) >= 6:
        banded_report(held, "HELD OUT ONLY")

    print("\n" + "=" * 72)
    print("3. OPERATIONAL THRESHOLD SWEEP")
    print("=" * 72)
    lo, hi = scored["actual_range_km"].quantile([0.1, 0.9])
    thresholds = sorted({int(round(t / 5) * 5) for t in np.linspace(lo, hi, 8)})
    threshold_sweep(scored, thresholds)

    print("\n" + "=" * 72)
    print("4. CALIBRATION -- is the error noise, or a systematic offset?")
    print("=" * 72)
    calibration_diagnostic(scored, epochs)

    print("\n" + "=" * 72)
    print("WORST PREDICTIONS (largest absolute error)")
    print("=" * 72)
    cols = ["vehicle", "device", "n_epochs", "in_training",
            "actual_range_km", "predicted_range_km", "error_km"]
    print(scored.nlargest(12, "abs_error_km")[cols].round(1).to_string(index=False))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored[cols + ["abs_error_km"]].round(2).sort_values(
        "abs_error_km", ascending=False
    ).to_csv(RESULTS_PATH, index=False)
    print(f"\nPer-vehicle results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
