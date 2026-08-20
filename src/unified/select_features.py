"""Rank the unified candidate features and pick the top 8 for the merged
range model.

"Most correlated with the target" on the pooled table is not a safe enough
criterion on its own here, for two reasons specific to this dataset:

  1. **The pool is 85% Intellicar.** A feature that only works for that
     fleet would top a pooled ranking while being useless -- or actively
     harmful -- on the other two.
  2. **Sign flips.** odometer_start correlates +0.31 with range on the
     Intellicar fleet and ~0.00 on Tata, because the fleets are at totally
     different points in their life (median odometer ~96k km vs ~6k km).
     Pooled, that reads as signal; per-fleet, it's two unrelated effects.

So each candidate is scored on four things, and the report prints all of
them rather than collapsing to a single number behind the reader's back:

    |rho|         Spearman vs implied_range_km, pooled
    consistency   fraction of devices whose per-device rho has the pooled sign
    MI            mutual information (captures non-monotonic structure rho misses)
    importance    permutation importance in a GradientBoosting fit, which is
                  the only measure here that accounts for redundancy between
                  features

Redundancy matters because three of the candidates (moving_hours,
active_hours, pct_time_moving) are near-restatements of each other; a
correlation ranking would happily take all three and spend three of the
eight slots on one underlying signal.

Output: data/processed/unified_feature_ranking.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit

from src.common.paths import UNIFIED_EPOCHS_PATH, UNIFIED_FEATURE_RANKING_PATH
from src.common.unified_features import CANDIDATE_FEATURES

TARGET = "implied_range_km"
N_KEEP = 8
RNG = 42


def spearman_table(df):
    """Pooled rho, plus per-device rho and how consistently they agree."""
    rows = {}
    pooled = df[CANDIDATE_FEATURES + [TARGET]].corr(method="spearman")[TARGET]
    for f in CANDIDATE_FEATURES:
        per_device = {}
        for dev, grp in df.groupby("device"):
            if len(grp) < 30:
                continue
            per_device[dev] = grp[f].corr(grp[TARGET], method="spearman")
        p = pooled[f]
        agree = [v for v in per_device.values() if not np.isnan(v)]
        consistency = (
            sum(np.sign(v) == np.sign(p) for v in agree) / len(agree) if agree else 0.0
        )
        rows[f] = {
            "rho_pooled": p,
            "consistency": consistency,
            **{f"rho_{d}": v for d, v in per_device.items()},
        }
    return pd.DataFrame(rows).T


def greedy_mrmr(relevance, feat_corr, n_keep, redundancy_weight=0.7):
    """Minimum-redundancy / maximum-relevance selection.

    Ranking on relevance alone is the wrong tool here: the candidates are
    heavily collinear (avg_speed vs pct_time_highway is 0.96), so the top 8
    by score spend five slots restating one speed signal and leave genuinely
    independent signals unselected. Each pick here is scored on its own
    relevance minus its mean absolute correlation with what's already been
    taken, so the second speed-shaped feature has to beat that penalty to
    earn a slot."""
    remaining = list(relevance.index)
    chosen = [relevance.idxmax()]
    remaining.remove(chosen[0])

    while len(chosen) < n_keep and remaining:
        best, best_val = None, -np.inf
        for f in remaining:
            redundancy = feat_corr.loc[f, chosen].mean()
            val = relevance[f] - redundancy_weight * redundancy
            if val > best_val:
                best, best_val = f, val
        chosen.append(best)
        remaining.remove(best)
    return chosen


def main():
    df = pd.read_csv(UNIFIED_EPOCHS_PATH)
    X = df[CANDIDATE_FEATURES]
    y = df[TARGET]

    print(f"Ranking {len(CANDIDATE_FEATURES)} candidates on {len(df)} epochs "
          f"({df['device'].nunique()} devices, {df['vehicle'].nunique()} vehicles)\n")

    rank = spearman_table(df)

    rank["mutual_info"] = mutual_info_regression(X, y, random_state=RNG)

    # Permutation importance on a vehicle-disjoint split: a random split
    # would let the model memorise per-vehicle quirks and inflate every
    # feature's apparent worth.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RNG)
    tr, te = next(gss.split(X, y, groups=df["vehicle"]))
    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RNG
    ).fit(X.iloc[tr], y.iloc[tr], sample_weight=df["soc_used"].iloc[tr])
    perm = permutation_importance(
        gb, X.iloc[te], y.iloc[te], n_repeats=10, random_state=RNG,
        sample_weight=df["soc_used"].iloc[te],
    )
    rank["perm_importance"] = perm.importances_mean

    # Composite: rank-normalise each measure to [0,1] so no single scale
    # dominates, weight correlation and importance equally, and multiply by
    # consistency so a feature that behaves differently per fleet is
    # penalised rather than rewarded.
    def norm(s):
        r = s.rank()
        return (r - r.min()) / (r.max() - r.min()) if r.max() > r.min() else s * 0

    rank["score"] = (
        0.35 * norm(rank["rho_pooled"].abs())
        + 0.35 * norm(rank["perm_importance"])
        + 0.30 * norm(rank["mutual_info"])
    ) * (0.5 + 0.5 * rank["consistency"])

    rank = rank.sort_values("score", ascending=False)

    show = ["rho_pooled", "consistency", "mutual_info", "perm_importance", "score"]
    print(rank[show].round(3).to_string())

    print("\nPer-device Spearman (sign flips are the thing to watch):")
    print(rank[[c for c in rank.columns if c.startswith("rho_")]].round(3).to_string())

    print("\nFeature-feature redundancy (|Spearman| > 0.7 among candidates):")
    cc = df[CANDIDATE_FEATURES].corr(method="spearman").abs()
    seen = set()
    for a in CANDIDATE_FEATURES:
        for b in CANDIDATE_FEATURES:
            if a != b and (b, a) not in seen and cc.loc[a, b] > 0.7:
                seen.add((a, b))
                print(f"  {a} <-> {b}: {cc.loc[a, b]:.2f}")
    if not seen:
        print("  none")

    top8 = list(rank.index[:N_KEEP])
    mrmr = greedy_mrmr(rank["score"], cc, N_KEEP)

    print(f"\nTop {N_KEEP} by relevance alone: {top8}")
    print(f"Top {N_KEEP} after redundancy penalty: {mrmr}")

    rank["selected"] = [f in mrmr for f in rank.index]
    rank["selection_order"] = [
        mrmr.index(f) + 1 if f in mrmr else None for f in rank.index
    ]

    UNIFIED_FEATURE_RANKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    rank.to_csv(UNIFIED_FEATURE_RANKING_PATH)
    print(f"\nWrote ranking to {UNIFIED_FEATURE_RANKING_PATH}")
    print("The trainer reads `selected` from that file -- rerun it after this.")


if __name__ == "__main__":
    main()
