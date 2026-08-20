# EV Range & Charging Dashboard

Predicts a vehicle's full-charge range (km at 100% SOC) and summarizes its
recent charging behaviour (fast vs slow sessions), per telematics device
type (Intellicar, Tata, Citroen).

## Quick start

```
python webapp/server.py
```

Open http://127.0.0.1:8787 — pick a device, pick a vehicle, click **Get
range**. The manual CSV-export utility (download one vehicle-month of raw
Intellicar CAN data) lives at http://127.0.0.1:8787/export-tool.

## Folder layout

```
config/
  secrets.local.json        Shared API bearer token, local only (see "Credentials" below)

data/
  reference/
    vehicle_device_map.csv  vehicle -> telematics device (Intellicar/Tata/Citroen/NO_DEVICE)
  raw/
    intellicar_can/         Per-vehicle CAN pulls. `live-*` files are fetched + cached
                             here by the webapp (30-day trailing window); the rest are
                             the original historical pulls used to train the model.
    tata_gps/                GPS+CAN feed. `live-*` files are fetched + cached here by
                             the webapp; `tataraw-*` files are the original historical
                             bulk exports used to train the model -- same folder, two
                             provenances, distinguished by filename prefix.
    tata_charging/            Dedicated charging-only feed for the Tata fleet (SOC,
                             charging state/type), fetched + cached the same way.
                             Historical training doesn't use this folder -- only the
                             webapp's live charging-session lookups do.
    citroen/                  Single combined feed (driving + charging in one
                             stream). `citroenraw-*` files are historical bulk
                             pulls from src/tools/fetch_citroen_bulk.py;
                             `live-*` would be webapp pulls, same as the others.
  processed/
    intellicar_epochs.csv    One row per discharge epoch (Intellicar fleet), built by
                             src/intellicar/build_epochs.py. Training input.
    intellicar_vehicle_range.csv      Per-vehicle predicted range table (training-time snapshot).
    intellicar_new_vehicle_range.csv  Same, for vehicles added after the last training run.
    tata_epochs.csv           One row per discharge epoch (Tata fleet), same idea.
    tata_vehicle_range.csv    Per-vehicle predicted range table for Tata.

    unified_epochs.csv           THE CENTRALISED TABLE. Every device's epochs in one
                                 schema, built by src/unified/build_epochs.py.
    unified_feature_ranking.csv  All 15 candidate features scored; `selected` marks
                                 the top 8 the model uses.
    unified_vehicle_range.csv    Per-vehicle predicted range, all fleets, one table.

models/
  intellicar_range_model.joblib   Trained pipeline + metadata (model name, feature list,
                                   epochs trained on) for the Intellicar fleet.
  tata_range_model.joblib          Same, for the Tata fleet.
  unified_range_model.joblib       The single cross-device model (see "Unified model").

src/
  common/                    Shared, device-agnostic code.
    paths.py                  Every file/folder location the project uses, in one place.
    config.py                  Loads config/secrets.local.json (API tokens) -- one
                               token, shared by every device's export API (same host).
    device_map.py               vehicle -> device lookups for the webapp's dropdowns.
    epoch_splitting.py           Shared "cut the SOC timeline where it changes
                                 direction" logic, used by the discharge-epoch builders.
    charging_sessions.py         Charging-session detection + fast/slow classification
                                 (device-agnostic core; see src/intellicar/charging.py
                                 and src/tata/charging.py for the per-device column
                                 adapters).
    export_api_client.py         Shared fetch-or-reuse-cache HTTP plumbing that both
                                 device API clients below are thin wrappers around.
    intellicar_api_client.py     Live client for Intellicar's CAN export endpoint.
    tata_api_client.py           Live client for Tata's export endpoint -- two feeds,
                                 gps (range) and charging (sessions), see its docstring.
    citroen_api_client.py        Live client for Citroen's export endpoint -- one feed
                                 carrying both driving and charging signals. Note its
                                 hard 31-day-per-request cap.
    device_adapters.py           Each device's raw CSV -> one normalized row schema
                                 (time/soc/odometer/speed) + that device's own cleaning
                                 rules. The seam that makes a merged model possible.
    unified_features.py          Device-agnostic epoch features computed from those four
                                 signals. Everything is time-integrated rather than
                                 row-counted -- see "Unified model" for why that matters.
    epoch_pipeline.py            The epoch rules (noise filters, odometer-jump guard,
                                 dedupe, rest_hours) in ONE place, so the batch trainer
                                 and the live webapp cannot drift apart. If they did,
                                 the service would serve predictions computed on a
                                 different distribution than the model was fit on --
                                 and nothing would visibly break.
    central_store.py             Read/merge/persist the central epoch table. Every
                                 webapp lookup folds its new epochs in, so the store
                                 grows with use.

  intellicar/                Everything specific to the Intellicar CAN fleet.
    build_epochs.py            raw CSV -> intellicar_epochs.csv (discharge epochs).
    charging.py                  raw CSV -> charging sessions (uses fast_charge_indicator).
    train_range_model.py         Trains Decision Tree / Random Forest / Gradient
                                 Boosting / Linear Regression, keeps the lowest-MAE one.
    predict_new_vehicles.py      Scores vehicles not in the original training set.

  tata/                       Everything specific to the Tata fleet. Same shape as
                              src/intellicar/ (build_epochs.py trains off the gps feed,
                              charging.py reads the dedicated charging feed,
                              train_range_model.py) -- kept as a separate model from
                              Intellicar because the sensors, sampling rate, and
                              vehicle type are all different.

  unified/                    The cross-device pipeline that supersedes the two
                              per-device ones above. See "Unified model".
    build_epochs.py            all three raw feeds -> data/processed/unified_epochs.csv
    select_features.py          scores 15 candidate features, picks the top 8
    train_range_model.py        trains the single model, and checks merging beat
                                per-device models rather than assuming it

  citroen/                    Citroen-specific code.
    charging.py                Sessions from the single combined feed, classified by
                               the device's own typeOfCharge (Quick/Slow) flag.

  tools/
    fetch_citroen_bulk.py      One-off historical Citroen pull (chunks around the
                              endpoint's 31-day cap). Seeds data/raw/citroen/.
    fetch_intellicar_bulk.py   Pull named Intellicar vehicles (used to fetch the
                              held-out test vehicles).
    explore_raw_csv.py         Ad-hoc analyzer for any of the raw CSV formats
                              encountered so far (event-level, daily-aggregated, or
                              raw telemetry) -- useful for a first look at a new export,
                              not part of the training/serving pipeline.

webapp/
  server.py                  Thin HTTP layer (routing, JSON encoding). Run this.
  range_service.py            The actual "given a device + vehicle, return its range
                              and charging sessions" logic that server.py calls into.
  index.html                   The dashboard page.
  intellicar_export_tool.html  The older manual CSV-download page, kept at /export-tool.
```

## How a "Get range" click resolves, per device

All three devices now run through **one** service path and **one** model
(`models/unified_range_model.joblib`). The operator types a vehicle number
and nothing else -- the device is resolved from the reference sheet:

```
vehicle number
  -> resolve device        data/reference/vehicle_device_map.csv
  -> fetch telemetry       that device's export API (30-day window, 6h cache)
  -> normalize             src/common/device_adapters.py
  -> build epochs          src/common/epoch_pipeline.py  (same rules as training)
  -> merge into store      data/processed/unified_epochs.csv  (grows with use)
  -> predict               the one unified model
  -> charging sessions     from the same pull
```

| Device | Range data source | Charging-session data source |
|---|---|---|
| **Intellicar** | Live `type=can` feed, 30-day window, cached 6h | Same CAN pull -- device flag `fast_charge_indicator` |
| **Tata** | Live `type=gps` feed, 30-day window, cached 6h | Live `type=charging` feed (dedicated, cleaner than the gps stream); falls back to inferring from the gps feed. **Rate-based** fast/slow -- no trustworthy device flag |
| **Citroen** | Live feed, 30-day window, cached 6h | **Same single pull** -- device flag `typeOfCharge` (Quick/Slow) |
| **NO_DEVICE** | None | Rejected with a clear message (no telematics hardware installed) |

`GET /api/range?vehicle=<VEH>` is all that's required; `&device=<key>` is
optional and only cross-checked against the reference sheet if supplied.

### The store grows with use

Every lookup folds whatever epochs that pull revealed into
`data/processed/unified_epochs.csv`, through the *same* dedupe/filter chain
the batch builder uses. So a vehicle nobody has ever looked up gets its data
gathered on first use and is "known" to the store from then on, and the next
retraining automatically sees everything the fleet has looked at.

### Known vs new vehicles are not equally reliable

`vehicle` is a categorical feature, so a vehicle the model trained on gets
its own learned offset, while an unseen one is scored from `device` plus
driving conditions alone. On the held-out test that second case compressed
predictions toward the fleet mean and under-read genuinely long-range
vehicles by ~27 km (see "Test-set evaluation").

The API therefore returns `confidence` and `confidence_note`, and the UI
shows them as a coloured banner, so an operator can tell a vehicle-specific
answer from a fleet-average one instead of both looking equally solid:

| `confidence` | Means |
|---|---|
| `high` | In the trained set, 10+ usable epochs |
| `medium` | In the trained set, but fewer than 10 epochs -- noisy |
| `low` | **Not in the trained set** -- device + conditions only; expect it to read low on a long-range vehicle |
| `none` | No usable driving epochs in the last 30 days |

**A `low` reading is fixed by retraining**, which promotes every vehicle
currently in the store to `high`.

Both Intellicar and Tata live endpoints are on the same host
(`15.206.222.173:3000`) and accept the same bearer token
(`config/secrets.local.json`'s `api_bearer_token`) -- confirmed by using it
against both. Tata's endpoint takes plain `YYYY-MM-DD` dates (no time
component), unlike Intellicar's full ISO timestamps.

## Charging sessions

A **session** is a stretch where SOC is actually rising, tolerating gaps up
to 20 minutes between readings but no longer, and it only counts if the
vehicle gained **more than 10% SOC** during it (filters out relay chatter /
brief plug wiggles). The dashboard shows the most recent **16** sessions
found in the **last 30 days** of data -- if a vehicle charged less often
than that, it'll show fewer than 16, not padded.

The 20-minute gap rule matters in practice: an earlier version that simply
cut on "SOC never decreases" produced a 70+ hour "session" for a Tata
vehicle left plugged in over a weekend that only actively charged for ~2
hours of that (the rest was idle-while-topped-off), reporting a nonsense
~1%/hr rate for what was really a normal charge. Ending the session at the
last point it was still rising fixes that.

**Fast vs slow**: Intellicar's own `fast_charge_indicator`/`charging_status`
flags are the source of truth when present -- validated on real data (slow
sessions cluster ~10-11%/hr, fast ~30-75%/hr, a clean gap either side of the
15%/hr cutoff used below). Tata's charging feed does carry a
`data.hvChargeType` field that looked like it might be an equivalent flag,
but checked against real sessions' measured rate it showed no clean
separation, so it isn't trusted for classification. Tata (and Citroen, once
connected) fall back to the same 15%/hr rate cutoff instead.

## Unified model (one model, all three devices)

The per-device models below were replaced by a single model trained on
`data/processed/unified_epochs.csv`, which holds every fleet's discharge
epochs in one schema. Retrain it with:

```
python src/unified/build_epochs.py
python src/unified/select_features.py
python src/unified/train_range_model.py
```

### What the three feeds actually share

Almost nothing, at the column level. Their raw intersection is four signals:

| | Intellicar | Tata | Citroen |
|---|---|---|---|
| soc / odometer / speed / time | yes | yes | yes |
| battery temperature | **yes** | no | no |
| AC state | no | yes (constant-on in sample) | yes |
| ignition / gear | no | yes | yes |
| GPS altitude | no | **yes** | no |
| accelerometer | no | **yes** | event counts only |
| charge-type flag | yes | unreliable | **yes** (Quick/Slow) |

So every unified feature is *derived* from those four common signals. The
device-specific ones (battery temp, altitude, accelerometers) are dropped —
that is the actual price of merging.

### The cadence trap

The feeds sample at very different rates:

```
Intellicar   ~3-30 s        Tata  ~5 min        Citroen  ~5 min
```

Up to **100x apart**, which silently breaks any feature built by *counting*
rows. The old Tata pipeline's `driving_minutes` is literally `n_rows * 5` —
run that same code over Intellicar data and it is off by ~100x. Row counts,
stop counts and sampled variance are all cadence-dependent.

So every feature in `src/common/unified_features.py` is **integrated over
time, never counted**: each row gets a dwell (time to the next ping, capped
at 15 min so one overnight gap can't swamp an epoch) and every average is
dwell-weighted. A dwell-weighted mean speed means the same thing at 3-second
and 5-minute cadence, which is what makes the merge legitimate.

### The top 8

`select_features.py` scores 15 candidates on four measures — pooled Spearman,
**per-device sign consistency**, mutual information, and permutation
importance — then applies a redundancy penalty (mRMR), because the candidates
are heavily collinear (`avg_speed` vs `pct_time_highway` is 0.96) and a plain
relevance ranking spends five of eight slots restating one speed signal.

| # | Feature | What it carries |
|---|---|---|
| 1 | `pct_time_highway` | share of moving time above 40 km/h |
| 2 | `moving_hours` | time actually in motion |
| 3 | `pct_time_congested` | share of moving time under 15 km/h |
| 4 | `avg_speed` | dwell-weighted mean speed while moving |
| 5 | `odometer_start` | vehicle wear / age |
| 6 | `peak_speed` | weighted 95th pct (not `max` — cadence-sensitive) |
| 7 | `pct_time_moving` | duty cycle: moving time ÷ active time |
| 8 | `rest_hours_before` | idle time since this vehicle's previous epoch |

The list is regenerated by `select_features.py`, not hardcoded — the trainer
reads whatever that script marked `selected`. It is somewhat unstable in the
tail: swapping the last two or three for `speed_std` / `active_hours` /
`speed_iqr` changes MAE by well under a kilometre, because those candidates
are near-duplicates of ones already chosen. The first four are stable across
every run.

Plus `device` and `vehicle` as categoricals. Those two matter: they carry the
fleet's battery size and vehicle class, which set the *level* of range, while
the 8 numeric features explain variation around that level. Together they are
the single largest importance block (0.42).

Sign consistency is the measure that earns its keep. `odometer_start`
correlates **+0.29 on Intellicar but −0.29 on Citroen** — the fleets are at
opposite ends of their lives (median odometer ~96k km vs ~6k km on Tata), so
pooled it looks like signal while per-fleet it is two unrelated effects. It
still makes the 8, but the ranking flags it rather than hiding it.
`avg_soc`, `month` and `rest_hours_before` were dropped partly for this.

### Does merging actually beat one model per device?

`train_range_model.py` checks rather than assumes — it fits per-device models
on the identical features and CV and prints both, using vehicle-disjoint
5-fold (whole vehicles held out, since `vehicle` is itself a feature).

**It is a wash.** Averaged over 5 different fold assignments:

| device | epochs | merged MAE | device-only MAE | delta |
|---|---|---|---|---|
| Citroen | 569 | 22.1 ± 0.2 km | 22.2 ± 0.3 km | −0.1 |
| Intellicar | 4220 | 28.4 ± 0.1 km | 28.1 ± 0.3 km | +0.3 |
| Tata | 356 | 26.4 ± 0.4 km | 26.5 ± 0.3 km | −0.1 |

Every delta sits inside one standard deviation, so none of them is a real
effect. A single fold assignment will happily show "merged wins on all
three" or "separate wins on all three" — both were observed while building
this, which is exactly why the numbers above are averaged.

So the case for merging is **not** accuracy. It is that one model, one
feature pipeline and one retraining path replace three, and that Citroen
gets a range model at all — it previously had none. Accuracy-wise you give
up nothing, and the two small fleets are no longer dependent on having
enough of their own data to support a model.

The corollary is worth stating plainly: the merge does not *transfer* much
between fleets. `device` and `vehicle` account for a large share of
importance, i.e. the model mostly learns each fleet's level and then applies
broadly similar driving-condition corrections within it.

On the *random-split* protocol the old per-device numbers below were measured
with, the unified model scores **MAE 22.2 km, R²=0.52** — versus Intellicar's
old 24.9 and Tata's old 23.5. That protocol flatters every model here (it
leaks through the `vehicle` feature); the vehicle-disjoint numbers above are
the honest ones and are the ones to compare against in future.

### Two data-quality fixes that mattered more than any modelling choice

**Odometer jumps.** The odometer and the speed trace independently measure
the same journey, so distance ÷ moving time must land near the measured
average speed. When a device swap or counter rollover makes the odometer
leap, they disagree by 10x or more. **29 such epochs (0.5% of rows)** were in
the table — one Tata epoch claimed 1409 km of travel and a 15,656 km implied
range.

Because the target is a ratio, those few rows dominated the error on the
small fleets. Removing them moved overall MAE from **50.0 km to 25.8 km** and
R² from 0.01 to 0.38 — Tata alone went from 87.9 km to 26.3 km. The guard
lives in `build_epochs.py` (`MIN/MAX_ODO_SPEED_RATIO`).

**Duplicate epochs.** `data/raw/` holds overlapping export windows for many
vehicles: historical bulk pulls, the webapp's trailing-30-day `live-` caches,
and (for Intellicar) several bulk pulls whose ranges overlap each other. The
same real discharge was therefore built two or more times — **~25% of rows
were re-counted copies**. `build_epochs.py` now identifies an epoch by
`(vehicle, start_time)` and keeps the copy with the largest `soc_used`, since
a window boundary can clip one copy short.

This one did not change accuracy much (GroupKFold holds out whole vehicles,
so copies stayed inside the same fold and never leaked train→test), but it
did skew the training distribution and each vehicle's effective weight — and
it changed the merged-vs-separate verdict below, which is why it is called
out rather than quietly fixed.

## Retraining (legacy per-device models)

Superseded by the unified pipeline above; kept because the webapp still
routes per-device (see "Known limitations").

```
python src/intellicar/build_epochs.py && python src/intellicar/train_range_model.py
python src/tata/build_epochs.py       && python src/tata/train_range_model.py
```

Each trainer compares Decision Tree, Random Forest, Gradient Boosting, and
Linear Regression (as a sanity-check baseline) and keeps whichever has the
lowest held-out MAE. As of the last run, **Gradient Boosting** won for both
fleets (Intellicar: MAE 24.9 km, R²=0.46; Tata: MAE 23.5 km, R²=0.56).

## Known limitations / next steps

- **Tata's fast/slow split leans entirely on the rate fallback.** Its
  `data.hvChargeType` field looked like it could be a device flag but didn't
  correlate cleanly with measured charge rate on real sessions (see
  "Charging sessions" above), so it isn't used. In practice most of this
  fleet's sessions currently classify as "fast" under the 15%/hr cutoff
  (borrowed from Intellicar's validated split) -- plausible if Tata's depot
  chargers are genuinely higher-power, but unverified for this fleet
  specifically. Worth a ground-truth check (actual charger type/power per
  session) if that data becomes available.
- **The model is behind the store.** The store grows on every lookup, but
  the model only learns from it when retrained. Vehicles added since the
  last training run answer with `confidence: low`. Retraining
  (`select_features.py` then `train_range_model.py`) promotes them.
  **Caveat:** the 23 vehicles fetched as the held-out test set are now in
  the store, so retraining makes them in-sample and the honest held-out
  numbers in "Test-set evaluation" can no longer be reproduced. Keep a copy
  of `test_set_predictions.csv` first, or carve out a fresh holdout.
- **Citroen's device flag is not yet used to validate the 15%/hr cutoff.**
  Its sessions now classify from `typeOfCharge` (Quick/Slow), so this fleet
  has *both* a device flag and a measured rate — exactly what is needed to
  check the rate cutoff Tata is stuck with. That comparison hasn't been run.
- **The unified table's fleet balance is uneven** — ~5.8k Intellicar epochs
  vs ~600 Citroen and ~370 Tata. The trainer reweights so each device
  contributes equally, but the two small fleets are still only 7 and 10
  vehicles; their per-fleet MAE will move as more vehicles are pulled.
  `fetch_citroen_bulk.py` pulled 14 of 173 available Citroen vehicles, so
  there is a lot of headroom there.
- **Single-process store, no locking.** `central_store.py` writes atomically
  (temp file + replace), so a reader never sees a half-written CSV, but two
  simultaneous writers can have one overwrite the other's addition. The lost
  epochs come back on the next lookup or batch rebuild. If this is ever
  hosted for concurrent users, that CSV should become a real database.
- **The webapp binds to 127.0.0.1** ([server.py](webapp/server.py)) and has
  no authentication. Hosting it for the team means changing the bind
  address, putting it behind a real WSGI server, and adding auth — the
  bundled `ThreadingHTTPServer` is a development server.
- The unused/uninteresting raw columns in each export (GPS lat/long,
  driver-seatbelt flags, etc.) haven't been split out into a separate
  "extras" file yet, per the "deal with them later" note -- they're just
  ignored by the loaders for now, not removed from the source CSVs.

## Credentials

`config/secrets.local.json` holds the shared API bearer token
(`api_bearer_token`) used for every device's live export API. It's
deliberately kept out of `src/` so it's obvious it's local machine
configuration, not code. `src/common/config.py` also accepts a
`MILODRIVE_BEARER_TOKEN` environment variable as an override/alternative.
