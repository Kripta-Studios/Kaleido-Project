# Decision 0004: aligned AIS ETA and OCEL logistics benchmarks

Date: 2026-07-16

Status: accepted for `smoke_only` demonstration

## Hypothesis

A cutoff-safe model combining vessel movement, distance and port context can
predict entry into a port geofence more accurately than direct distance/speed and
a historical port-distance median. Object relationships may also improve
container-process prediction over a flattened trace.

These tasks align more directly with Shipping Board, Freight Intelligence, Trace
Port and TWINPORTS than the historical warehouse remaining-time benchmark.

## Changes

- Added an AIS adapter that builds grouped arrival trips and half-hour prefixes
  from 12 hours to 15 minutes before geofence entry.
- Excluded MMSI, arrival time and future positions from features.
- Fixed January train/validation boundaries and an untouched 1-7 February 2025
  future test before the final run.
- Added kinematic, historical, tabular boosting and physics-residual models.
- Added trip-bootstrap uncertainty, split conformal intervals, lead-time/port
  slices and six predeclared gates.
- Added an OCEL 2.0 logistics benchmark with flat, correct-graph and shuffled-graph
  variants plus immutable visible plan revisions.
- Preserved failed AIS v1/v2 and the old warehouse/JEPA experiments as negative
  audit evidence. They are not the product demonstrator.

## Evidence

### NOAA AIS ETA v3

- Dataset/export: `noaa_marinecadastre_ais_2025_01_01_02_07`, 38 official daily
  NOAA MarineCadastre files, 6.92 GiB compressed; every source file hash is in
  `outputs/noaa_ais_eta_v3/data_manifest.json`.
- Split: grouped complete arrival trips; 303 train / 73 validation / 85 future
  test trips; 5,893 / 1,381 / 1,780 prefixes.
- Test influence: none on model, threshold, feature or gate selection.
- Selected model: `tabular_eta`, selected by validation MAE.
- Seeds: config records 41/42/43; deterministic scikit-learn fit plus trip-level
  bootstrap for uncertainty.

| Model | Test MAE (h) | Decision |
|---|---:|---|
| Tabular ETA | **1.8750** | selected |
| Physics-residual ETA | 1.8810 | comparator |
| Port-distance median | 2.7264 | historical baseline |
| Kinematic ETA | 7.7863 | physical baseline |

Selected model diagnostics:

- median absolute error 1.37 h; P90 absolute error 4.28 h;
- trip-bootstrap IC95 % for MAE 1.70-2.08 h;
- 41.97 % within +/-1 h, 60.62 % within +/-2 h, 87.3 % within +/-4 h;
- improvement 75.92 % versus kinematic and 31.23 % versus historical;
- MAE 1.96 h at 0-2 h, 1.49 h at 2-6 h and 2.13 h at 6-12 h;
- conformal P90 coverage 94.49 %, interval width 9.04 h.

All six predeclared gates passed: at least 50 test trips, MAE <=2.5 h,
bootstrap upper bound <=3 h, at least 50 % within +/-2 h, at least 30 % gain over
kinematic and at least 5 % over the historical comparator.

### OCEL 2.0 Container Logistics v1

- Dataset: public simulated OCEL 2.0 Container Logistics, DOI
  `10.5281/zenodo.18373888`, source SHA-256
  `3C91FF97F736615196902039EEC8F00BCD2F83F653D282D8A5AAEF3BAA7D3D7E`.
- Split: 1,376 / 295 / 295 finished containers, grouped chronologically.
- Test MAE: flat 88.17 h; correct graph 86.64 h; shuffled graph 88.28 h.
- Validation selected the flat model, so the graph promotion gate failed. Test
  improvement is diagnostic only and did not revise the decision.

## Decision

Use AIS ETA as the primary public capability story and dashboard evidence. Use OCEL
for process intelligence and object-centric diagnostics. Keep JEPA in research
shadow until it provides incremental value on richer objects, context or genuine
timestamped actions. Do not serve or foreground the ~734-minute warehouse model.

Claim state remains `smoke_only`.

## Tests

- Unit tests cover haversine distance, trip grouping, chronological split
  disjointness and OCEL prefix construction.
- Integration tests verify the dashboard/API prefers AIS evidence, exposes all
  six gates and returns the AIS model card.
- Generated run manifests contain artifact hashes; full Ruff, mypy and pytest
  results accompany the Git commit.

## Limitations

- NOAA AIS is a public US surrogate, not Kaleido or European-port data.
- Circular geofences are inferred targets, not terminal-specific milestones.
- 87.6 % of test prefixes are New Orleans; there are no New York test trips.
- The P90 interval is too wide for an unqualified pilot claim.
- No plan-deviation label, operator-controllable action or business value is
  measured.
- OCEL Container Logistics is simulated and its graph was not selected on
  validation.

## Next falsifiable step

Pre-register port-specific worst-group and interval-width gates, then run the same
cutoff-safe protocol on a frozen Vigo/Kaleido export. Reject transfer if it fails
the agreed ETA tolerance, lead-time, calibration or subgroup requirements.
