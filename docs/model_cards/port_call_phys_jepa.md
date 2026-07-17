# Model card: Port Call Deviation Twin Phys-JEPA v1

## Intended role

Read-only shadow module for Shipping Board/Freight Intelligence. It predicts
distance/speed state at 0.5/1/2 hours and enriches a trajectory GBT with a
physics-informed JEPA state and future forecast. Trace Port may consume an
exception, but the model never writes source data or ranks operator actions.

## Model and selection

- Product candidate: `trajectory_boosting_plus_phys_vicreg`
- JEPA: context encoder, stopped target encoder/EMA, multi-horizon latent
  predictor and physical-residual decoder
- Known physics: constant-course future supplied to the predictor
- Collapse control: VICReg, selected on development validation against VISReg,
  SIGReg and none
- Seeds: 11, 42 and 73
- Frozen commit: `cdae9b7ac9b46b4dfda4186f0ebb135f41335ab8`
- Threshold: two-hour actual-minus-physical distance shortfall over 10 km,
  fixed before clean test
- Claim state: `claim_eligible` public core evidence

## Clean test evidence

Dataset/export version, hashes and split are recorded in
`docs/data_cards/noaa_ais_phys_jepa_holdout.md`. Metrics artifact SHA-256:
`99b4162f7b40a71506491d433b74306748e6c887c8b947534662a8b48f8270ca`.

| Metric | Raw GBT | GBT + Phys-JEPA |
|---|---:|---:|
| trajectory MAE, individual-seed mean | 2.635 km | 2.587 +/- 0.053 km |
| trajectory MAE, served 3-model ensemble | 2.635 km | **2.326 km** |
| deviation AUPRC, seed mean | 0.880 | 0.904 |
| sparse ETA MAE | 1.133 h | 1.127 h |
| sparse delay AUPRC | 0.619 | 0.606 |

Paired trip bootstrap for the ensemble: 11.72% improvement, IC95%
5.90%-17.13%, 2,000 resamples over 57 trips, probability of improvement 0.9995.
Conformal 90% intervals achieve 89.79% mean coverage with 12.00 km mean width.
No selected seed collapsed; test effective rank is 11.42-12.46.

## Decision and prohibited use

The trajectory/deviation core passed its frozen gates. The full product gate
failed because sparse ETA improved only 0.59% and delay AUPRC regressed. Keep
GBT for ETA, reject the delay head and serve Phys-JEPA only in shadow.

Do not use this model for autonomous control, causal action ranking, safety
decisions, savings/ROI claims, or Kaleido production decisions. Do not describe
the public result as Vigo accuracy or successful Kaleido deployment.

## Monitoring and next gate

Expose model/data versions, cutoff, conformal band, physical shortfall, latent
surprise and abstention. Maintain a frozen reference. A Kaleido pilot must use
future grouped data, operator-agreed thresholds, worst-group metrics and a
rollback path before any promotion.
