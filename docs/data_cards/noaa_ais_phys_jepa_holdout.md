# Data card: NOAA AIS Phys-JEPA future holdout

## Identity and provenance

- Dataset ID: `noaa_marinecadastre_ais_2025_phys_jepa_holdout_02_08_02_14`
- Export version: 1
- Source: NOAA MarineCadastre AccessAIS, public US AIS
- Dates: development through 2025-02-07; clean holdout 2025-02-08--14
- Holdout: seven compressed daily files; 1,431,169,298 bytes
- Download receipt SHA-256:
  `95948657f9be6242dc617b5268264490476a908efa41090df21ce70a82e7f130`
- Processed prefix SHA-256:
  `4f1c7bce92f6567599b3e1ebb66678f631c1a884adc2f4540a6bca0489ef382d`
- Manifest: `data/manifests/noaa_ais_2025_phys_jepa_holdout.yaml`
- License/terms: verify NOAA/MarineCadastre terms before redistribution; raw
  files are ignored by Git.

## Intended use

Capability benchmark for multi-horizon vessel-state prediction and a physical
port-call deviation proxy. It supports a Shipping Board/Freight Intelligence
prototype and cannot validate Trace Port operations or Kaleido business value.

## Construction

The adapter builds approach trips for New York, Houston, Los Angeles and New
Orleans and samples cutoff-safe trajectories with futures near 0.5, 1 and 2
hours. Arrival time, remaining hours and delay labels are excluded from JEPA
pretraining. Trips remain intact.

| Partition | Samples | Trips | Trip-ID SHA-256 |
|---|---:|---:|---|
| train | 3,778 | 341 | `a0f942af...4d2be` |
| validation | 976 | 83 | `f052b676...e1b40267` |
| test | 750 | 57 | `46889051...3c6ff1` |

Protocol: fixed chronological future split grouped by arrival trip; no trip
crosses partitions. Test did not influence architecture, regularizer, feature,
head or threshold selection.

## Field roles

- observations: cutoff-safe position, course, speed and approach history;
- context: vessel/trip/port identity and calendar attributes;
- outcomes: future distance/speed state, arrival-derived ETA probe and physical
  two-hour deviation proxy;
- actions: none;
- forbidden for JEPA pretraining: arrival time, remaining hours, delay label and
  any event after the cutoff.

## Known limitations

- US ports and inferred circular geofences do not represent Vigo.
- AIS gaps, reception coverage and vessel mix can create selection bias.
- The 10 km physical shortfall is an engineering proxy, not a material Kaleido
  incident or causal effect.
- The clean test has 57 trips; subgroup evidence is limited.
- Raw AIS may contain vessel identifiers and must not be republished casually.

Claim state: `claim_eligible` for the frozen public core experiment only.
