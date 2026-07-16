from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import yaml

from flowtwin.benchmarks.ais_eta import build_ais_eta_prefixes
from flowtwin.provenance import sha256_file

EXISTING_PREFIXES = Path("outputs/noaa_ais_eta_v3/prefixes.parquet")
RAW_DIR = Path("data/raw/public/noaa_ais_2025")
OUTPUT = Path("data/processed/noaa_ais_phys_jepa_holdout/prefixes.parquet")
REPORT = Path("outputs/noaa_ais_phys_jepa_holdout_build/build_report.json")
MANIFEST = Path("data/manifests/noaa_ais_2025_phys_jepa_holdout.yaml")
TRAIN_END = datetime(2025, 2, 1, tzinfo=UTC)
VALIDATION_END = datetime(2025, 2, 8, tzinfo=UTC)
TEST_END = datetime(2025, 2, 15, tzinfo=UTC)


def main() -> None:
    holdout_files = [
        RAW_DIR / f"ais-2025-02-{day:02d}.csv.zst" for day in range(8, 15)
    ]
    files = [RAW_DIR / "ais-2025-02-07.csv.zst", *holdout_files]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing frozen holdout source files: {missing}")
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    expected_hashes = manifest.get("holdout_hashes")
    if not isinstance(expected_hashes, list):
        raise RuntimeError("holdout manifest hashes must be committed before target build")
    expected = {str(item["file"]): item for item in expected_hashes}
    for path in holdout_files:
        item = expected.get(path.name)
        if item is None:
            raise RuntimeError(f"holdout manifest is missing {path.name}")
        actual_hash = sha256_file(path)
        if actual_hash != item["sha256"] or path.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"holdout hash/size mismatch: {path}")
    existing_hash = sha256_file(EXISTING_PREFIXES)
    if existing_hash != manifest["development_prefix_sha256"]:
        raise RuntimeError("development prefix cache hash does not match manifest")
    development = pl.read_parquet(EXISTING_PREFIXES).drop("partition").filter(
        pl.col("arrival_time") < VALIDATION_END
    )
    future = build_ais_eta_prefixes(files).filter(
        pl.col("arrival_time").is_between(
            VALIDATION_END, TEST_END, closed="left"
        )
    )
    combined = (
        pl.concat([development, future], how="vertical_relaxed")
        .unique(["trip_id", "prediction_cutoff"], keep="first")
        .with_columns(
            pl.when(pl.col("arrival_time") < TRAIN_END)
            .then(pl.lit("train"))
            .when(pl.col("arrival_time") < VALIDATION_END)
            .then(pl.lit("validation"))
            .otherwise(pl.lit("test"))
            .alias("partition")
        )
        .sort("arrival_time", "trip_id", "prediction_cutoff")
    )
    counts = {
        name: {
            "rows": partition.height,
            "trips": partition["trip_id"].n_unique(),
        }
        for name in ("train", "validation", "test")
        if not (
            partition := combined.filter(pl.col("partition") == name)
        ).is_empty()
    }
    if set(counts) != {"train", "validation", "test"}:
        raise RuntimeError(f"holdout build produced an empty partition: {counts}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(OUTPUT)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "existing_prefixes": {
                    "path": str(EXISTING_PREFIXES),
                    "sha256": existing_hash,
                },
                "source_files": [
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in files
                ],
                "boundaries": {
                    "train_end": TRAIN_END.isoformat(),
                    "validation_end": VALIDATION_END.isoformat(),
                    "test_end": TEST_END.isoformat(),
                },
                "counts": counts,
                "output": str(OUTPUT),
                "output_sha256": sha256_file(OUTPUT),
                "test_outcomes_opened_by_build": True,
                "models_or_thresholds_changed_after_build": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
