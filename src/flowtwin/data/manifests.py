from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from flowtwin.config import load_yaml
from flowtwin.provenance import sha256_file


class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str
    bytes: int = Field(ge=0)
    sha256: str
    published_md5: str | None = None


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_id: str
    owner: str
    source_system: str
    export_version: str | int
    access_date: date
    license_or_agreement: str
    timezone_source: str
    rows: int | str | None = None
    operations: int | str | None = None
    projects: int | str | None = None
    contains_personal_data: bool | str
    contains_photos: bool
    plan_revisions_available: bool | str
    outcomes_available: bool | str
    action_columns: list[str]
    context_columns: list[str]
    observation_columns: list[str]
    outcome_columns: list[str]
    forbidden_columns: list[str]
    known_limitations: list[str]
    files: list[ManifestFile] = Field(min_length=1)


class FileVerification(BaseModel):
    path: str
    exists: bool
    size_matches: bool
    sha256_matches: bool
    observed_bytes: int | None
    observed_sha256: str | None


def load_manifest(path: Path) -> DatasetManifest:
    return DatasetManifest.model_validate(load_yaml(path))


def verify_manifest_files(
    manifest: DatasetManifest, repository_root: Path
) -> list[FileVerification]:
    results: list[FileVerification] = []
    for entry in manifest.files:
        path = repository_root / entry.path
        if not path.is_file():
            results.append(
                FileVerification(
                    path=entry.path,
                    exists=False,
                    size_matches=False,
                    sha256_matches=False,
                    observed_bytes=None,
                    observed_sha256=None,
                )
            )
            continue
        observed_bytes = path.stat().st_size
        observed_sha256 = sha256_file(path)
        results.append(
            FileVerification(
                path=entry.path,
                exists=True,
                size_matches=observed_bytes == entry.bytes,
                sha256_matches=observed_sha256 == entry.sha256,
                observed_bytes=observed_bytes,
                observed_sha256=observed_sha256,
            )
        )
    return results


def manifest_summary(manifest: DatasetManifest) -> dict[str, Any]:
    return {
        "dataset_id": manifest.dataset_id,
        "owner": manifest.owner,
        "export_version": manifest.export_version,
        "license_or_agreement": manifest.license_or_agreement,
        "limitations": manifest.known_limitations,
    }
