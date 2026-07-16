from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass
class RunContext:
    output_dir: Path
    command: list[str]
    claim_state: str
    started_at: datetime

    @classmethod
    def start(cls, output_dir: Path, command: list[str], claim_state: str) -> RunContext:
        output_dir.mkdir(parents=True, exist_ok=True)
        context = cls(output_dir, command, claim_state, datetime.now(UTC))
        context.write_environment()
        return context

    def write_environment(self) -> None:
        atomic_json(
            self.output_dir / "environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "executable": sys.executable,
                "commit": _git("rev-parse", "HEAD"),
                "dirty": bool(_git("status", "--porcelain")),
                "command": self.command,
                "started_at_utc": self.started_at,
            },
        )

    def finish(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        ended_at = datetime.now(UTC)
        # A manifest cannot contain a stable checksum of itself. Exclude an older
        # manifest when a run is resumed or regenerated in the same directory.
        manifest_path = self.output_dir / "run_manifest.json"
        files = sorted(
            path
            for path in self.output_dir.rglob("*")
            if path.is_file() and path != manifest_path
        )
        manifest: dict[str, Any] = {
            "claim_state": self.claim_state,
            "command": self.command,
            "started_at_utc": self.started_at,
            "ended_at_utc": ended_at,
            "duration_seconds": (ended_at - self.started_at).total_seconds(),
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
            "artifacts": {
                str(path.relative_to(self.output_dir)): sha256_file(path) for path in files
            },
        }
        if extra:
            manifest.update(extra)
        atomic_json(manifest_path, manifest)
        return manifest
