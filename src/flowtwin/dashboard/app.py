from __future__ import annotations

from pathlib import Path

import uvicorn

from flowtwin.serving.api import create_app


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 8000,
    artifact_root: Path = Path("outputs"),
) -> None:
    uvicorn.run(create_app(artifact_root), host=host, port=port)
