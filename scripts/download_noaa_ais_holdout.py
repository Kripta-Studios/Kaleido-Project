from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import date, timedelta
from pathlib import Path

BASE_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2025"
START = date(2025, 2, 8)
END = date(2025, 2, 14)
OUTPUT_DIR = Path("data/raw/public/noaa_ais_2025")
RECEIPT = Path("outputs/noaa_ais_phys_jepa_holdout_download/receipt.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "FlowTwin/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
        "wb"
    ) as handle:
        while chunk := response.read(4 * 1024 * 1024):
            handle.write(chunk)
    if temporary.stat().st_size < 1_000_000:
        raise RuntimeError(f"download is unexpectedly small: {temporary}")
    temporary.replace(destination)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    current = START
    while current <= END:
        name = f"ais-{current.isoformat()}.csv.zst"
        path = OUTPUT_DIR / name
        url = f"{BASE_URL}/{name}"
        if not path.is_file():
            print(f"downloading {url}", flush=True)
            download(url, path)
        files.append(
            {
                "date": current.isoformat(),
                "path": str(path),
                "url": url,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        current += timedelta(days=1)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(
        json.dumps(
            {
                "purpose": "frozen_noaa_ais_phys_jepa_future_holdout",
                "content_inspected": False,
                "files": files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"receipt: {RECEIPT}")


if __name__ == "__main__":
    main()
