import os
import re
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

INPUT_PARQUET = DATA_DIR / "women_data.parquet"
OUTPUT_DIR = DATA_DIR / "images"
TIMEOUT_SECONDS = 30


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


def _download_image(url: str, dest_path: Path) -> None:
    headers = {"User-Agent": "Mozilla/5.0"}
    req = Request(url, headers=headers)
    with urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        dest_path.write_bytes(resp.read())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PARQUET)
    if "image_url" not in df.columns or "reference_number" not in df.columns:
        raise SystemExit("Missing image_url or reference_number column")

    for i, row in df.iterrows():
        url = str(row["image_url"]).strip()
        ref = str(row["reference_number"]).strip()
        if not url or not ref:
            continue

        filename = f"{_safe_filename(ref)}.jpg"
        dest_path = OUTPUT_DIR / filename
        if dest_path.exists():
            continue

        try:
            _download_image(url, dest_path)
            print(f"[{i}] saved {dest_path}")
        except Exception as e:
            print(f"[{i}] failed {url}: {e}")


if __name__ == "__main__":
    main()
