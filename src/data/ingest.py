"""Download raw Austin Animal Center data from the City of Austin open data portal (Socrata API).

Four datasets are pulled:
    legacy_intakes / legacy_outcomes : old system, 2013-10-01 to 2025-05-05 (training + test)
    new_intakes    / new_outcomes    : ShelterBuddy system, from 2025-05-05 (live scoring + drift)

Raw data is saved unchanged (every column as string) so later steps are reproducible.
A manifest.json records when each file was downloaded and how many rows it has.

Usage (run from the repo root):
    python -m src.data.ingest --peek          # print columns + 3 rows of each dataset, save nothing
    python -m src.data.ingest                 # download all four datasets to data/raw/
    python -m src.data.ingest --only legacy_intakes
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data.austintexas.gov/resource/{id}.json"
DATASETS = {
    "legacy_intakes": "wter-evkm",
    "legacy_outcomes": "9t4d-g238",
    "new_intakes": "pyqf-r2dc",
    "new_outcomes": "gsvs-ypi7",
}
PAGE_SIZE = 50_000
RAW_DIR = Path("data/raw")


def make_session() -> requests.Session:
    """HTTP session; adds a Socrata app token if SOCRATA_APP_TOKEN is set (avoids rate limits)."""
    session = requests.Session()
    token = os.getenv("SOCRATA_APP_TOKEN")
    if token:
        session.headers["X-App-Token"] = token
    return session


def get_page(session: requests.Session, dataset_id: str, limit: int, offset: int) -> list:
    """Fetch one page of rows, retrying up to 3 times on network errors."""
    params = {"$limit": limit, "$offset": offset, "$order": ":id"}  # fixed order -> stable paging
    for attempt in range(3):
        try:
            resp = session.get(BASE_URL.format(id=dataset_id), params=params, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as err:
            if attempt == 2:
                raise
            print(f"  retry {attempt + 1} after error: {err}")
            time.sleep(5 * (attempt + 1))


def fetch_dataset(session: requests.Session, dataset_id: str, max_rows: int | None = None) -> pd.DataFrame:
    """Page through a whole dataset (or its first max_rows rows) and return it as a DataFrame."""
    rows, offset = [], 0
    while True:
        limit = PAGE_SIZE if max_rows is None else min(PAGE_SIZE, max_rows - offset)
        batch = get_page(session, dataset_id, limit, offset)
        rows.extend(batch)
        offset += len(batch)
        if max_rows is None:
            print(f"  {offset:,} rows")
        if len(batch) < limit or (max_rows is not None and offset >= max_rows):
            break
    # Socrata omits empty fields, so rows can have different keys; the DataFrame fills them with NaN.
    return pd.DataFrame(rows).astype("string")


def peek(session: requests.Session) -> None:
    """Show the columns and first rows of each dataset without saving anything."""
    for name, dataset_id in DATASETS.items():
        df = fetch_dataset(session, dataset_id, max_rows=3)
        print(f"\n===== {name} ({dataset_id}) =====")
        print("columns:", list(df.columns))
        print(df.to_string(max_colwidth=30))


def download(session: requests.Session, names: list[str]) -> None:
    """Download the named datasets to data/raw/<name>.parquet and update the manifest."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for name in names:
        dataset_id = DATASETS[name]
        print(f"Downloading {name} ({dataset_id}) ...")
        df = fetch_dataset(session, dataset_id)
        out = RAW_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False)
        manifest[name] = {
            "dataset_id": dataset_id,
            "rows": len(df),
            "columns": list(df.columns),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        print(f"  saved {out} ({len(df):,} rows)")

    manifest_path.write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--peek", action="store_true", help="print columns and 3 rows of each dataset")
    parser.add_argument("--only", nargs="+", choices=DATASETS, help="download only these datasets")
    args = parser.parse_args()

    session = make_session()
    if args.peek:
        peek(session)
    else:
        download(session, args.only or list(DATASETS))


if __name__ == "__main__":
    main()
