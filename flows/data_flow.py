"""Step 7: Prefect flow that runs the whole data pipeline: ingest -> clean -> label.

Each step is a Prefect task, so every run records its state, duration and logs, and a failed
download is retried automatically. Teammate B's training flow can call `data_pipeline()` first.

Usage (from the repo root):
    python -m flows.data_flow                      # full run: download + clean + label
    python -m flows.data_flow --skip-download      # reuse data/raw (fast, for development)
    python -m flows.data_flow --as-of 2025-12-31   # labels as known on that date
    python -m flows.data_flow --serve              # keep running; full pipeline every day at 06:00

To see runs in the Prefect UI, start the server in another terminal first:
    prefect server start        # then open http://127.0.0.1:4200
"""
import argparse

from prefect import flow, get_run_logger, task

from src.data import clean, ingest, label


@task(name="ingest", retries=2, retry_delay_seconds=60)
def ingest_task() -> dict:
    """Download the four Austin Animal Center datasets from the Socrata API."""
    manifest = ingest.run()
    get_run_logger().info("Downloaded: %s", {k: v["rows"] for k, v in manifest.items()})
    return manifest


@task(name="clean")
def clean_task() -> dict:
    """Map both systems onto one schema (dogs only) and validate with Pandera."""
    counts = clean.run(show_profile=False)
    get_run_logger().info("Clean rows: %s", counts)
    return counts


@task(name="label")
def label_task(as_of: str | None) -> dict:
    """Match intakes to outcomes, build 30-day labels and validate with Pandera."""
    counts = label.run(as_of=as_of, show_report=False)
    get_run_logger().info("Label status counts: %s", counts)
    return counts


@flow(name="shelter-stay-data-pipeline", log_prints=True)
def data_pipeline(skip_download: bool = False, as_of: str | None = None) -> dict:
    """Ingest -> clean -> label. Each task only starts after the previous one succeeds."""
    if not skip_download:
        ingest_task()
    clean_task()
    return label_task(as_of)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-download", action="store_true", help="reuse files already in data/raw")
    parser.add_argument("--as-of", help="evaluate labels as known on this date (YYYY-MM-DD)")
    parser.add_argument("--serve", action="store_true", help="run on a daily schedule (06:00)")
    args = parser.parse_args()

    if args.serve:
        data_pipeline.serve(name="daily-data-refresh", cron="0 6 * * *")
    else:
        data_pipeline(skip_download=args.skip_download, as_of=args.as_of)


if __name__ == "__main__":
    main()
