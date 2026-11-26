"""Steps 4-5: match each dog intake to its outcome and build the 30-day long-stay label.

Matching rule (done separately for each source system, because animal IDs differ between them):
    For an intake of animal X at time t_in, its outcome is the FIRST outcome of X at or after t_in,
    provided it happens before X's next intake. Otherwise this stay has no recorded outcome.

Label rule, evaluated "as of" a cutoff time (only outcomes known by then are used):
    observed            outcome recorded            -> label = 1 if stay_days > 30 else 0
    in_shelter_over_30  no outcome, no later intake,
                        30 < days since intake <= 365 -> label = 1 (still in shelter after 30 days)
    pending             no outcome, < 30 days so far  -> label unknown yet (backfilled later)
    no_outcome_record   no outcome but a later intake, or no outcome for over a year
                        -> data gap, label unknown
    invalid_negative_stay  outcome before intake by data error -> label unknown

Each source has its own end of data: the legacy system stops at the migration (2025-05-05),
the new system at its latest recorded event. The cutoff used is min(as_of, end of that source).

Usage (from the repo root, after src.data.clean):
    python -m src.data.label                       # labels as of the latest data
    python -m src.data.label --as-of 2025-12-31    # labels as they were known on that date
"""
import argparse
from pathlib import Path

import pandas as pd

INTERIM_DIR = Path("data/interim")
OUT_DIR = Path("data/processed")

LONG_STAY_DAYS = 30
MAX_OPEN_STAY_DAYS = 365          # an open stay older than this is treated as a missing record
LEGACY_END = pd.Timestamp("2025-05-05")

# Records that are not a real shelter stay and should not be predicted.
EXCLUDE_INTAKE_TYPES = {"Euthanasia Request", "Wildlife"}
EXCLUDE_INTAKE_CONDITIONS = {"Dead"}
EXCLUDE_OUTCOME_PREFIXES = ("DOA", "Disposal")

OUTCOME_COLS = ["outcome_time", "outcome_type", "outcome_subtype", "sys_intake_time", "sys_days_in_shelter"]


def _prepare(intakes: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make key columns the same dtype on both sides so merge_asof can join them."""
    intakes = intakes.copy()
    outcomes = outcomes.copy()
    for df in (intakes, outcomes):
        df["animal_id"] = df["animal_id"].astype("string")
    intakes["intake_time"] = intakes["intake_time"].astype("datetime64[ns]")
    outcomes["outcome_time"] = outcomes["outcome_time"].astype("datetime64[ns]")
    outcomes["sys_intake_time"] = outcomes["sys_intake_time"].astype("datetime64[ns]")
    return intakes, outcomes


def match_outcomes(intakes: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Attach to every intake the first outcome of the same animal before its next intake."""
    intakes, outcomes = _prepare(intakes, outcomes)
    parts = []
    for source, ins in intakes.groupby("source", sort=False):
        outs = outcomes[outcomes["source"] == source]
        ins = ins.sort_values(["animal_id", "intake_time"]).copy()
        grouped = ins.groupby("animal_id")
        ins["next_intake_time"] = grouped["intake_time"].shift(-1)
        ins["n_prior_intakes"] = grouped.cumcount()          # only counts EARLIER intakes -> no leakage

        matched = pd.merge_asof(
            ins.sort_values("intake_time"),
            outs[["animal_id"] + OUTCOME_COLS].sort_values("outcome_time"),
            left_on="intake_time", right_on="outcome_time", by="animal_id", direction="forward",
        )
        # An outcome after the next intake belongs to that next stay, not this one.
        belongs_to_next = matched["next_intake_time"].notna() & (matched["outcome_time"] >= matched["next_intake_time"])
        matched.loc[belongs_to_next, OUTCOME_COLS] = None
        parts.append(matched)
    return pd.concat(parts, ignore_index=True)


def assign_labels(matched: pd.DataFrame, cutoff_by_source: dict) -> pd.DataFrame:
    """Add stay_days, label_status and label (1 = stayed more than 30 days)."""
    df = matched.copy()
    cutoff = df["source"].map(cutoff_by_source).astype("datetime64[ns]")
    df["stay_days"] = (df["outcome_time"] - df["intake_time"]).dt.total_seconds() / 86400
    days_open = (cutoff - df["intake_time"]).dt.total_seconds() / 86400

    observed = df["outcome_time"].notna()
    no_next = df["next_intake_time"].isna()
    status = pd.Series("no_outcome_record", index=df.index, dtype="string")
    status[observed] = "observed"
    status[observed & (df["stay_days"] < 0)] = "invalid_negative_stay"
    status[~observed & no_next & (days_open <= LONG_STAY_DAYS)] = "pending"
    status[~observed & no_next & (days_open > LONG_STAY_DAYS) & (days_open <= MAX_OPEN_STAY_DAYS)] = "in_shelter_over_30"
    df["label_status"] = status

    label = pd.Series(pd.NA, index=df.index, dtype="Int64")
    is_obs = status == "observed"
    label[is_obs] = (df.loc[is_obs, "stay_days"] > LONG_STAY_DAYS).astype("int64")
    label[status == "in_shelter_over_30"] = 1
    df["label"] = label
    return df


def exclude_non_stays(df: pd.DataFrame) -> pd.DataFrame:
    """Drop intakes that are not real shelter stays (dead on arrival, wildlife, euthanasia requests)."""
    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    reason[df["intake_type"].isin(EXCLUDE_INTAKE_TYPES)] = "intake_type"
    reason[df["intake_condition"].isin(EXCLUDE_INTAKE_CONDITIONS)] = "intake_condition"
    reason[df["outcome_type"].fillna("").str.startswith(EXCLUDE_OUTCOME_PREFIXES)] = "outcome_type"
    print("Excluded as non-stays:")
    print(reason.value_counts().to_string() if reason.notna().any() else "  none")
    return df[reason.isna()].reset_index(drop=True)


def build_stays(intakes: pd.DataFrame, outcomes: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Full step: restrict to what was known at as_of, match, label, exclude non-stays."""
    new_end = max(intakes.loc[intakes["source"] == "new", "intake_time"].max(),
                  outcomes.loc[outcomes["source"] == "new", "outcome_time"].max())
    as_of = pd.Timestamp(as_of) if as_of is not None else new_end
    # Nothing after the cutoff may be used: this is what makes the label leakage-free.
    intakes = intakes[intakes["intake_time"] <= as_of]
    outcomes = outcomes[outcomes["outcome_time"] <= as_of]
    cutoff_by_source = {"legacy": min(as_of, LEGACY_END), "new": min(as_of, new_end)}

    stays = assign_labels(match_outcomes(intakes, outcomes), cutoff_by_source)
    stays = exclude_non_stays(stays)
    return stays.sort_values(["source", "intake_time"]).reset_index(drop=True)


def report(stays: pd.DataFrame) -> None:
    print("\nlabel_status by source:")
    print(pd.crosstab(stays["label_status"], stays["source"], margins=True).to_string())
    labelled = stays[stays["label"].notna()].copy()
    labelled["year"] = labelled["intake_time"].dt.year
    print("\nlabelled rows and long-stay rate by intake year:")
    by_year = labelled.groupby(["source", "year"])["label"].agg(rows="size", long_stay_rate="mean")
    print(by_year.round(3).to_string())
    print("\nstay_days of observed stays (by source):")
    obs = stays[stays["label_status"] == "observed"]
    print(obs.groupby("source")["stay_days"].describe().round(1).to_string())

    new_obs = obs[obs["source"] == "new"]
    if len(new_obs):
        diff = (new_obs["stay_days"].round() - new_obs["sys_days_in_shelter"]).abs()
        print(f"\ncross-check, new system: our stay_days within 1 day of days_in_shelter for "
              f"{(diff <= 1).mean():.1%} of {len(new_obs):,} matched stays")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", help="evaluate labels as known on this date (YYYY-MM-DD)")
    args = parser.parse_args()

    intakes = pd.read_parquet(INTERIM_DIR / "intakes_clean.parquet")
    outcomes = pd.read_parquet(INTERIM_DIR / "outcomes_clean.parquet")
    stays = build_stays(intakes, outcomes, as_of=args.as_of)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stays.to_parquet(OUT_DIR / "stays_labeled.parquet", index=False)
    # Small sample for teammates to start coding against before the full pipeline is ready.
    sample = stays[stays["label"].notna()].sample(n=min(500, stays["label"].notna().sum()), random_state=42)
    sample.to_parquet(OUT_DIR / "sample_stays.parquet", index=False)
    print(f"\nSaved {OUT_DIR}/stays_labeled.parquet ({len(stays):,} rows) "
          f"and {OUT_DIR}/sample_stays.parquet ({len(sample)} rows)")
    report(stays)


if __name__ == "__main__":
    main()
