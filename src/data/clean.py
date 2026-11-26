"""Step 2: clean the raw Austin Animal Center tables into one shared schema (dogs only).

The legacy system (2013-10 to 2025-05-05) and the new ShelterBuddy system (from 2025-05-05)
use different column names, ID formats and age formats. This script maps both onto the same
columns so that later steps (matching, labelling, features) can treat them alike.

Unified intake columns:
    source, animal_id, intake_id, intake_time, intake_type, intake_condition,
    sex, neutered, age_days, breed, color
Unified outcome columns:
    source, animal_id, outcome_time, outcome_type, outcome_subtype,
    sys_intake_time, sys_days_in_shelter   (the last two exist only in the new system)

Note: animal IDs are NOT shared between systems (legacy "A521520" vs new "7022"),
so the two sources must never be matched against each other.

Usage (from the repo root, after src.data.ingest has downloaded data/raw/):
    python -m src.data.clean                      # writes data/interim/*.parquet
    python -m src.data.clean > clean_report.txt   # same, and keeps the printed profile
"""
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/interim")

INTAKE_COLS = ["source", "animal_id", "intake_id", "intake_time", "intake_type", "intake_condition",
               "sex", "neutered", "age_days", "breed", "color"]
OUTCOME_COLS = ["source", "animal_id", "outcome_time", "outcome_type", "outcome_subtype",
                "sys_intake_time", "sys_days_in_shelter"]

# Days per unit for legacy age strings such as "7 years", "1 week", "3 months".
AGE_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
AGE_PATTERN = re.compile(r"^\s*(-?\d+)\s*(day|week|month|year)s?\s*$", re.IGNORECASE)


# ---------- small parsing helpers ----------

def parse_local_time(series: pd.Series) -> pd.Series:
    """Parse ISO timestamps to naive local time.

    Some legacy outcome timestamps carry an offset (e.g. "2013-12-02T00:00:00-05:00"). We keep the
    wall-clock part (first 19 characters) and drop the offset, so every table uses the same naive
    local time and durations are not shifted by an hour.
    """
    return pd.to_datetime(series.str.slice(0, 19), format="%Y-%m-%dT%H:%M:%S", errors="coerce")


def parse_age_text(series: pd.Series) -> pd.Series:
    """Convert legacy age text ("7 years", "1 week") to days; invalid or negative values become NaN."""
    def one(text):
        if pd.isna(text):
            return float("nan")
        match = AGE_PATTERN.match(str(text))
        if not match:
            return float("nan")
        value = int(match.group(1))
        return float("nan") if value < 0 else float(value * AGE_UNIT_DAYS[match.group(2).lower()])
    return series.map(one).astype("float64")


def split_legacy_sex(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split "Neutered Male" / "Intact Female" / "Unknown" into sex and a neutered flag."""
    text = series.fillna("Unknown").str.strip()
    sex = text.str.extract(r"(Male|Female)$", expand=False).fillna("Unknown")
    neutered = pd.Series(pd.NA, index=series.index, dtype="boolean")
    neutered[text.str.startswith(("Neutered", "Spayed"))] = True
    neutered[text.str.startswith("Intact")] = False
    return sex, neutered


def to_bool(series: pd.Series) -> pd.Series:
    """Map "True"/"False" (and Yes/No) strings to a nullable boolean."""
    mapping = {"true": True, "yes": True, "false": False, "no": False}
    return series.str.strip().str.lower().map(mapping).astype("boolean")


def report_dropped(name: str, before: int, after: int, reason: str) -> None:
    print(f"  [{name}] {reason}: dropped {before - after:,} rows ({after:,} left)")


# ---------- one cleaner per raw table ----------

def clean_legacy_intakes(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["animal_type"].str.strip() == "Dog"].copy()
    sex, neutered = split_legacy_sex(df["sex_upon_intake"])
    out = pd.DataFrame({
        "source": "legacy",
        "animal_id": df["animal_id"].str.strip(),
        "intake_id": pd.NA,                      # legacy system has no intake record id
        "intake_time": parse_local_time(df["datetime"]),
        "intake_type": df["intake_type"].str.strip(),
        "intake_condition": df["intake_condition"].str.strip(),
        "sex": sex,
        "neutered": neutered,
        "age_days": parse_age_text(df["age_upon_intake"]),
        "breed": df["breed"].str.strip(),
        "color": df["color"].str.strip(),
    })
    return out


def clean_new_intakes(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["type"].str.strip() == "Dog"].copy()
    intake_time = parse_local_time(df["source_date"])
    birth = parse_local_time(df["date_of_birth"])
    age_days = (intake_time.dt.normalize() - birth.dt.normalize()).dt.days.astype("float64")
    age_days[age_days < 0] = float("nan")       # birth date after intake = data error
    out = pd.DataFrame({
        "source": "new",
        "animal_id": df["animal_id"].str.strip(),
        "intake_id": df["id"].str.strip(),
        "intake_time": intake_time,
        "intake_type": df["source_name"].str.strip(),
        "intake_condition": df["intake_health_condition"].str.strip(),
        "sex": df["sex"].fillna("Unknown").str.strip(),
        "neutered": to_bool(df["ispreviouslyspayedneutered"]),
        "age_days": age_days,
        "breed": df["primary_breed"].str.strip(),
        "color": df["primary_color"].str.strip(),
    })
    return out


def clean_legacy_outcomes(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["animal_type"].str.strip() == "Dog"].copy()
    out = pd.DataFrame({
        "source": "legacy",
        "animal_id": df["animal_id"].str.strip(),
        "outcome_time": parse_local_time(df["datetime"]),
        "outcome_type": df["outcome_type"].str.strip(),
        "outcome_subtype": df["outcome_subtype"].str.strip(),
        "sys_intake_time": pd.NaT,
        "sys_days_in_shelter": float("nan"),
    })
    return out


def clean_new_outcomes(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["type"].str.strip() == "Dog"].copy()
    out = pd.DataFrame({
        "source": "new",
        "animal_id": df["animal_id"].str.strip(),
        "outcome_time": parse_local_time(df["outcome_date"]),
        "outcome_type": df["outcome_status"].str.strip(),
        "outcome_subtype": df["euthanasia_reason"].str.strip(),
        "sys_intake_time": parse_local_time(df["intake_date"]),
        "sys_days_in_shelter": pd.to_numeric(df["days_in_shelter"], errors="coerce"),
    })
    return out


def finish(df: pd.DataFrame, name: str, time_col: str, key_cols: list[str]) -> pd.DataFrame:
    """Drop rows without an ID or a valid time, remove exact duplicate events, sort."""
    n = len(df)
    df = df.dropna(subset=["animal_id", time_col])
    report_dropped(name, n, len(df), f"missing animal_id or {time_col}")
    n = len(df)
    df = df.drop_duplicates(subset=key_cols)
    report_dropped(name, n, len(df), "duplicate events")
    return df.sort_values(["animal_id", time_col]).reset_index(drop=True)


# ---------- profile printed for the schema-mapping table ----------

def profile(name: str, df: pd.DataFrame, cat_cols: list[str], time_col: str) -> None:
    print(f"\n===== {name}: {len(df):,} rows, {df['animal_id'].nunique():,} animals =====")
    print(f"{time_col}: {df[time_col].min()} -> {df[time_col].max()}")
    print("share of times exactly at 00:00:", round((df[time_col].dt.time == pd.Timestamp(0).time()).mean(), 3))
    print("missing share per column:")
    print(df.isna().mean().round(3).to_string())
    for col in cat_cols:
        print(f"\n-- {col} (top 15) --")
        print(df[col].value_counts(dropna=False).head(15).to_string())
    if "age_days" in df:
        print("\nage_days describe:")
        print(df["age_days"].describe().round(1).to_string())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = {name: pd.read_parquet(RAW_DIR / f"{name}.parquet") for name in
           ["legacy_intakes", "new_intakes", "legacy_outcomes", "new_outcomes"]}

    print("Cleaning ...")
    intakes = pd.concat([
        finish(clean_legacy_intakes(raw["legacy_intakes"]), "legacy_intakes", "intake_time",
               ["animal_id", "intake_time"]),
        finish(clean_new_intakes(raw["new_intakes"]), "new_intakes", "intake_time",
               ["animal_id", "intake_time"]),
    ], ignore_index=True)[INTAKE_COLS]
    outcomes = pd.concat([
        finish(clean_legacy_outcomes(raw["legacy_outcomes"]), "legacy_outcomes", "outcome_time",
               ["animal_id", "outcome_time", "outcome_type"]),
        finish(clean_new_outcomes(raw["new_outcomes"]), "new_outcomes", "outcome_time",
               ["animal_id", "outcome_time", "outcome_type"]),
    ], ignore_index=True)[OUTCOME_COLS]

    intakes.to_parquet(OUT_DIR / "intakes_clean.parquet", index=False)
    outcomes.to_parquet(OUT_DIR / "outcomes_clean.parquet", index=False)
    print(f"\nSaved {OUT_DIR}/intakes_clean.parquet ({len(intakes):,}) "
          f"and {OUT_DIR}/outcomes_clean.parquet ({len(outcomes):,})")

    for source in ["legacy", "new"]:
        profile(f"{source} intakes", intakes[intakes["source"] == source],
                ["intake_type", "intake_condition", "sex", "neutered", "breed", "color"], "intake_time")
        profile(f"{source} outcomes", outcomes[outcomes["source"] == source],
                ["outcome_type", "outcome_subtype"], "outcome_time")
    new_out = outcomes[outcomes["source"] == "new"]
    print("\nnew outcomes: sys_days_in_shelter describe")
    print(new_out["sys_days_in_shelter"].describe().round(1).to_string())
    print("new outcomes with sys_intake_time before 2025-05-05:",
          int((new_out["sys_intake_time"] < "2025-05-05").sum()))


if __name__ == "__main__":
    main()
