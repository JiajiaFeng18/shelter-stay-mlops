"""Step 6: Pandera schemas, the data contract between the data step (A) and modelling (B).

Each pipeline step validates its output before saving it, so a broken download or a changed
upstream format fails loudly here instead of silently producing a bad model later.

    INTAKES_CLEAN   output of src/data/clean.py   -> data/interim/intakes_clean.parquet
    OUTCOMES_CLEAN  output of src/data/clean.py   -> data/interim/outcomes_clean.parquet
    STAYS_LABELED   output of src/data/label.py   -> data/processed/stays_labeled.parquet

Validation is "lazy": every failing check is collected and reported together.
"""
import pandas as pd
import pandera.pandas as pa

SOURCES = ["legacy", "new"]
SEXES = ["Male", "Female", "Unknown"]
LABEL_STATUSES = ["observed", "in_shelter_over_30", "pending", "no_outcome_record", "invalid_negative_stay"]
LABELLED_STATUSES = ["observed", "in_shelter_over_30"]

FIRST_DATE = pd.Timestamp("2013-10-01")
MAX_AGE_DAYS = 30 * 365          # no dog is older than 30 years
MAX_STAY_DAYS = 10 * 365         # a stay longer than 10 years would be a data error


def is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def in_time_range(series: pd.Series) -> pd.Series:
    """Dates must fall between the first record (Oct 2013) and today (+1 day for time zones)."""
    upper = pd.Timestamp.now() + pd.Timedelta(days=1)
    return series.isna() | ((series >= FIRST_DATE) & (series <= upper))


# String columns are checked by value, not dtype, because pandas versions differ in string dtypes.
def text(nullable: bool = False, **kwargs) -> pa.Column:
    return pa.Column(nullable=nullable, **kwargs)


def timestamp(nullable: bool = False) -> pa.Column:
    return pa.Column(nullable=nullable, checks=[pa.Check(is_datetime, error="not a datetime column"),
                                                pa.Check(in_time_range, error="date out of range")])


INTAKES_CLEAN = pa.DataFrameSchema(
    {
        "source": text(checks=pa.Check.isin(SOURCES)),
        "animal_id": text(checks=pa.Check.str_length(min_value=1)),
        "intake_id": text(nullable=True),
        "intake_time": timestamp(),
        "intake_type": text(),
        "intake_condition": text(nullable=True),
        "sex": text(checks=pa.Check.isin(SEXES)),
        "neutered": pa.Column(nullable=True),
        "age_days": pa.Column(float, nullable=True, checks=pa.Check.in_range(0, MAX_AGE_DAYS)),
        "breed": text(nullable=True),
        "color": text(nullable=True),
    },
    unique=["source", "animal_id", "intake_time"],
    strict=True,                 # no unexpected columns
    name="intakes_clean",
)

OUTCOMES_CLEAN = pa.DataFrameSchema(
    {
        "source": text(checks=pa.Check.isin(SOURCES)),
        "animal_id": text(checks=pa.Check.str_length(min_value=1)),
        "outcome_time": timestamp(),
        "outcome_type": text(nullable=True),
        "outcome_subtype": text(nullable=True),
        "sys_intake_time": pa.Column(nullable=True),
        "sys_days_in_shelter": pa.Column(float, nullable=True),
    },
    unique=["source", "animal_id", "outcome_time", "outcome_type"],
    strict=True,
    name="outcomes_clean",
)


def label_matches_status(df: pd.DataFrame) -> pd.Series:
    """A label exists exactly when the status says the outcome is known."""
    has_label = df["label"].notna()
    should_have = df["label_status"].isin(LABELLED_STATUSES)
    return has_label == should_have


def long_stay_label_is_consistent(df: pd.DataFrame) -> pd.Series:
    """For observed stays, label must equal (stay_days > 30)."""
    observed = df["label_status"] == "observed"
    expected = (df["stay_days"] > 30).astype("Int64")
    return ~observed | (df["label"] == expected).fillna(False)


STAYS_LABELED = pa.DataFrameSchema(
    {
        **{name: col for name, col in INTAKES_CLEAN.columns.items()},
        "next_intake_time": pa.Column(nullable=True),
        "n_prior_intakes": pa.Column(int, checks=pa.Check.ge(0)),
        "outcome_time": pa.Column(nullable=True),
        "outcome_type": text(nullable=True),
        "outcome_subtype": text(nullable=True),
        "sys_intake_time": pa.Column(nullable=True),
        "sys_days_in_shelter": pa.Column(float, nullable=True),
        "stay_days": pa.Column(float, nullable=True, checks=pa.Check.in_range(0, MAX_STAY_DAYS)),
        "label_status": text(checks=pa.Check.isin(LABEL_STATUSES)),
        "label": pa.Column("Int64", nullable=True, checks=pa.Check.isin([0, 1])),
    },
    checks=[
        pa.Check(label_matches_status, error="label present/absent does not match label_status"),
        pa.Check(long_stay_label_is_consistent, error="label != (stay_days > 30) for an observed stay"),
    ],
    unique=["source", "animal_id", "intake_time"],
    strict=True,
    name="stays_labeled",
)


def validate(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Validate and print a short summary of failures (all of them) before raising."""
    try:
        out = schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        cases = err.failure_cases
        print(f"\n[{schema.name}] FAILED validation: {len(cases)} failure case(s)")
        print(cases[["column", "check", "failure_case"]].drop_duplicates().head(20).to_string())
        raise
    print(f"[{schema.name}] passed validation ({len(out):,} rows)")
    return out
