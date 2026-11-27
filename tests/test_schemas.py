"""Unit tests for the Pandera data contract (run with: pytest -q)."""
import pandas as pd
import pandera.pandas as pa
import pytest

from src.data.label import build_stays
from src.data.schemas import STAYS_LABELED


def good_stays() -> pd.DataFrame:
    """A small, valid labelled table built through the real pipeline functions."""
    ins = pd.DataFrame({
        "source": ["legacy", "legacy", "new"],
        "animal_id": ["A1", "A2", "7"],
        "intake_id": [pd.NA, pd.NA, "100"],
        "intake_time": pd.to_datetime(["2021-01-01", "2021-02-01", "2025-06-01"]),
        "intake_type": ["Stray", "Owner Surrender", "Stray"],
        "intake_condition": ["Normal", "Sick", pd.NA],
        "sex": ["Male", "Female", "Unknown"],
        "neutered": pd.array([True, False, pd.NA], dtype="boolean"),
        "age_days": [365.0, 2555.0, float("nan")],
        "breed": ["Pit Bull Mix", "Labrador Retriever", "Mixed Breed"],
        "color": ["Black/White", "Tan", "Black"],
    })
    outs = pd.DataFrame({
        "source": ["legacy", "legacy", "new"],
        "animal_id": ["A1", "A2", "7"],
        "outcome_time": pd.to_datetime(["2021-01-05", "2021-04-01", "2025-06-10"]),
        "outcome_type": ["Adoption", "Transfer", "Adopted"],
        "outcome_subtype": [pd.NA, "Partner", pd.NA],
        "sys_intake_time": pd.to_datetime([pd.NaT, pd.NaT, "2025-06-01"]),
        "sys_days_in_shelter": [float("nan"), float("nan"), 9.0],
    })
    return build_stays(ins, outs, as_of="2025-12-31")


def test_valid_table_passes():
    STAYS_LABELED.validate(good_stays(), lazy=True)


@pytest.mark.parametrize("column, bad_value", [
    ("source", "other_system"),        # unknown source
    ("sex", "M"),                      # category outside the contract
    ("age_days", -10.0),               # impossible age
    ("intake_time", pd.Timestamp("2099-01-01")),   # date in the future
])
def test_bad_values_are_caught(column, bad_value):
    df = good_stays()
    df.loc[0, column] = bad_value
    with pytest.raises(pa.errors.SchemaErrors):
        STAYS_LABELED.validate(df, lazy=True)


def test_duplicate_intake_is_caught():
    df = good_stays()
    with pytest.raises(pa.errors.SchemaErrors):
        STAYS_LABELED.validate(pd.concat([df, df.iloc[[0]]], ignore_index=True), lazy=True)


def test_wrong_label_is_caught():
    df = good_stays()
    df.loc[df["stay_days"] < 30, "label"] = 1      # short stay wrongly marked long
    with pytest.raises(pa.errors.SchemaErrors):
        STAYS_LABELED.validate(df, lazy=True)


def test_extra_column_is_caught():
    df = good_stays().assign(unexpected=1)
    with pytest.raises(pa.errors.SchemaErrors):
        STAYS_LABELED.validate(df, lazy=True)
