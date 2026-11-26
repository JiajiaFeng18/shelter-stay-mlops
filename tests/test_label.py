"""Unit tests for intake-outcome matching and labelling (run with: pytest -q)."""
import pandas as pd

from src.data.label import build_stays

T = pd.Timestamp


def intakes(rows):
    """rows: (source, animal_id, intake_time, intake_type, intake_condition)."""
    df = pd.DataFrame(rows, columns=["source", "animal_id", "intake_time", "intake_type", "intake_condition"])
    df["intake_time"] = pd.to_datetime(df["intake_time"])
    return df


def outcomes(rows):
    """rows: (source, animal_id, outcome_time, outcome_type)."""
    df = pd.DataFrame(rows, columns=["source", "animal_id", "outcome_time", "outcome_type"])
    df["outcome_time"] = pd.to_datetime(df["outcome_time"])
    df["outcome_subtype"] = None
    df["sys_intake_time"] = pd.NaT
    df["sys_days_in_shelter"] = float("nan")
    return df


def row(stays, animal_id, intake_time):
    hit = stays[(stays["animal_id"] == animal_id) & (stays["intake_time"] == T(intake_time))]
    assert len(hit) == 1
    return hit.iloc[0]


def test_short_and_long_stays_are_labelled():
    ins = intakes([("legacy", "A1", "2020-01-01", "Stray", "Normal"),
                   ("legacy", "A2", "2020-01-01", "Stray", "Normal")])
    outs = outcomes([("legacy", "A1", "2020-01-10", "Adoption"),
                     ("legacy", "A2", "2020-03-01", "Adoption")])
    stays = build_stays(ins, outs, as_of="2021-01-01")
    assert row(stays, "A1", "2020-01-01")["label"] == 0
    assert row(stays, "A2", "2020-01-01")["label"] == 1


def test_outcome_after_next_intake_is_not_used():
    # A1 has no recorded outcome for its first stay; the Mar outcome belongs to the second stay.
    ins = intakes([("legacy", "A1", "2020-01-01", "Stray", "Normal"),
                   ("legacy", "A1", "2020-02-01", "Stray", "Normal")])
    outs = outcomes([("legacy", "A1", "2020-03-01", "Adoption")])
    stays = build_stays(ins, outs, as_of="2021-01-01")
    first = row(stays, "A1", "2020-01-01")
    assert first["label_status"] == "no_outcome_record" and pd.isna(first["label"])
    second = row(stays, "A1", "2020-02-01")
    assert second["label_status"] == "observed" and second["n_prior_intakes"] == 1


def test_future_outcomes_are_hidden_by_as_of():
    # On 2020-01-20 the Feb adoption is not known yet: the stay must be pending, not 1.
    ins = intakes([("new", "7", "2020-01-01", "Stray", "Normal")])
    outs = outcomes([("new", "7", "2020-02-15", "Adopted")])
    early = build_stays(ins, outs, as_of="2020-01-20")
    assert row(early, "7", "2020-01-01")["label_status"] == "pending"
    middle = build_stays(ins, outs, as_of="2020-02-10")
    assert row(middle, "7", "2020-01-01")["label_status"] == "in_shelter_over_30"
    late = build_stays(ins, outs, as_of="2020-03-01")
    assert row(late, "7", "2020-01-01")["label"] == 1


def test_sources_are_never_cross_matched():
    ins = intakes([("legacy", "123", "2020-01-01", "Stray", "Normal")])
    outs = outcomes([("new", "123", "2020-01-05", "Adopted")])
    stays = build_stays(ins, outs, as_of="2020-01-10")
    assert row(stays, "123", "2020-01-01")["label_status"] == "pending"


def test_non_stays_are_excluded():
    ins = intakes([("new", "1", "2020-01-01", "Stray", "Dead"),
                   ("legacy", "A9", "2020-01-01", "Wildlife", "Normal"),
                   ("new", "2", "2020-01-01", "Stray", "Normal")])
    outs = outcomes([("new", "2", "2020-01-01", "DOA")])
    stays = build_stays(ins, outs, as_of="2020-06-01")
    assert stays.empty


def test_negative_stay_is_not_labelled():
    ins = intakes([("new", "5", "2020-01-02", "Stray", "Normal")])
    outs = outcomes([("new", "5", "2020-01-02", "Adopted")])
    outs.loc[0, "outcome_time"] = T("2020-01-01 23:00")   # recorded before intake
    stays = build_stays(ins, outs, as_of="2020-06-01")
    # merge_asof looks forward only, so an earlier outcome is simply not matched
    assert pd.isna(row(stays, "5", "2020-01-02")["label"])
