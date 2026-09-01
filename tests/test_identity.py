import numpy as np
import pandas as pd
import pytest

from qpviz import identity
from qpviz.loader import load


# --------------------------------------------------------------------------- #
# A hand-built frame: two collections, one static (terrestrial-like) and one
# churning through a new cell every few samples (satellite-like).
# --------------------------------------------------------------------------- #


def _frame(values, collection="C", start="2026-07-22 14:00:00", step_s=1.0, **extra):
    times = pd.date_range(start, periods=len(values), freq=pd.Timedelta(seconds=step_s))
    data = {
        "time": times,
        "collection": collection,
        "pci": pd.array(values, dtype="Int32"),
        "cell_type": "MCG PCell",
    }
    data.update(extra)
    return pd.DataFrame(data)


def test_serving_only_drops_scells():
    frame = _frame([1, 1, 1])
    frame.loc[1, "cell_type"] = "MCG SCell1"
    assert len(identity.serving_only(frame)) == 2


def test_serving_only_passes_through_without_a_cell_type_column():
    frame = _frame([1, 2]).drop(columns=["cell_type"])
    assert len(identity.serving_only(frame)) == 2


def test_serving_only_drops_rows_with_no_cell_type():
    frame = _frame([1, 1])
    frame["cell_type"] = [None, "MCG PCell"]
    assert len(identity.serving_only(frame)) == 1


def test_available_levels_reports_only_populated_columns():
    frame = _frame([1, 2], enb=pd.array([10, 10], dtype="Int64"), sector=pd.NA)
    assert identity.available_levels(frame) == ["enb", "pci"]
    assert identity.available_levels(pd.DataFrame()) == []


def test_label_for():
    assert identity.label_for("enb") == "eNB / gNB-Id"
    assert identity.label_for("nope") == "nope"


# --------------------------------------------------------------------------- #
# Dwell runs
# --------------------------------------------------------------------------- #


def test_runs_collapse_consecutive_samples_on_one_cell():
    runs = identity.dwell_runs(_frame([7, 7, 7, 9, 9, 7]), "pci")
    assert runs["value"].tolist() == [7, 9, 7]
    assert runs["samples"].tolist() == [3, 2, 1]
    assert runs["order"].tolist() == [0, 1, 2]


def test_run_duration_covers_the_gap_to_the_next_run():
    # 1 s sampling: three samples on cell 7 own 3 s, not 2
    runs = identity.dwell_runs(_frame([7, 7, 7, 9]), "pci")
    assert runs.loc[0, "duration_s"] == pytest.approx(3.0)
    assert runs["duration_s"].sum() == pytest.approx(4.0)


def test_elapsed_is_measured_from_each_collections_first_sample():
    a = _frame([1, 1, 2], collection="A", start="2026-07-22 14:00:00")
    b = _frame([5, 6], collection="B", start="2026-08-01 09:30:00")
    runs = identity.dwell_runs(pd.concat([a, b], ignore_index=True), "pci")
    for _, group in runs.groupby("collection"):
        assert group["elapsed_start"].min() == pytest.approx(0.0)
    assert set(runs["collection"]) == {"A", "B"}


def test_a_logging_pause_splits_a_run_rather_than_stretching_it():
    times = pd.to_datetime(
        [
            "2026-07-22 14:00:00",
            "2026-07-22 14:00:01",
            "2026-07-22 16:00:00",  # two-hour pause, same cell
            "2026-07-22 16:00:01",
        ]
    )
    frame = pd.DataFrame(
        {"time": times, "collection": "C", "pci": pd.array([7, 7, 7, 7], dtype="Int32")}
    )
    runs = identity.dwell_runs(frame, "pci", gap_seconds=60)
    assert len(runs) == 2
    assert runs["duration_s"].max() < 10  # neither run swallows the pause


def test_runs_ignore_null_identities_and_empty_input():
    frame = _frame([1, None, 1])
    runs = identity.dwell_runs(frame, "pci")
    assert runs["samples"].sum() == 2
    assert identity.dwell_runs(pd.DataFrame(), "pci").empty
    assert identity.dwell_runs(frame, "not_a_column").empty


def test_order_identities_by_first_seen_dwell_and_value():
    frame = _frame([9, 9, 9, 9, 3, 3, 7])
    runs = identity.dwell_runs(frame, "pci")
    assert identity.order_identities(runs, "first seen") == [9, 3, 7]
    assert identity.order_identities(runs, "dwell") == [9, 3, 7]
    assert identity.order_identities(runs, "value") == [3, 7, 9]


def test_order_identities_limit_keeps_the_longest_held_in_requested_order():
    frame = _frame([1, 2, 2, 2, 3, 3])
    runs = identity.dwell_runs(frame, "pci")
    # 1 is seen first but held shortest, so a limit of 2 drops it
    assert identity.order_identities(runs, "first seen", limit=2) == [2, 3]
    assert identity.order_identities(runs, "first seen") == [1, 2, 3]
    assert identity.order_identities(pd.DataFrame(), "dwell") == []


# --------------------------------------------------------------------------- #
# Hierarchy
# --------------------------------------------------------------------------- #


def test_hierarchy_totals_match_the_measurement_span():
    frame = _frame(
        [1, 1, 2, 2],
        enb=pd.array([10, 10, 11, 11], dtype="Int64"),
        sector=pd.array([1, 1, 2, 2], dtype="Int32"),
    )
    table = identity.hierarchy(frame, ["enb", "sector", "pci"])
    assert len(table) == 2
    assert table["dwell_s"].sum() == pytest.approx(4.0)
    assert table["samples"].sum() == 4


def test_hierarchy_drops_rows_with_an_incomplete_path():
    frame = _frame([1, 2], enb=pd.array([10, None], dtype="Int64"))
    table = identity.hierarchy(frame, ["enb", "pci"])
    assert len(table) == 1
    assert table.loc[0, "enb"] == 10


def test_hierarchy_is_sorted_by_dwell_and_survives_empty_input():
    frame = _frame([1, 2, 2, 2])
    table = identity.hierarchy(frame, ["pci"])
    assert table["pci"].tolist() == [2, 1]
    assert identity.hierarchy(pd.DataFrame(), ["pci"]).empty
    assert identity.hierarchy(frame, ["not_a_column"]).empty


# --------------------------------------------------------------------------- #
# Against the synthetic exports
# --------------------------------------------------------------------------- #


def test_identity_columns_survive_the_loader(fixture_dir, cache_dir):
    frame = load(fixture_dir, ["lte_radio", "nr_radio"], cache_dir)
    for column in ("pci", "enb", "sector", "cell_id"):
        assert column in frame.columns
    serving = identity.serving_only(frame)
    assert len(serving) <= len(frame)
    levels = identity.available_levels(serving)
    assert "pci" in levels


def test_lte_enb_and_sector_are_split_out_of_the_combined_column(fixture_dir, cache_dir):
    frame = load(fixture_dir, ["lte_radio"], cache_dir)
    assert frame["enb"].notna().all() and frame["sector"].notna().all()
    assert frame["enb"].between(11000, 11004).all()
    assert set(identity.available_levels(frame)) >= {"enb", "sector", "pci"}


def test_nr_sector_is_derived_from_the_gnb_id_and_cell_id(fixture_dir, cache_dir):
    frame = load(fixture_dir, ["nr_radio"], cache_dir)
    derived = frame["cell_id"] - frame["enb"] * 256
    assert (frame["sector"] == derived).all()
    assert frame["ssb"].dropna().isin([0, 2]).all()


def test_scell_rows_are_excluded_from_identity_analysis(fixture_dir, cache_dir):
    frame = load(fixture_dir, ["lte_radio"], cache_dir)
    serving = identity.serving_only(frame)
    assert 0 < len(serving) < len(frame)
    assert set(serving["cell_type"].dropna().unique()) == {"MCG PCell"}


def test_dwell_and_hierarchy_agree_on_total_time(fixture_dir, cache_dir):
    frame = identity.serving_only(load(fixture_dir, ["lte_radio"], cache_dir))
    runs = identity.dwell_runs(frame, "pci")
    table = identity.hierarchy(frame, ["enb", "sector", "pci"])
    assert runs["duration_s"].sum() == pytest.approx(table["dwell_s"].sum(), rel=1e-6)


def test_elapsed_excludes_logging_pauses():
    # two minutes of samples either side of a two-hour pause
    first = pd.date_range("2026-07-22 14:00:00", periods=60, freq="1s")
    second = pd.date_range("2026-07-22 16:00:00", periods=60, freq="1s")
    frame = pd.DataFrame(
        {
            "time": list(first) + list(second),
            "collection": "C",
            "pci": pd.array([1] * 60 + [2] * 60, dtype="Int32"),
        }
    )
    runs = identity.dwell_runs(frame, "pci")
    wall = (frame["time"].max() - frame["time"].min()).total_seconds()
    assert wall == pytest.approx(7259.0)
    # active time is the 120 s actually logged, not the two-hour span
    assert runs["elapsed_end"].max() == pytest.approx(120.0)
    assert runs["duration_s"].sum() == pytest.approx(120.0)


def test_duration_is_the_time_the_cell_owned_not_the_wall_clock_span():
    frame = _frame([1, 1, 2], step_s=2.0)
    runs = identity.dwell_runs(frame, "pci")
    assert runs.loc[0, "duration_s"] == pytest.approx(4.0)
    assert runs.loc[0, "end"] - runs.loc[0, "start"] == pd.Timedelta(seconds=4)
    assert runs["elapsed_start"].tolist() == pytest.approx([0.0, 4.0])


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_dwell_is_independent_of_the_datetime_resolution(unit):
    """pandas 2 stores datetimes as ns and pandas 3 as us - durations must not
    change by a factor of a thousand between them."""
    frame = _frame([1, 1, 1, 2, 2], step_s=2.0)
    frame["time"] = frame["time"].astype(f"datetime64[{unit}]")
    runs = identity.dwell_runs(frame, "pci")
    assert runs["duration_s"].tolist() == pytest.approx([6.0, 4.0])
    assert runs["elapsed_end"].max() == pytest.approx(10.0)
    table = identity.hierarchy(frame, ["pci"])
    assert table["dwell_s"].sum() == pytest.approx(10.0)


def test_loader_pins_the_time_resolution(fixture_dir, cache_dir):
    frame = load(fixture_dir, ["lte_radio"], cache_dir)
    assert frame["time"].dtype == "datetime64[ns]"
    cached = load(fixture_dir, ["lte_radio"], cache_dir)
    assert cached["time"].dtype == "datetime64[ns]"


# --------------------------------------------------------------------------- #
# Fan-out
# --------------------------------------------------------------------------- #


def _tree_frame():
    """eNB 10 has 3 sectors, eNB 11 has 1; every sector has exactly 1 PCI
    except sector 3, which has 2."""
    rows = [
        (10, 1, 100), (10, 2, 200), (10, 3, 300), (10, 3, 301), (11, 9, 900),
    ]
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-07-22 14:00:00", periods=len(rows), freq="1s"),
            "collection": "C",
            "tech": ["LTE"] * 4 + ["NR"],
            "enb": pd.array([r[0] for r in rows], dtype="Int64"),
            "sector": pd.array([r[1] for r in rows], dtype="Int32"),
            "pci": pd.array([r[2] for r in rows], dtype="Int32"),
        }
    )


def test_relationships_pairs_each_level_with_its_prefix():
    assert identity.relationships(["enb", "sector", "pci"]) == [
        (("enb",), "sector"),
        (("enb", "sector"), "pci"),
        (("enb",), "pci"),  # outer to inner, only when deeper than two
    ]
    assert identity.relationships(["enb", "sector"]) == [(("enb",), "sector")]
    assert identity.relationships(["enb"]) == []


def test_relationship_label_names_the_immediate_parent():
    assert identity.relationship_label(("enb", "sector"), "pci") == "PCI per Sector Id"


def test_fan_out_counts_one_row_per_parent():
    counts = identity.fan_out_counts(_tree_frame(), ["enb", "sector", "pci"])
    sectors = counts[counts["relationship"] == "Sector Id per eNB / gNB-Id"]
    assert sorted(sectors["children"]) == [1, 3]
    pcis = counts[counts["relationship"] == "PCI per Sector Id"]
    assert sorted(pcis["children"]) == [1, 1, 1, 2]
    assert set(counts["group"]) == {"All"}


def test_sector_ids_are_scoped_to_their_enb():
    """Two eNBs reusing sector id 1 are two parents, not one."""
    frame = pd.DataFrame(
        {
            "enb": pd.array([10, 10, 11, 11], dtype="Int64"),
            "sector": pd.array([1, 1, 1, 1], dtype="Int32"),
            "pci": pd.array([100, 101, 200, 201], dtype="Int32"),
        }
    )
    counts = identity.fan_out_counts(frame, ["enb", "sector", "pci"])
    pcis = counts[counts["relationship"] == "PCI per Sector Id"]
    assert sorted(pcis["children"]) == [2, 2]


def test_fan_out_counts_can_split_the_population():
    counts = identity.fan_out_counts(_tree_frame(), ["enb", "sector"], group_by="tech")
    assert set(counts["group"]) == {"LTE", "NR"}


def test_fan_out_summary_columns_and_values():
    summary = identity.fan_out_summary(
        identity.fan_out_counts(_tree_frame(), ["enb", "sector", "pci"])
    )
    assert list(summary.columns) == [
        "Relationship", "Group", "Parents", "Children", "Min", "Median", "P90", "Max", "Mean"
    ]
    sectors = summary[summary["Relationship"] == "Sector Id per eNB / gNB-Id"].iloc[0]
    assert sectors["Parents"] == 2 and sectors["Children"] == 4
    assert sectors["Min"] == 1 and sectors["Max"] == 3
    assert sectors["Mean"] == pytest.approx(2.0)


def test_fan_out_distribution_counts_parents_and_shares():
    dist = identity.fan_out_distribution(
        identity.fan_out_counts(_tree_frame(), ["enb", "sector"])
    )
    assert dist["parents"].sum() == 2
    assert dist["share"].sum() == pytest.approx(1.0)
    assert sorted(dist["children"]) == [1, 3]


def test_fan_out_handles_too_few_levels_and_empty_frames():
    assert identity.fan_out_counts(_tree_frame(), ["enb"]).empty
    assert identity.fan_out_counts(pd.DataFrame(), ["enb", "sector"]).empty
    assert identity.fan_out_summary(pd.DataFrame()).empty
    assert identity.fan_out_distribution(pd.DataFrame()).empty


def test_fan_out_on_the_synthetic_export(fixture_dir, cache_dir):
    frame = identity.serving_only(load(fixture_dir, ["lte_radio"], cache_dir))
    summary = identity.fan_out_summary(
        identity.fan_out_counts(frame, ["enb", "sector", "pci"])
    )
    assert not summary.empty
    assert (summary["Min"] >= 1).all()
    assert (summary["Median"] <= summary["Max"]).all()


def test_relationships_keep_tree_order_not_alphabetical():
    counts = identity.fan_out_counts(_tree_frame(), ["enb", "sector", "pci"])
    order = identity.relationship_order(counts)
    assert order == [
        "Sector Id per eNB / gNB-Id",  # the outer pair leads
        "PCI per Sector Id",
        "PCI per eNB / gNB-Id",
    ]
    assert sorted(order) != order  # i.e. this is not alphabetical
    assert identity.fan_out_summary(counts)["Relationship"].tolist() == order
    assert identity.relationship_order(pd.DataFrame()) == []
