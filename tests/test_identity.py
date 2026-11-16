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
    assert identity.label_for("enb") == "eNB ID"
    assert identity.label_for("sector") == "Sector ID"
    # the NR-only SSB index is no longer a level
    assert "ssb" not in identity.IDENTITY_LEVELS
    assert identity.label_for("nope") == "nope"


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


def test_time_held_excludes_logging_pauses():
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
    table = identity.hierarchy(frame, ["pci"])
    wall = (frame["time"].max() - frame["time"].min()).total_seconds()
    assert wall == pytest.approx(7259.0)
    # the time held is the 120 s actually logged, not the two-hour span
    assert table["dwell_s"].sum() == pytest.approx(120.0)
    assert table["dwell_s"].tolist() == pytest.approx([60.0, 60.0])


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_time_held_is_independent_of_the_datetime_resolution(unit):
    """pandas 2 stores datetimes as ns and pandas 3 as us - durations must not
    change by a factor of a thousand between them."""
    frame = _frame([1, 1, 1, 2, 2], step_s=2.0)
    frame["time"] = frame["time"].astype(f"datetime64[{unit}]")
    table = identity.hierarchy(frame, ["pci"]).set_index("pci")["dwell_s"]
    assert table[1] == pytest.approx(6.0) and table[2] == pytest.approx(4.0)


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
    assert identity.relationship_label(("enb", "sector"), "pci") == "PCIs seen per sector"
    assert identity.relationship_label(("enb",), "sector") == "Sectors seen per eNB"


def test_parents_label_says_what_a_bar_counts():
    assert identity.parents_label(("enb",)) == "Number of eNBs"
    assert identity.parents_label(("enb", "sector")) == "Number of sectors"


def test_fan_out_counts_one_row_per_parent():
    counts = identity.fan_out_counts(_tree_frame(), ["enb", "sector", "pci"])
    sectors = counts[counts["relationship"] == "Sectors seen per eNB"]
    assert sorted(sectors["children"]) == [1, 3]
    pcis = counts[counts["relationship"] == "PCIs seen per sector"]
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
    pcis = counts[counts["relationship"] == "PCIs seen per sector"]
    assert sorted(pcis["children"]) == [2, 2]


def test_fan_out_counts_can_split_the_population():
    counts = identity.fan_out_counts(_tree_frame(), ["enb", "sector"], group_by="tech")
    assert set(counts["group"]) == {"LTE", "NR"}


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
    assert identity.fan_out_distribution(pd.DataFrame()).empty


def test_fan_out_on_the_synthetic_export(fixture_dir, cache_dir):
    frame = identity.serving_only(load(fixture_dir, ["lte_radio"], cache_dir))
    counts = identity.fan_out_counts(frame, ["enb", "sector", "pci"])
    assert not counts.empty
    assert (counts["children"] >= 1).all()


def test_relationships_keep_tree_order_not_alphabetical():
    counts = identity.fan_out_counts(_tree_frame(), ["enb", "sector", "pci"])
    order = list(counts["relationship"].cat.categories)
    assert order == [
        "Sectors seen per eNB",  # the outer pair leads
        "PCIs seen per sector",
        "PCIs seen per eNB",
    ]
    assert sorted(order) != order  # i.e. this is not alphabetical


# --------------------------------------------------------------------------- #
# Paths, the per-eNB summary and the per-network summary
# --------------------------------------------------------------------------- #


def _network_frame():
    """D2C: eNB 10 with sectors 1, 2, 3 (sector 3 shows two PCIs) and eNB 12
    with one sector. Terrestrial: eNB 500 with two sectors. One sample a
    second, so every path is held for about its number of samples."""
    rows = [
        ("D2C", 12, 9, 900), ("D2C", 12, 9, 900), ("D2C", 12, 9, 900),
        ("D2C", 10, 1, 100), ("D2C", 10, 2, 200),
        ("D2C", 10, 3, 300), ("D2C", 10, 3, 301),
        ("Terrestrial", 500, 1, 7), ("Terrestrial", 500, 2, 8),
    ]
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-07-22 14:00:00", periods=len(rows), freq="1s"),
            "collection": "C",
            "network": pd.Categorical([r[0] for r in rows]),
            "enb": pd.array([r[1] for r in rows], dtype="Int64"),
            "sector": pd.array([r[2] for r in rows], dtype="Int32"),
            "pci": pd.array([r[3] for r in rows], dtype="Int32"),
        }
    )


def test_paths_lists_every_path_once_per_network():
    table = identity.paths(_network_frame())
    assert list(table.columns) == ["network", "enb", "sector", "pci", "dwell_s", "samples"]
    assert len(table) == 7
    assert set(table["network"]) == {"D2C", "Terrestrial"}
    assert table["samples"].sum() == 9
    held = table.set_index(["enb", "sector", "pci"])["samples"]
    assert held[(12, 9, 900)] == 3


def test_paths_needs_the_whole_path():
    frame = _network_frame().drop(columns=["sector"])
    assert identity.paths(frame).empty
    assert identity.paths(pd.DataFrame()).empty
    # a row with no sector cannot be placed
    partial = _network_frame()
    partial.loc[0:2, "sector"] = pd.NA
    assert 12 not in set(identity.paths(partial)["enb"])


def test_paths_works_without_a_network_column():
    table = identity.paths(_network_frame().drop(columns=["network"]))
    assert list(table.columns) == ["enb", "sector", "pci", "dwell_s", "samples"]
    assert len(table) == 7


def test_enb_summary_counts_sectors_and_pcis_per_enb():
    table = identity.paths(_network_frame())
    d2c = identity.enb_summary(table[table["network"] == "D2C"])
    assert list(d2c.columns) == ["enb", "sectors", "pcis", "minutes"]
    # most sectors first, although eNB 12 was held longest
    assert d2c["enb"].tolist() == [10, 12]
    assert d2c["sectors"].tolist() == [3, 1]
    assert d2c["pcis"].tolist() == [4, 1]
    assert d2c["minutes"].sum() == pytest.approx(
        table.loc[table["network"] == "D2C", "dwell_s"].sum() / 60
    )


def test_enb_summary_can_order_by_enb_id():
    table = identity.paths(_network_frame())
    d2c = table[table["network"] == "D2C"]
    assert identity.enb_summary(d2c, "enb")["enb"].tolist() == [10, 12]
    # ties on sectors fall back to PCIs, then to the eNB ID
    tie = pd.DataFrame(
        {
            "enb": [7, 5, 6, 6], "sector": [1, 1, 1, 1], "pci": [1, 1, 1, 2],
            "dwell_s": [1.0] * 4, "samples": [1] * 4,
        }
    )
    assert identity.enb_summary(tie)["enb"].tolist() == [6, 5, 7]
    assert set(identity.ENB_ORDERS) == {"sectors", "enb"}  # never by time held
    assert identity.enb_summary(pd.DataFrame()).empty


def test_network_summary_compares_the_networks():
    summary = identity.network_summary(identity.paths(_network_frame()))
    by_net = summary.set_index("network")
    d2c, terr = by_net.loc["D2C"], by_net.loc["Terrestrial"]
    assert (d2c["enbs"], d2c["sectors"], d2c["pcis"]) == (2, 4, 5)
    assert d2c["sectors_per_enb_median"] == 2.0 and d2c["sectors_per_enb_max"] == 3
    assert d2c["pcis_per_sector_median"] == 1.0 and d2c["pcis_per_sector_max"] == 2
    assert (terr["enbs"], terr["sectors"], terr["pcis"]) == (1, 2, 2)
    assert terr["sectors_per_enb_max"] == 2 and terr["pcis_per_sector_max"] == 1
    assert identity.network_summary(pd.DataFrame()).empty


def test_fan_out_counts_split_by_network_from_the_paths_table():
    table = identity.paths(_network_frame())
    counts = identity.fan_out_counts(table, identity.TREE_LEVELS, group_by="network")
    sectors = counts[counts["relationship"] == "Sectors seen per eNB"]
    assert sorted(sectors.loc[sectors["group"] == "D2C", "children"]) == [1, 3]
    assert sorted(sectors.loc[sectors["group"] == "Terrestrial", "children"]) == [2]
