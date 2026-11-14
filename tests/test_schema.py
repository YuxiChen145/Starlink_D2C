import pytest

from qpviz import schema


def test_every_dataset_has_time_and_collection():
    for ds in schema.DATASETS:
        assert "time" in ds.dims and "collection" in ds.dims
        assert ds.dims["pci"] and ds.dims["freq"] and ds.dims["band"]
        assert ds.metrics, f"{ds.key} declares no metric"
        assert set(ds.metrics) <= set(schema.METRICS)
        assert ds.tech in ("LTE", "NR")


def test_dataset_keys_and_filenames_unique():
    assert len({d.key for d in schema.DATASETS}) == len(schema.DATASETS)
    assert len({d.filename for d in schema.DATASETS}) == len(schema.DATASETS)


def test_freq_label_follows_technology():
    assert schema.DATASETS_BY_KEY["lte_radio"].freq_label == "EARFCN"
    assert schema.DATASETS_BY_KEY["nr_radio"].freq_label == "NR-ARFCN"


def test_qp_exports_carry_gps():
    assert schema.DATASETS_BY_KEY["lte_radio"].has_geo is True
    # the PUCCH export (CQI / Tx power only, no GPS) is no longer read
    assert "lte_pucch" not in schema.DATASETS_BY_KEY


def test_datasets_with_metric_filters_by_tech():
    nr = schema.datasets_with_metric("rsrp", tech="NR")
    assert [d.key for d in nr] == ["nr_radio"]
    assert {d.key for d in schema.datasets_with_metric("throughput")} == {
        "lte_pdsch_pc",
        "lte_pdsch_agg",
        "lte_pusch_pc",
        "nr_pdsch_pc",
        "nr_pdsch_agg",
        "nr_pusch_pc",
    }


def test_every_registered_metric_is_offered():
    assert set(schema.METRIC_ORDER) == set(schema.METRICS)
    assert len(schema.METRIC_ORDER) == len(set(schema.METRIC_ORDER))


def test_all_sources_together_supply_exactly_the_offered_metrics():
    # the union of every source (QP + PRISM + SigCap) supplies exactly the
    # offered metric set, in display order - rtt comes from SigCap only
    every_dataset = list(schema.ALL_DATASETS_BY_KEY)
    assert schema.metrics_available(every_dataset) == schema.METRIC_ORDER


@pytest.mark.parametrize(
    "metric", ["app_tput", "tput_per_rb", "cqi", "tx_power", "pathloss"]
)
def test_cut_metrics_are_gone_from_the_registry_and_every_dataset(metric):
    assert metric not in schema.METRICS
    for ds in schema.ALL_DATASETS_BY_KEY.values():
        assert metric not in ds.metrics


def test_metrics_available_is_in_display_order():
    got = schema.metrics_available(["lte_radio", "lte_pusch_pc"])
    assert got == [m for m in schema.METRIC_ORDER if m in got]
    assert "rsrp" in got and "throughput" in got


def test_metrics_available_ignores_unknown_keys():
    assert schema.metrics_available(["nope"]) == []


def test_technologies_map_to_dataset_tech_values():
    assert schema.TECHNOLOGIES == {"All": None, "LTE": "LTE", "5G NR": "NR"}


def test_datasets_for_resolves_metric_and_technology():
    # RSRP for "All" pools the LTE serving + neighbour and the NR radio exports
    assert {d.key for d in schema.datasets_for("rsrp")} == {
        "lte_radio",
        "lte_neighbor",
        "nr_radio",
    }
    assert {d.key for d in schema.datasets_for("rsrp", "NR")} == {"nr_radio"}


def test_datasets_for_spans_both_directions_where_the_metric_does():
    resolved = schema.datasets_for("throughput", "LTE")
    assert {d.link for d in resolved} == {"DL", "UL"}


def test_datasets_for_honours_what_is_on_disk():
    resolved = schema.datasets_for("rsrp", available=["nr_radio", "lte_pdsch_pc"])
    assert [d.key for d in resolved] == ["nr_radio"]
    assert schema.datasets_for("rsrp", available=[]) == []


def test_rb_metrics_map_scheduled_vs_all_sf():
    """`rb` is the scheduled-subframe grant (Avg RBs per SF), `rb_occ` the
    all-subframe occupancy (Avg RBs all SF) - both exposed on the per-carrier
    exports, and never crossed."""
    from qpviz.schema import METRICS, DATASETS_BY_KEY

    assert METRICS["rb"].label == "RBs per subframe (scheduled)"
    assert METRICS["rb_occ"].label == "RBs per subframe (all SF)"

    checks = {
        "lte_pdsch_pc": ("PDSCH Avg RBs per SF", "PDSCH Avg RBs all SF"),
        "lte_pusch_pc": ("PUSCH Avg RBs per SF", "PUSCH Avg RBs all SF"),
        "nr_pdsch_pc": ("PDSCH Avg RBs per Slot", "PDSCH Avg RBs all Slots"),
        "nr_pusch_pc": ("PUSCH Avg RBs per Slot", "PUSCH Avg RBs all Slots"),
    }
    for key, (sched, allsf) in checks.items():
        d = DATASETS_BY_KEY[key]
        assert d.metrics["rb"] == sched
        assert d.metrics["rb_occ"] == allsf


def test_traffic_label_classifies_session_names():
    from qpviz.schema import traffic_label
    assert traffic_label("D2C_OpenField_Static_twoPhone1_iperfDL") == "iperf DL"
    assert traffic_label("D2C_OpenField_Static_twoPhone1_iperfUL") == "iperf UL"
    assert traffic_label("D2C_Static_IperfDlUlPing_Building") == "Mixed"
    assert traffic_label("PingiperfULDL") == "Mixed"
    assert traffic_label("NoForcing_Static_Ookla_Field") == "Ookla"
    assert traffic_label("D2C_FitzWalking_Ping_Coverage") == "Ping"
    assert traffic_label("1011 http") == "HTTP"
    assert traffic_label("D2C_CampusWalking_IDLE_EARFCN8665") == "Idle"
    assert traffic_label("910 iperf") == "iperf"          # no direction in name
    assert traffic_label("iperfULDL") == "iperf DL+UL"
    assert traffic_label("D2C_Boulder_MostStatic_EARFCN8665") == "Other"
    assert traffic_label(None) == "Other"
