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


def test_pucch_has_no_geo():
    assert schema.DATASETS_BY_KEY["lte_pucch"].has_geo is False
    assert schema.DATASETS_BY_KEY["lte_radio"].has_geo is True


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


def test_offered_and_hidden_metrics_partition_the_registry():
    assert set(schema.METRIC_ORDER) | set(schema.HIDDEN_METRICS) == set(schema.METRICS)
    assert not set(schema.METRIC_ORDER) & set(schema.HIDDEN_METRICS)


def test_hidden_metrics_are_never_offered():
    # the union of every source (QP + PRISM + SigCap) supplies exactly the
    # offered metric set, in display order - app_tput/rtt come from SigCap only
    every_dataset = list(schema.ALL_DATASETS_BY_KEY)
    offered = schema.metrics_available(every_dataset)
    assert offered == schema.METRIC_ORDER
    for metric in schema.HIDDEN_METRICS:
        assert metric not in offered
        # still mapped by the datasets, so restoring one needs no other change
        assert schema.datasets_with_metric(metric)


def test_pucch_carries_only_hidden_metrics_so_it_is_not_reachable():
    assert schema.metrics_available(["lte_pucch"]) == []


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
