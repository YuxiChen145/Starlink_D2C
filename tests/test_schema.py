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


def test_metrics_available_is_in_display_order():
    got = schema.metrics_available(["lte_radio", "lte_pusch_pc"])
    assert got == [m for m in schema.METRIC_ORDER if m in got]
    assert "rsrp" in got and "throughput" in got


def test_metrics_available_ignores_unknown_keys():
    assert schema.metrics_available(["nope"]) == []
