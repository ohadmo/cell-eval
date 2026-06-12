import numpy as np
import pandas as pd
import polars as pl

from cell_eval import MetricPipeline
from cell_eval._cli._const import (
    DEFAULT_DOROTHEA_LEVELS,
    DEFAULT_DOROTHEA_LICENSE,
    DEFAULT_DOROTHEA_METHOD,
    DEFAULT_DOROTHEA_ORGANISM,
)
from cell_eval._cli._run import parse_args_run
from cell_eval._types import initialize_de_comparison
from cell_eval.metrics import DoRothEAActivitySpearman
from cell_eval.metrics._dorothea import DOROTHEA_ACTIVITY_METHODS, DoRothEAMethod


def _de_frame(scale: float = 1.0) -> pl.DataFrame:
    perts = ["pert_a", "pert_b"]
    genes = [f"g{i}" for i in range(1, 9)]
    values = {
        "pert_a": [2.0, 1.5, -1.0, -1.5, 0.5, -0.25, 1.0, -0.75],
        "pert_b": [-1.5, -1.0, 2.0, 1.5, -0.5, 0.25, -1.0, 0.75],
    }
    rows = []
    for pert in perts:
        for idx, gene in enumerate(genes):
            lfc = values[pert][idx] * scale
            rows.append(
                {
                    "target": pert,
                    "feature": gene,
                    "log2_fold_change": lfc,
                    "p_value": 0.001 + idx * 0.01,
                    "fdr": 0.01 + idx * 0.01,
                }
            )
    return pl.DataFrame(rows)


def _net() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": [
                "tf_1",
                "tf_1",
                "tf_1",
                "tf_2",
                "tf_2",
                "tf_2",
                "tf_3",
                "tf_3",
                "tf_3",
                "tf_4",
                "tf_4",
                "tf_4",
            ],
            "target": [
                "g1",
                "g2",
                "g7",
                "g3",
                "g4",
                "g7",
                "g5",
                "g6",
                "g8",
                "g2",
                "g5",
                "g8",
            ],
            "weight": [
                1.0,
                0.8,
                -0.3,
                -1.0,
                -0.8,
                0.4,
                1.0,
                -1.0,
                0.6,
                0.5,
                -0.5,
                1.0,
            ],
        }
    )


def test_dorothea_activity_spearman_identical_de_is_perfect():
    comparison = initialize_de_comparison(real=_de_frame(), pred=_de_frame())
    metric = DoRothEAActivitySpearman(net=_net(), tmin=2)

    scores = metric(comparison)

    assert set(scores) == {"pert_a", "pert_b"}
    assert np.allclose(list(scores.values()), 1.0)


def test_dorothea_activity_spearman_supports_weighted_activity_methods():
    comparison = initialize_de_comparison(real=_de_frame(), pred=_de_frame())
    methods: tuple[DoRothEAMethod, ...] = DOROTHEA_ACTIVITY_METHODS

    for method in methods:
        metric = DoRothEAActivitySpearman(net=_net(), method=method, tmin=2)
        scores = metric(comparison)

        assert np.allclose(list(scores.values()), 1.0)


def test_vcc_profile_runs_dorothea_through_pipeline():
    comparison = initialize_de_comparison(real=_de_frame(), pred=_de_frame(scale=-1.0))
    pipeline = MetricPipeline(
        profile="vcc",
        metric_configs={"dorothea_activity_spearman": {"net": _net(), "tmin": 2}},
        break_on_error=True,
    )
    pipeline.skip_metrics(
        [
            "gsea_nes_spearman",
            "aucell_auc_spearman",
            "collectri_activity_spearman",
            "progeny_activity_spearman",
        ]
    )

    pipeline.compute_de_metrics(comparison)
    results = pipeline.get_results()

    assert "dorothea_activity_spearman" in results.columns
    assert results.shape[0] == 2


def test_run_parser_defaults_dorothea_resource_options():
    import argparse as ap

    parser = ap.ArgumentParser()
    parse_args_run(parser)

    args = parser.parse_args(
        [
            "--adata-pred",
            "pred.h5ad",
            "--adata-real",
            "real.h5ad",
        ]
    )

    assert args.dorothea_method == DEFAULT_DOROTHEA_METHOD
    assert args.dorothea_method == "viper"
    assert args.dorothea_organism == DEFAULT_DOROTHEA_ORGANISM
    assert args.dorothea_levels == DEFAULT_DOROTHEA_LEVELS
    assert args.dorothea_license == DEFAULT_DOROTHEA_LICENSE


def test_run_parser_accepts_compact_dorothea_levels():
    import argparse as ap

    parser = ap.ArgumentParser()
    parse_args_run(parser)

    args = parser.parse_args(
        [
            "--adata-pred",
            "pred.h5ad",
            "--adata-real",
            "real.h5ad",
            "--dorothea-levels",
            "AC",
        ]
    )

    assert args.dorothea_levels == ("A", "C")
