import numpy as np
import pandas as pd
import polars as pl

from cell_eval import MetricPipeline
from cell_eval._cli._const import (
    DEFAULT_PROGENY_LICENSE,
    DEFAULT_PROGENY_METHOD,
    DEFAULT_PROGENY_ORGANISM,
    DEFAULT_PROGENY_THR_PADJ,
    DEFAULT_PROGENY_TOP,
)
from cell_eval._cli._run import parse_args_run
from cell_eval._types import initialize_de_comparison
from cell_eval.metrics import PROGENyActivitySpearman
from cell_eval.metrics._progeny import PROGENyMethod


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
                "pathway_1",
                "pathway_1",
                "pathway_1",
                "pathway_2",
                "pathway_2",
                "pathway_2",
                "pathway_3",
                "pathway_3",
                "pathway_3",
                "pathway_4",
                "pathway_4",
                "pathway_4",
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


def test_progeny_activity_spearman_identical_de_is_perfect():
    comparison = initialize_de_comparison(real=_de_frame(), pred=_de_frame())
    metric = PROGENyActivitySpearman(net=_net(), tmin=2)

    scores = metric(comparison)

    assert set(scores) == {"pert_a", "pert_b"}
    assert np.allclose(list(scores.values()), 1.0)


def test_progeny_activity_spearman_supports_weighted_activity_methods():
    comparison = initialize_de_comparison(real=_de_frame(), pred=_de_frame())
    methods: tuple[PROGENyMethod, ...] = ("ulm", "mlm", "waggr", "zscore")

    for method in methods:
        metric = PROGENyActivitySpearman(net=_net(), method=method, tmin=2)
        scores = metric(comparison)

        assert np.allclose(list(scores.values()), 1.0)


def test_vcc_profile_runs_progeny_through_pipeline():
    comparison = initialize_de_comparison(real=_de_frame(), pred=_de_frame(scale=-1.0))
    pathway_config = {
        "gsea_nes_spearman": {
            "net": _net()[["source", "target"]],
            "times": 5,
            "tmin": 2,
        },
        "progeny_activity_spearman": {"net": _net(), "tmin": 2},
    }
    pipeline = MetricPipeline(
        profile="vcc",
        metric_configs=pathway_config,
        break_on_error=True,
    )

    pipeline.compute_de_metrics(comparison)
    results = pipeline.get_results()

    assert "progeny_activity_spearman" in results.columns
    assert results.shape[0] == 2


def test_run_parser_defaults_progeny_resource_options():
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

    assert args.progeny_method == DEFAULT_PROGENY_METHOD
    assert args.progeny_organism == DEFAULT_PROGENY_ORGANISM
    assert args.progeny_top == DEFAULT_PROGENY_TOP
    assert args.progeny_thr_padj == DEFAULT_PROGENY_THR_PADJ
    assert args.progeny_license == DEFAULT_PROGENY_LICENSE
