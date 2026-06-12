import numpy as np
import pandas as pd
import polars as pl
import pytest

from cell_eval import MetricPipeline
from cell_eval._cli._const import (
    DEFAULT_AUCELL_DIRECTION,
    DEFAULT_AUCELL_GENE_SETS,
    DEFAULT_AUCELL_N_UP,
)
from cell_eval._cli._run import parse_args_run
from cell_eval._types import initialize_de_comparison
from cell_eval.metrics import AUCellAUCSpearman


def _de_frame(scale: float = 1.0) -> pl.DataFrame:
    perts = ["pert_a", "pert_b"]
    genes = [f"g{i}" for i in range(1, 7)]
    values = {
        "pert_a": [2.0, 1.5, -1.0, -1.5, 0.5, -0.25],
        "pert_b": [-1.5, -1.0, 2.0, 1.5, -0.5, 0.25],
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
            "source": ["set_1", "set_1", "set_2", "set_2", "set_3", "set_3"],
            "target": ["g1", "g2", "g3", "g4", "g5", "g6"],
        }
    )


def _single_gene_set_net() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["set_1", "set_1"],
            "target": ["g1", "g2"],
        }
    )


def test_aucell_auc_spearman_identical_de_is_perfect():
    comparison = initialize_de_comparison(real=_de_frame(), pred=_de_frame())
    metric = AUCellAUCSpearman(net=_net(), n_up=3, tmin=2)

    scores = metric(comparison)

    assert set(scores) == {"pert_a", "pert_b"}
    assert np.allclose(list(scores.values()), 1.0)


@pytest.mark.parametrize("direction", ["up", "down", "both"])
def test_aucell_auc_spearman_supports_de_directions(direction):
    comparison = initialize_de_comparison(real=_de_frame(), pred=_de_frame())
    metric = AUCellAUCSpearman(
        net=_net(),
        n_up=3,
        direction=direction,
        tmin=2,
    )

    scores = metric(comparison)

    assert np.allclose(list(scores.values()), 1.0)


def test_aucell_auc_spearman_counts_underlying_gene_sets_for_both_direction():
    comparison = initialize_de_comparison(real=_de_frame(), pred=_de_frame())
    metric = AUCellAUCSpearman(
        net=_single_gene_set_net(),
        n_up=3,
        direction="both",
        tmin=2,
    )

    with pytest.raises(ValueError, match="at least 2 shared AUCell"):
        metric(comparison)


def test_aucell_auc_spearman_loads_gene_set_csv(tmp_path):
    gene_sets = tmp_path / "gene_sets.csv"
    _net().to_csv(gene_sets, index=False)

    comparison = initialize_de_comparison(real=_de_frame(), pred=_de_frame())
    metric = AUCellAUCSpearman(gene_set_path=str(gene_sets), n_up=3, tmin=2)

    scores = metric(comparison)

    assert np.allclose(list(scores.values()), 1.0)


def test_vcc_profile_runs_aucell_through_pipeline():
    comparison = initialize_de_comparison(real=_de_frame(), pred=_de_frame(scale=-1.0))
    pipeline = MetricPipeline(
        profile="vcc",
        metric_configs={"aucell_auc_spearman": {"net": _net(), "n_up": 3, "tmin": 2}},
        break_on_error=True,
    )
    pipeline.skip_metrics(
        [
            "gsea_nes_spearman",
            "dorothea_activity_spearman",
            "collectri_activity_spearman",
            "progeny_activity_spearman",
        ]
    )

    pipeline.compute_de_metrics(comparison)
    results = pipeline.get_results()

    assert "aucell_auc_spearman" in results.columns
    assert results.shape[0] == 2


def test_run_parser_defaults_aucell_options():
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

    assert args.aucell_gene_sets == DEFAULT_AUCELL_GENE_SETS
    assert args.aucell_n_up == DEFAULT_AUCELL_N_UP
    assert args.aucell_direction == DEFAULT_AUCELL_DIRECTION


def test_run_parser_accepts_aucell_n_up_integer():
    import argparse as ap

    parser = ap.ArgumentParser()
    parse_args_run(parser)

    args = parser.parse_args(
        [
            "--adata-pred",
            "pred.h5ad",
            "--adata-real",
            "real.h5ad",
            "--aucell-n-up",
            "100",
        ]
    )

    assert args.aucell_n_up == 100


def test_run_parser_rejects_invalid_aucell_n_up():
    import argparse as ap

    parser = ap.ArgumentParser()
    parse_args_run(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--adata-pred",
                "pred.h5ad",
                "--adata-real",
                "real.h5ad",
                "--aucell-n-up",
                "1",
            ]
        )
