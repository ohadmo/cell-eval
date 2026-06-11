import numpy as np
import pandas as pd
import polars as pl

from cell_eval import KNOWN_PROFILES, MetricPipeline
from cell_eval._cli._const import DEFAULT_GSEA_GENE_SETS
from cell_eval._cli._run import parse_args_run
from cell_eval._types import initialize_de_comparison
from cell_eval.metrics import GSEANESSpearman


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


def test_gsea_nes_spearman_identical_de_is_perfect():
    comparison = initialize_de_comparison(real=_de_frame(), pred=_de_frame())
    metric = GSEANESSpearman(net=_net(), times=5, tmin=2, seed=7)

    scores = metric(comparison)

    assert set(scores) == {"pert_a", "pert_b"}
    assert np.allclose(list(scores.values()), 1.0)


def test_gsea_nes_spearman_loads_gene_set_csv(tmp_path):
    gene_sets = tmp_path / "gene_sets.csv"
    _net().to_csv(gene_sets, index=False)

    comparison = initialize_de_comparison(real=_de_frame(), pred=_de_frame())
    metric = GSEANESSpearman(gene_set_path=str(gene_sets), times=5, tmin=2, seed=7)

    scores = metric(comparison)

    assert np.allclose(list(scores.values()), 1.0)


def test_vcc_profile_runs_gsea_through_pipeline():
    comparison = initialize_de_comparison(real=_de_frame(), pred=_de_frame(scale=-1.0))
    pipeline = MetricPipeline(
        profile="vcc",
        metric_configs={"gsea_nes_spearman": {"net": _net(), "times": 5, "tmin": 2}},
        break_on_error=True,
    )
    pipeline.skip_metrics("progeny_activity_spearman")

    pipeline.compute_de_metrics(comparison)
    results = pipeline.get_results()

    assert "gsea_nes_spearman" in results.columns
    assert results.shape[0] == 2


def test_gsea_nes_spearman_requires_normalization_permutations():
    try:
        GSEANESSpearman(net=_net(), times=1, tmin=2)
    except ValueError as err:
        assert "times > 1" in str(err)
    else:
        raise AssertionError("Expected times <= 1 to fail")


def test_gsea_does_not_add_extra_profiles():
    assert "gsea" not in KNOWN_PROFILES
    assert "vcc-gsea" not in KNOWN_PROFILES


def test_run_parser_defaults_gsea_gene_sets():
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

    assert args.gsea_gene_sets == DEFAULT_GSEA_GENE_SETS
