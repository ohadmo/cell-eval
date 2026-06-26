import sys
import types
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy import sparse

from cell_eval._de_engines import DEMethod
from cell_eval._de_engines._memento import TREATMENT_COL, _MementoConfig, run_memento_de
from cell_eval._de_engines._pdex import build_pdex_kwargs
from cell_eval._evaluator import _resolve_de_kwargs


def test_resolve_de_kwargs_preserves_legacy_pdex_kwargs():
    assert _resolve_de_kwargs(
        de_method="pdex",
        de_kwargs=None,
        pdex_kwargs={"geometric_mean": False},
    ) == {"geometric_mean": False}

    with pytest.raises(ValueError, match="only one"):
        _resolve_de_kwargs(
            de_method="pdex",
            de_kwargs={"clip_value": 100},
            pdex_kwargs={"geometric_mean": False},
        )

    with pytest.raises(ValueError, match="can only be used"):
        _resolve_de_kwargs(
            de_method="memento",
            de_kwargs=None,
            pdex_kwargs={"geometric_mean": False},
        )

    with pytest.raises(ValueError, match="Unsupported DE method"):
        _resolve_de_kwargs(
            de_method=cast(DEMethod, "unknown"),
            de_kwargs=None,
            pdex_kwargs=None,
        )


def test_build_pdex_kwargs_defaults_and_does_not_mutate_input():
    overrides = {"threads": 7}
    resolved = build_pdex_kwargs(
        reference="control",
        groupby="perturbation",
        threads=2,
        allow_discrete=False,
        kwargs=overrides,
    )

    assert resolved == {
        "reference": "control",
        "groupby": "perturbation",
        "threads": 7,
        "is_log1p": True,
    }
    assert overrides == {"threads": 7}


def test_memento_config_validation():
    with pytest.raises(ValueError, match="capture_rate"):
        _MementoConfig.from_kwargs({})

    with pytest.raises(ValueError, match="between 0 and 1"):
        _MementoConfig.from_kwargs({"capture_rate": 1.0})

    with pytest.raises(ValueError, match="Unexpected memento kwargs"):
        _MementoConfig.from_kwargs({"capture_rate": 0.1, "unknown": True})

    config = _MementoConfig.from_kwargs(
        {
            "capture_rate": 0.1,
            "replicate_cols": "batch,donor",
            "gene_list": "g1,g2",
        }
    )
    assert config.replicate_cols == ["batch", "donor"]
    assert config.gene_list == ["g1", "g2"]


def test_memento_de_maps_results_to_existing_de_schema(monkeypatch):
    calls: dict[str, object] = {}
    fake_memento = types.SimpleNamespace()

    def setup_memento(adata, **kwargs):
        calls["setup_kwargs"] = kwargs
        calls["subset_shape"] = adata.shape
        calls["capture_rate"] = adata.obs["_cell_eval_memento_capture_rate"].tolist()

    def create_groups(_adata, label_columns):
        calls["label_columns"] = label_columns

    def compute_1d_moments(_adata, **kwargs):
        calls["moment_kwargs"] = kwargs

    def get_groups(_adata):
        return pd.DataFrame(
            {
                TREATMENT_COL: [0, 1],
                "batch": ["b1", "b2"],
            }
        )

    def ht_1d_moments(_adata, treatment, covariate, **kwargs):
        calls["treatment"] = treatment.copy()
        calls["covariate"] = covariate.copy()
        calls["ht_kwargs"] = kwargs

    def get_1d_ht_result(_adata):
        return pd.DataFrame(
            {
                "gene": ["g1", "g2"],
                "tx": [TREATMENT_COL, TREATMENT_COL],
                "de_coef": [np.log(4.0), np.log(0.5)],
                "de_se": [0.1, 0.2],
                "de_pval": [0.01, 0.2],
                "dv_coef": [0.5, -0.5],
                "dv_se": [0.3, 0.4],
                "dv_pval": [0.03, 0.5],
            }
        )

    fake_memento.setup_memento = setup_memento
    fake_memento.create_groups = create_groups
    fake_memento.compute_1d_moments = compute_1d_moments
    fake_memento.get_groups = get_groups
    fake_memento.ht_1d_moments = ht_1d_moments
    fake_memento.get_1d_ht_result = get_1d_ht_result
    monkeypatch.setitem(sys.modules, "memento", fake_memento)

    adata = _build_small_adata()
    frame = run_memento_de(
        adata=adata,
        reference="control",
        groupby="perturbation",
        threads=2,
        kwargs={
            "capture_rate": 0.25,
            "num_boot": 11,
            "replicate_cols": "batch",
            "covariate_cols": "batch",
            "filter_mean_thresh": 0.0,
            "min_cell_count": 1,
            "min_perc_group": 0.0,
        },
    )

    assert isinstance(frame, pl.DataFrame)
    assert frame.columns[:6] == [
        "target",
        "feature",
        "log2_fold_change",
        "abs_log2_fold_change",
        "p_value",
        "fdr",
    ]
    assert frame["target"].to_list() == ["pert_a", "pert_a"]
    np.testing.assert_allclose(frame["log2_fold_change"].to_numpy(), [2.0, -1.0])
    np.testing.assert_allclose(frame["fdr"].to_numpy(), [0.02, 0.2])
    np.testing.assert_allclose(frame["dv_fdr"].to_numpy(), [0.06, 0.5])

    assert calls["subset_shape"] == (4, 3)
    assert calls["capture_rate"] == [0.25, 0.25, 0.25, 0.25]
    assert calls["label_columns"] == [TREATMENT_COL, "batch"]
    assert calls["setup_kwargs"] == {
        "q_column": "_cell_eval_memento_capture_rate",
        "filter_mean_thresh": 0.0,
        "trim_percent": 0.1,
        "shrinkage": 0.5,
        "num_bins": 30,
        "min_cell_count": 1,
        "estimator_type": "hyper_relative",
    }
    assert calls["moment_kwargs"] == {"min_perc_group": 0.0, "gene_list": None}
    assert calls["ht_kwargs"] == {
        "num_boot": 11,
        "verbose": 1,
        "num_cpus": 2,
        "approx": "norm",
        "resample_rep": False,
    }
    assert list(cast(pd.DataFrame, calls["treatment"])[TREATMENT_COL]) == [0, 1]
    assert cast(pd.DataFrame, calls["covariate"]).shape == (2, 2)


def _build_small_adata() -> ad.AnnData:
    return ad.AnnData(
        X=sparse.csr_matrix(
            np.array(
                [
                    [1.0, 0.0, 4.0],
                    [2.0, 1.0, 3.0],
                    [8.0, 2.0, 1.0],
                    [7.0, 2.0, 2.0],
                ]
            )
        ),
        obs=pd.DataFrame(
            {
                "perturbation": ["control", "control", "pert_a", "pert_a"],
                "batch": ["b1", "b2", "b1", "b2"],
            },
            index=pd.Index(["c1", "c2", "c3", "c4"]),
        ),
        var=pd.DataFrame(index=pd.Index(["g1", "g2", "g3"])),
    )
