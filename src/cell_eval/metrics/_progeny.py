"""PROGENy pathway activity concordance metrics."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from .._types import DEComparison
from ._gsea_nes import (
    GSEARankBy,
    _bounded_spearman,
    _build_rank_matrix,
    _prepare_mpl_config,
)

PROGENyLicense = Literal["academic", "commercial", "nonprofit"]
PROGENyMethod = Literal["ulm", "mlm", "waggr", "zscore"]


class PROGENyActivitySpearman:
    """Compare real and predicted PROGENy pathway activities per perturbation."""

    def __init__(
        self,
        net: pd.DataFrame | pl.DataFrame | None = None,
        method: PROGENyMethod = "ulm",
        rank_by: GSEARankBy = "log2_fold_change",
        organism: str = "human",
        top: int | float = np.inf,
        thr_padj: float = 0.05,
        license: PROGENyLicense = "commercial",
        tmin: int = 5,
        min_pathways: int = 2,
        verbose: bool = False,
    ) -> None:
        self.net = net
        if method not in {"ulm", "mlm", "waggr", "zscore"}:
            raise ValueError("PROGENy method must be one of: ulm, mlm, waggr, zscore")
        self.method = method
        self.rank_by = rank_by
        self.organism = organism
        self.top = top
        self.thr_padj = thr_padj
        self.license = license
        self.tmin = tmin
        self.min_pathways = min_pathways
        self.verbose = verbose

    def __call__(self, data: DEComparison) -> dict[str, float]:
        net = _load_progeny_net(
            net=self.net,
            organism=self.organism,
            top=self.top,
            thr_padj=self.thr_padj,
            license=self.license,
            verbose=self.verbose,
        )

        real_rank = _build_rank_matrix(data.real, rank_by=self.rank_by)
        pred_rank = _build_rank_matrix(data.pred, rank_by=self.rank_by)

        shared_features = [g for g in real_rank.columns if g in pred_rank.columns]
        if not shared_features:
            raise ValueError(
                "No shared DE features found between real and predicted results"
            )

        real_scores = _run_progeny_activity(
            real_rank.loc[:, shared_features],
            net=net,
            method=self.method,
            tmin=self.tmin,
            verbose=self.verbose,
        )
        pred_scores = _run_progeny_activity(
            pred_rank.loc[:, shared_features],
            net=net,
            method=self.method,
            tmin=self.tmin,
            verbose=self.verbose,
        )

        shared_pathways = [p for p in real_scores.columns if p in pred_scores.columns]
        if len(shared_pathways) < self.min_pathways:
            raise ValueError(
                f"Need at least {self.min_pathways} shared PROGENy pathways after "
                f"pruning, found {len(shared_pathways)}"
            )

        scores: dict[str, float] = {}
        for pert in data.iter_perturbations():
            real_vec = real_scores.loc[str(pert), shared_pathways].to_numpy(dtype=float)
            pred_vec = pred_scores.loc[str(pert), shared_pathways].to_numpy(dtype=float)
            scores[str(pert)] = _bounded_spearman(real_vec, pred_vec)
        return scores


def _load_progeny_net(
    net: pd.DataFrame | pl.DataFrame | None,
    organism: str,
    top: int | float,
    thr_padj: float,
    license: PROGENyLicense,
    verbose: bool,
) -> pd.DataFrame:
    if net is not None:
        frame = net.to_pandas() if isinstance(net, pl.DataFrame) else net.copy()
    else:
        _prepare_mpl_config()
        import decoupler as dc

        frame = dc.op.progeny(
            organism=organism,
            top=top,
            thr_padj=thr_padj,
            license=license,
            verbose=verbose,
        )

    frame = _normalize_progeny_columns(frame)
    keep_cols = ["source", "target"] + (["weight"] if "weight" in frame.columns else [])
    frame = frame.loc[:, keep_cols].dropna().drop_duplicates(["source", "target"])
    if frame.empty:
        raise ValueError("PROGENy network is empty after normalization")
    return frame


def _normalize_progeny_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {str(c).lower(): c for c in frame.columns}
    rename = {}
    if "source" not in columns:
        if "pathway" in columns:
            rename[columns["pathway"]] = "source"
        elif "geneset" in columns:
            rename[columns["geneset"]] = "source"
    if "target" not in columns:
        if "gene" in columns:
            rename[columns["gene"]] = "target"
        elif "genesymbol" in columns:
            rename[columns["genesymbol"]] = "target"
        elif "gene_symbol" in columns:
            rename[columns["gene_symbol"]] = "target"
    frame = frame.rename(columns=rename)
    if "source" not in frame.columns or "target" not in frame.columns:
        raise ValueError(
            "PROGENy network must contain source/target columns, or recognizable "
            "synonyms such as pathway/genesymbol"
        )
    return frame


def _run_progeny_activity(
    matrix: pd.DataFrame,
    net: pd.DataFrame,
    method: PROGENyMethod,
    tmin: int,
    verbose: bool,
) -> pd.DataFrame:
    _prepare_mpl_config()
    import decoupler as dc

    kwargs = {"tval": True} if method in {"ulm", "mlm"} else {}
    scores, _ = getattr(dc.mt, method)(
        data=matrix,
        net=net,
        tmin=tmin,
        verbose=verbose,
        **kwargs,
    )
    return scores
