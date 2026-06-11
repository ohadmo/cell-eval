"""Pathway-level concordance metrics based on pre-ranked GSEA NES."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr

from .._types import DEComparison, DEResults

logger = logging.getLogger(__name__)

GSEARankBy = Literal["log2_fold_change", "signed_pvalue", "signed_fdr"]


class GSEANESSpearman:
    """Compare real and predicted pathway-level GSEA NES per perturbation."""

    def __init__(
        self,
        gene_set_path: str | None = None,
        net: pd.DataFrame | pl.DataFrame | None = None,
        rank_by: GSEARankBy = "log2_fold_change",
        tmin: int = 5,
        times: int = 1000,
        seed: int = 42,
        min_pathways: int = 2,
        verbose: bool = False,
    ) -> None:
        if times <= 1:
            raise ValueError(
                "GSEA NES requires times > 1. Decoupler returns raw ES when "
                "times <= 1, so increase --gsea-times to compute NES."
            )
        self.gene_set_path = gene_set_path
        self.net = net
        self.rank_by = rank_by
        self.tmin = tmin
        self.times = times
        self.seed = seed
        self.min_pathways = min_pathways
        self.verbose = verbose

    def __call__(self, data: DEComparison) -> dict[str, float]:
        net = _load_gene_sets(
            gene_set_path=self.gene_set_path,
            net=self.net,
            verbose=self.verbose,
        )

        real_rank = _build_rank_matrix(data.real, rank_by=self.rank_by)
        pred_rank = _build_rank_matrix(data.pred, rank_by=self.rank_by)

        shared_features = [g for g in real_rank.columns if g in pred_rank.columns]
        if not shared_features:
            raise ValueError(
                "No shared DE features found between real and predicted results"
            )

        real_scores = _run_gsea(
            real_rank.loc[:, shared_features],
            net=net,
            tmin=self.tmin,
            times=self.times,
            seed=self.seed,
            verbose=self.verbose,
        )
        pred_scores = _run_gsea(
            pred_rank.loc[:, shared_features],
            net=net,
            tmin=self.tmin,
            times=self.times,
            seed=self.seed,
            verbose=self.verbose,
        )

        shared_pathways = [p for p in real_scores.columns if p in pred_scores.columns]
        if len(shared_pathways) < self.min_pathways:
            raise ValueError(
                f"Need at least {self.min_pathways} shared pathways after pruning, "
                f"found {len(shared_pathways)}"
            )

        scores: dict[str, float] = {}
        for pert in data.iter_perturbations():
            real_vec = real_scores.loc[str(pert), shared_pathways].to_numpy(dtype=float)
            pred_vec = pred_scores.loc[str(pert), shared_pathways].to_numpy(dtype=float)
            scores[str(pert)] = _bounded_spearman(real_vec, pred_vec)
        return scores


def _load_gene_sets(
    gene_set_path: str | None,
    net: pd.DataFrame | pl.DataFrame | None,
    verbose: bool = False,
) -> pd.DataFrame:
    if gene_set_path is not None and net is not None:
        raise ValueError("Specify only one of gene_set_path or net")

    if net is not None:
        frame = net.to_pandas() if isinstance(net, pl.DataFrame) else net.copy()
    elif gene_set_path is not None:
        path = Path(gene_set_path)
        if not path.exists():
            raise FileNotFoundError(f"Gene set file does not exist: {path}")
        if path.suffix.lower() == ".gmt":
            _prepare_mpl_config()
            import decoupler as dc

            frame = dc.pp.read_gmt(str(path))
        else:
            sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
            frame = pd.read_csv(path, sep=sep)
    else:
        _prepare_mpl_config()
        import decoupler as dc

        frame = dc.op.hallmark(verbose=verbose)

    frame = _normalize_gene_set_columns(frame)
    frame = frame.loc[:, ["source", "target"]].dropna().drop_duplicates()
    if frame.empty:
        raise ValueError("Gene set network is empty after normalization")
    return frame


def _normalize_gene_set_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {str(c).lower(): c for c in frame.columns}
    source_col = _first_present(
        columns, ["source", "geneset", "gene_set", "pathway", "term"]
    )
    target_col = _first_present(
        columns, ["target", "gene", "genesymbol", "gene_symbol", "feature"]
    )
    if source_col is None or target_col is None:
        raise ValueError(
            "Gene set table must contain source/target columns, or recognizable "
            "synonyms such as geneset/genesymbol"
        )
    return frame.rename(columns={source_col: "source", target_col: "target"})


def _first_present(columns: dict[str, object], candidates: list[str]) -> object | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _build_rank_matrix(de: DEResults, rank_by: GSEARankBy) -> pd.DataFrame:
    frame = de.data.select(
        [
            de.target_col,
            de.feature_col,
            de.log2_fold_change_col,
            de.pvalue_col,
            de.fdr_col,
        ]
    ).to_pandas()

    lfc = frame[de.log2_fold_change_col].astype(float).to_numpy()
    match rank_by:
        case "log2_fold_change":
            values = lfc
        case "signed_pvalue":
            pvals = _clip_probabilities(frame[de.pvalue_col].astype(float).to_numpy())
            values = np.sign(lfc) * -np.log10(pvals)
        case "signed_fdr":
            fdr = _clip_probabilities(frame[de.fdr_col].astype(float).to_numpy())
            values = np.sign(lfc) * -np.log10(fdr)
        case _:
            raise ValueError(f"Unsupported GSEA rank statistic: {rank_by}")

    frame = frame.assign(_gsea_rank_value=values)
    matrix = frame.pivot(
        index=de.target_col,
        columns=de.feature_col,
        values="_gsea_rank_value",
    )
    matrix = matrix.sort_index().reindex(sorted(matrix.columns), axis=1).fillna(0.0)
    return _sanitize_rank_matrix(matrix)


def _clip_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values, nan=1.0, posinf=1.0, neginf=1.0)
    return np.clip(values, 1e-300, 1.0)


def _sanitize_rank_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    values = matrix.to_numpy(dtype=float, copy=True)
    for i in range(values.shape[0]):
        row = values[i]
        finite_mask = np.isfinite(row)
        if not finite_mask.any():
            values[i] = 0.0
            continue
        finite = row[finite_mask]
        lo = finite.min()
        hi = finite.max()
        step = max(1.0, abs(lo), abs(hi)) * 1e-6
        row[np.isnan(row)] = 0.0
        row[np.isposinf(row)] = hi + step
        row[np.isneginf(row)] = lo - step
    return pd.DataFrame(
        values, index=matrix.index.astype(str), columns=matrix.columns.astype(str)
    )


def _run_gsea(
    matrix: pd.DataFrame,
    net: pd.DataFrame,
    tmin: int,
    times: int,
    seed: int,
    verbose: bool,
) -> pd.DataFrame:
    _prepare_mpl_config()
    import decoupler as dc

    scores, _ = dc.mt.gsea(
        data=matrix,
        net=net,
        tmin=tmin,
        times=times,
        seed=seed,
        verbose=verbose,
    )
    return scores


def _prepare_mpl_config() -> None:
    # decoupler imports plotting modules at package import time; make that quiet
    # in restricted environments where HOME is not writable.
    if "MPLCONFIGDIR" not in os.environ:
        path = Path(tempfile.gettempdir()) / "cell_eval_mpl"
        path.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(path)


def _bounded_spearman(real: np.ndarray, pred: np.ndarray) -> float:
    """Return Spearman rho mapped from [-1, 1] to [0, 1]."""
    real = np.nan_to_num(real, nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    real_constant = np.allclose(real, real[0])
    pred_constant = np.allclose(pred, pred[0])
    if real_constant or pred_constant:
        return (
            1.0 if real_constant and pred_constant and np.allclose(real, pred) else 0.0
        )

    corr = spearmanr(real, pred).correlation
    if corr is None or np.isnan(corr):
        return 0.0
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))
