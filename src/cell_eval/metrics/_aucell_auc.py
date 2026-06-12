"""Gene-set concordance metrics based on AUCell AUC scores."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import polars as pl

from .._types import DEComparison
from ._gsea_nes import (
    GSEARankBy,
    _bounded_spearman,
    _build_rank_matrix,
    _load_gene_sets,
    _prepare_mpl_config,
)

AUCellDirection = Literal["up", "down", "both"]


class AUCellAUCSpearman:
    """Compare real and predicted AUCell gene-set AUCs per perturbation."""

    def __init__(
        self,
        gene_set_path: str | None = None,
        net: pd.DataFrame | pl.DataFrame | None = None,
        rank_by: GSEARankBy = "log2_fold_change",
        n_up: int | None = None,
        direction: AUCellDirection = "both",
        tmin: int = 5,
        min_gene_sets: int = 2,
        verbose: bool = False,
    ) -> None:
        if direction not in {"up", "down", "both"}:
            raise ValueError("AUCell direction must be one of: up, down, both")
        if n_up is not None and n_up <= 1:
            raise ValueError("AUCell n_up must be greater than 1, or None for auto")
        self.gene_set_path = gene_set_path
        self.net = net
        self.rank_by = rank_by
        self.n_up = n_up
        self.direction = direction
        self.tmin = tmin
        self.min_gene_sets = min_gene_sets
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

        real_scores = _run_directional_aucell(
            real_rank.loc[:, shared_features],
            net=net,
            n_up=self.n_up,
            direction=self.direction,
            tmin=self.tmin,
            verbose=self.verbose,
        )
        pred_scores = _run_directional_aucell(
            pred_rank.loc[:, shared_features],
            net=net,
            n_up=self.n_up,
            direction=self.direction,
            tmin=self.tmin,
            verbose=self.verbose,
        )

        shared_gene_set_scores = [
            g for g in real_scores.columns if g in pred_scores.columns
        ]
        n_gene_sets = _count_gene_sets(shared_gene_set_scores, direction=self.direction)
        if n_gene_sets < self.min_gene_sets:
            raise ValueError(
                f"Need at least {self.min_gene_sets} shared AUCell gene-set scores "
                f"after pruning, found {n_gene_sets}"
            )

        scores: dict[str, float] = {}
        for pert in data.iter_perturbations():
            real_vec = real_scores.loc[str(pert), shared_gene_set_scores].to_numpy(
                dtype=float
            )
            pred_vec = pred_scores.loc[str(pert), shared_gene_set_scores].to_numpy(
                dtype=float
            )
            scores[str(pert)] = _bounded_spearman(real_vec, pred_vec)
        return scores


def _run_directional_aucell(
    matrix: pd.DataFrame,
    net: pd.DataFrame,
    n_up: int | None,
    direction: AUCellDirection,
    tmin: int,
    verbose: bool,
) -> pd.DataFrame:
    parts = []
    if direction in {"up", "both"}:
        up = _run_aucell(matrix, net=net, n_up=n_up, tmin=tmin, verbose=verbose)
        if direction == "both":
            up = up.add_suffix("_up")
        parts.append(up)
    if direction in {"down", "both"}:
        down = _run_aucell(-matrix, net=net, n_up=n_up, tmin=tmin, verbose=verbose)
        if direction == "both":
            down = down.add_suffix("_down")
        parts.append(down)
    return pd.concat(parts, axis=1)


def _count_gene_sets(columns: list[str], direction: AUCellDirection) -> int:
    if direction != "both":
        return len(columns)
    return len({column.removesuffix("_up").removesuffix("_down") for column in columns})


def _run_aucell(
    matrix: pd.DataFrame,
    net: pd.DataFrame,
    n_up: int | None,
    tmin: int,
    verbose: bool,
) -> pd.DataFrame:
    _prepare_mpl_config()
    import decoupler as dc

    scores, _ = dc.mt.aucell(
        data=matrix,
        net=net,
        tmin=tmin,
        n_up=n_up,
        verbose=verbose,
    )
    return scores
