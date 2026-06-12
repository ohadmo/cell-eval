"""CollecTRI transcription-factor activity concordance metrics."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import polars as pl

from .._decoupler_methods import COLLECTRI_ACTIVITY_METHODS, CollecTRIMethod
from .._types import DEComparison
from ._decoupler_activity import (
    run_decoupler_activity,
    validate_decoupler_activity_method,
)
from ._gsea_nes import (
    GSEARankBy,
    _bounded_spearman,
    _build_rank_matrix,
    _prepare_mpl_config,
)

CollecTRILicense = Literal["academic", "commercial", "nonprofit"]


class CollecTRIActivitySpearman:
    """Compare real and predicted CollecTRI TF activities per perturbation."""

    def __init__(
        self,
        net: pd.DataFrame | pl.DataFrame | None = None,
        method: CollecTRIMethod = "ulm",
        rank_by: GSEARankBy = "log2_fold_change",
        organism: str = "human",
        remove_complexes: bool = False,
        license: CollecTRILicense = "commercial",
        tmin: int = 5,
        min_regulators: int = 2,
        verbose: bool = False,
    ) -> None:
        self.net = net
        self.method = validate_decoupler_activity_method(
            method,
            COLLECTRI_ACTIVITY_METHODS,
        )
        self.rank_by = rank_by
        self.organism = organism
        self.remove_complexes = remove_complexes
        self.license = license
        self.tmin = tmin
        self.min_regulators = min_regulators
        self.verbose = verbose

    def __call__(self, data: DEComparison) -> dict[str, float]:
        net = _load_collectri_net(
            net=self.net,
            organism=self.organism,
            remove_complexes=self.remove_complexes,
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

        real_scores = run_decoupler_activity(
            real_rank.loc[:, shared_features],
            net=net,
            method=self.method,
            tmin=self.tmin,
            verbose=self.verbose,
        )
        pred_scores = run_decoupler_activity(
            pred_rank.loc[:, shared_features],
            net=net,
            method=self.method,
            tmin=self.tmin,
            verbose=self.verbose,
        )

        shared_regulators = [r for r in real_scores.columns if r in pred_scores.columns]
        if len(shared_regulators) < self.min_regulators:
            raise ValueError(
                f"Need at least {self.min_regulators} shared CollecTRI regulators "
                f"after pruning, found {len(shared_regulators)}"
            )

        scores: dict[str, float] = {}
        for pert in data.iter_perturbations():
            real_vec = real_scores.loc[str(pert), shared_regulators].to_numpy(
                dtype=float
            )
            pred_vec = pred_scores.loc[str(pert), shared_regulators].to_numpy(
                dtype=float
            )
            scores[str(pert)] = _bounded_spearman(real_vec, pred_vec)
        return scores


def _load_collectri_net(
    net: pd.DataFrame | pl.DataFrame | None,
    organism: str,
    remove_complexes: bool,
    license: CollecTRILicense,
    verbose: bool,
) -> pd.DataFrame:
    if net is not None:
        frame = net.to_pandas() if isinstance(net, pl.DataFrame) else net.copy()
    else:
        _prepare_mpl_config()
        import decoupler as dc

        frame = dc.op.collectri(
            organism=organism,
            remove_complexes=remove_complexes,
            license=license,
            verbose=verbose,
        )

    frame = _normalize_collectri_columns(frame)
    keep_cols = ["source", "target"] + (["weight"] if "weight" in frame.columns else [])
    frame = frame.loc[:, keep_cols].dropna().drop_duplicates(["source", "target"])
    if frame.empty:
        raise ValueError("CollecTRI network is empty after normalization")
    return frame


def _normalize_collectri_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {str(c).lower(): c for c in frame.columns}
    rename = {}
    if "source" not in columns:
        if "tf" in columns:
            rename[columns["tf"]] = "source"
        elif "regulator" in columns:
            rename[columns["regulator"]] = "source"
        elif "source_genesymbol" in columns:
            rename[columns["source_genesymbol"]] = "source"
    if "target" not in columns:
        if "gene" in columns:
            rename[columns["gene"]] = "target"
        elif "genesymbol" in columns:
            rename[columns["genesymbol"]] = "target"
        elif "target_genesymbol" in columns:
            rename[columns["target_genesymbol"]] = "target"
    frame = frame.rename(columns=rename)
    if "source" not in frame.columns or "target" not in frame.columns:
        raise ValueError(
            "CollecTRI network must contain source/target columns, or recognizable "
            "synonyms such as tf/target_genesymbol"
        )
    return frame
