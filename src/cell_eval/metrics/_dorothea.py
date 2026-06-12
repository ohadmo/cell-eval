"""DoRothEA transcription-factor activity concordance metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import pandas as pd
import polars as pl

from .._types import DEComparison
from .._decoupler_methods import DOROTHEA_ACTIVITY_METHODS, DoRothEAMethod
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

DoRothEALicense = Literal["academic", "commercial", "nonprofit"]
DoRothEALevel = Literal["A", "B", "C", "D"]


class DoRothEAActivitySpearman:
    """Compare real and predicted DoRothEA TF activities per perturbation."""

    def __init__(
        self,
        net: pd.DataFrame | pl.DataFrame | None = None,
        method: DoRothEAMethod = "viper",
        rank_by: GSEARankBy = "log2_fold_change",
        organism: str = "human",
        levels: str | Sequence[str] | None = ("A", "B", "C"),
        license: DoRothEALicense = "commercial",
        tmin: int = 5,
        min_regulators: int = 2,
        verbose: bool = False,
    ) -> None:
        self.net = net
        self.method = validate_decoupler_activity_method(
            method,
            DOROTHEA_ACTIVITY_METHODS,
        )
        self.rank_by = rank_by
        self.organism = organism
        self.levels = _normalize_dorothea_levels(levels)
        self.license = license
        self.tmin = tmin
        self.min_regulators = min_regulators
        self.verbose = verbose

    def __call__(self, data: DEComparison) -> dict[str, float]:
        net = _load_dorothea_net(
            net=self.net,
            organism=self.organism,
            levels=self.levels,
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
                f"Need at least {self.min_regulators} shared DoRothEA regulators "
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


def _load_dorothea_net(
    net: pd.DataFrame | pl.DataFrame | None,
    organism: str,
    levels: Sequence[DoRothEALevel],
    license: DoRothEALicense,
    verbose: bool,
) -> pd.DataFrame:
    if net is not None:
        frame = net.to_pandas() if isinstance(net, pl.DataFrame) else net.copy()
    else:
        _prepare_mpl_config()
        import decoupler as dc

        # Decoupler accepts levels as str/list/None; normalize internally for
        # CLI consistency, then pass a list to match the public API.
        frame = dc.op.dorothea(
            organism=organism,
            levels=list(levels),
            license=license,
            verbose=verbose,
        )

    frame = _normalize_dorothea_columns(frame)
    keep_cols = ["source", "target"] + (["weight"] if "weight" in frame.columns else [])
    frame = frame.loc[:, keep_cols].dropna().drop_duplicates(["source", "target"])
    if frame.empty:
        raise ValueError("DoRothEA network is empty after normalization")
    return frame


def _normalize_dorothea_columns(frame: pd.DataFrame) -> pd.DataFrame:
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
            "DoRothEA network must contain source/target columns, or recognizable "
            "synonyms such as tf/target_genesymbol"
        )
    return frame


def _normalize_dorothea_levels(
    levels: str | Sequence[str] | None,
) -> tuple[DoRothEALevel, ...]:
    if levels is None:
        levels = ("A", "B", "C")
    if isinstance(levels, str):
        raw = levels.replace(",", " ").split()
        if len(raw) == 1 and len(raw[0]) > 1:
            raw = list(raw[0])
    else:
        raw = list(levels)

    normalized = tuple(str(level).upper() for level in raw)
    allowed = {"A", "B", "C", "D"}
    if not normalized or any(level not in allowed for level in normalized):
        raise ValueError("DoRothEA levels must contain one or more of: A, B, C, D")
    return cast(tuple[DoRothEALevel, ...], normalized)
