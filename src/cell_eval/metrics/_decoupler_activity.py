"""Shared Decoupler activity-inference helpers."""

from __future__ import annotations

from typing import TypeVar, cast

import pandas as pd

from .._decoupler_methods import DecouplerActivityMethod
from ._gsea_nes import _prepare_mpl_config

ActivityMethodT = TypeVar("ActivityMethodT", bound=str)


def validate_decoupler_activity_method(
    method: str,
    allowed_methods: tuple[ActivityMethodT, ...],
) -> ActivityMethodT:
    if method not in allowed_methods:
        methods = ", ".join(allowed_methods)
        raise ValueError(f"Activity method must be one of: {methods}")
    return cast(ActivityMethodT, method)


def run_decoupler_activity(
    matrix: pd.DataFrame,
    net: pd.DataFrame,
    method: DecouplerActivityMethod,
    tmin: int,
    verbose: bool,
) -> pd.DataFrame:
    _prepare_mpl_config()
    import decoupler as dc

    kwargs = {}
    if method in {"ulm", "mlm"}:
        kwargs["tval"] = True
    elif method == "waggr":
        # We use only activity scores for concordance; skip permutation p-values.
        kwargs["times"] = 0

    scores, _ = getattr(dc.mt, method)(
        data=matrix,
        net=net,
        tmin=tmin,
        verbose=verbose,
        **kwargs,
    )
    return scores
