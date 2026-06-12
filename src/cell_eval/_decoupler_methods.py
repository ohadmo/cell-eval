"""Decoupler activity method choices used by CLI and metrics."""

from __future__ import annotations

from typing import Literal, cast, get_args

DecouplerActivityMethod = Literal["viper", "ulm", "mlm", "waggr", "zscore"]

DoRothEAMethod = Literal["viper", "ulm", "mlm", "waggr", "zscore"]
DOROTHEA_ACTIVITY_METHODS = cast(
    tuple[DoRothEAMethod, ...],
    get_args(DoRothEAMethod),
)

PROGENyMethod = Literal["ulm", "mlm", "waggr", "zscore"]
PROGENY_ACTIVITY_METHODS = cast(
    tuple[PROGENyMethod, ...],
    get_args(PROGENyMethod),
)

CollecTRIMethod = Literal["ulm", "mlm", "waggr", "zscore"]
COLLECTRI_ACTIVITY_METHODS = cast(
    tuple[CollecTRIMethod, ...],
    get_args(CollecTRIMethod),
)
