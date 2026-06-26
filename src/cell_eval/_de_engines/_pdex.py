from typing import Any

import anndata as ad
import pandas as pd
import polars as pl
from pdex import pdex


def build_pdex_kwargs(
    *,
    reference: str,
    groupby: str,
    threads: int,
    allow_discrete: bool,
    kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pdex_kwargs = dict(kwargs or {})
    if "reference" not in pdex_kwargs:
        pdex_kwargs["reference"] = reference
    if "groupby" not in pdex_kwargs:
        pdex_kwargs["groupby"] = groupby
    if "threads" not in pdex_kwargs:
        pdex_kwargs["threads"] = threads
    if "is_log1p" not in pdex_kwargs:
        pdex_kwargs["is_log1p"] = not allow_discrete
    return pdex_kwargs


def run_pdex_de(
    *,
    adata: ad.AnnData,
    reference: str,
    groupby: str,
    threads: int,
    allow_discrete: bool,
    kwargs: dict[str, Any],
) -> pl.DataFrame:
    pdex_kwargs = build_pdex_kwargs(
        reference=reference,
        groupby=groupby,
        threads=threads,
        allow_discrete=allow_discrete,
        kwargs=kwargs,
    )
    frame = pdex(
        adata=adata,
        mode="ref",
        **pdex_kwargs,
    )
    if isinstance(frame, pd.DataFrame):
        return pl.from_pandas(frame)
    return frame
