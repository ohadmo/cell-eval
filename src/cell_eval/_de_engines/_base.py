from typing import Any, Literal

import anndata as ad
import polars as pl

DEMethod = Literal["pdex", "memento"]


def run_de(
    *,
    method: DEMethod,
    adata: ad.AnnData,
    reference: str,
    groupby: str,
    threads: int,
    allow_discrete: bool,
    kwargs: dict[str, Any],
) -> pl.DataFrame:
    match method:
        case "pdex":
            from ._pdex import run_pdex_de

            return run_pdex_de(
                adata=adata,
                reference=reference,
                groupby=groupby,
                threads=threads,
                allow_discrete=allow_discrete,
                kwargs=kwargs,
            )
        case "memento":
            from ._memento import run_memento_de

            return run_memento_de(
                adata=adata,
                reference=reference,
                groupby=groupby,
                threads=threads,
                kwargs=kwargs,
            )
        case _:
            raise ValueError(f"Unsupported DE method: {method}")
