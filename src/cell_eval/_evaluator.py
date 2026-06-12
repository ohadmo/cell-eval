import logging
import multiprocessing as mp
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from pdex import pdex

from cell_eval.utils import guess_is_lognorm

from ._pipeline import MetricPipeline
from ._types import PerturbationAnndataPair, initialize_de_comparison
from .utils import _cast_float16_to_float32

logger = logging.getLogger(__name__)

FeatureNamesInput = os.PathLike[str] | str | Sequence[str]

_FEATURE_NAME_COLUMNS = (
    "gene_symbol",
    "gene_symbols",
    "gene_name",
    "gene_names",
    "genesymbol",
    "feature_name",
    "feature_names",
    "feature",
    "target",
    "gene",
)


def _available_cpus() -> int:
    """Return CPUs the current process is allowed to use.

    Uses ``os.sched_getaffinity`` on Linux so SLURM/cgroup/taskset limits are
    respected; falls back to ``mp.cpu_count`` on macOS/Windows where that API
    is unavailable (those platforms typically run locally without cgroup caps).
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return mp.cpu_count()


class MetricsEvaluator:
    """
    Evaluates benchmarking metrics of a predicted and real anndata object.

    Arguments
    =========

    adata_pred: ad.AnnData | str
        Predicted anndata object or path to anndata object.
    adata_real: ad.AnnData | str
        Real anndata object or path to anndata object.
    de_pred: pl.DataFrame | str | None = None
        Predicted differential expression results or path to differential expression results.
        If `None`, differential expression will be computed using parallel_differential_expression
    de_real: pl.DataFrame | str | None = None
        Real differential expression results or path to differential expression results.
        If `None`, differential expression will be computed using parallel_differential_expression
    control_pert: str = "non-targeting"
        Control perturbation name.
    pert_col: str = "target"
        Perturbation column name.
    num_threads: int = -1
        Number of threads for parallel differential expression.
    outdir: str = "./cell-eval-outdir"
        Output directory.
    allow_discrete: bool = False
        Allow discrete data.
    prefix: str | None = None
        Prefix for output files.
    pdex_kwargs: dict[str, Any] | None = None
        Keyword arguments for parallel_differential_expression.
        These will overwrite arguments passed to MetricsEvaluator.__init__ if they conflict.
    """

    def __init__(
        self,
        adata_pred: ad.AnnData | str,
        adata_real: ad.AnnData | str,
        de_pred: pl.DataFrame | str | None = None,
        de_real: pl.DataFrame | str | None = None,
        control_pert: str = "non-targeting",
        pert_col: str = "target",
        num_threads: int = -1,
        outdir: str = "./cell-eval-outdir",
        allow_discrete: bool = False,
        prefix: str | None = None,
        pdex_kwargs: dict[str, Any] | None = None,
        skip_de: bool = False,
        feature_names: FeatureNamesInput | None = None,
    ):
        # Enable a global string cache for categorical columns
        pl.enable_string_cache()

        if num_threads == -1:
            num_threads = _available_cpus()

        if os.path.exists(outdir):
            logger.warning(
                f"Output directory {outdir} already exists, potential overwrite occurring"
            )
        os.makedirs(outdir, exist_ok=True)

        resolved_feature_names = _resolve_feature_names(feature_names)
        self.anndata_pair = _build_anndata_pair(
            real=adata_real,
            pred=adata_pred,
            control_pert=control_pert,
            pert_col=pert_col,
            allow_discrete=allow_discrete,
            feature_names=resolved_feature_names,
        )

        if skip_de:
            self.de_comparison = None
        else:
            self.de_comparison = _build_de_comparison(
                anndata_pair=self.anndata_pair,
                de_pred=de_pred,
                de_real=de_real,
                num_threads=num_threads,
                allow_discrete=allow_discrete,
                outdir=outdir,
                prefix=prefix,
                pdex_kwargs=pdex_kwargs or {},
                feature_names=resolved_feature_names,
            )

        self.outdir = outdir
        self.prefix = prefix

    def compute(
        self,
        profile: Literal["full", "vcc", "minimal", "de", "anndata", "pds"] = "full",
        metric_configs: dict[str, dict[str, Any]] | None = None,
        skip_metrics: list[str] | None = None,
        basename: str = "results.csv",
        write_csv: bool = True,
        break_on_error: bool = False,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        pipeline = MetricPipeline(
            profile=profile,
            metric_configs=metric_configs,
            break_on_error=break_on_error,
        )
        if skip_metrics is not None:
            pipeline.skip_metrics(skip_metrics)
        pipeline.compute_de_metrics(self.de_comparison)
        pipeline.compute_anndata_metrics(self.anndata_pair)
        results = pipeline.get_results()
        agg_results = pipeline.get_agg_results()

        if write_csv:
            if self.prefix is not None:
                self.prefix = self.prefix.replace(
                    "/", "-"
                )  # some prefixes (e.g. HepG2/C3A) may have slashes in them
            if basename is not None:
                basename = basename.replace(
                    "/", "-"
                )  # some basenames (e.g. HepG2/C3A_results.csv) may have slashes in them
            outpath = os.path.join(
                self.outdir,
                f"{self.prefix}_{basename}" if self.prefix else basename,
            )
            agg_outpath = os.path.join(
                self.outdir,
                f"{self.prefix}_agg_{basename}" if self.prefix else f"agg_{basename}",
            )

            logger.info(f"Writing perturbation level metrics to {outpath}")
            results.write_csv(outpath)

            logger.info(f"Writing aggregate metrics to {agg_outpath}")
            agg_results.write_csv(agg_outpath)

        return results, agg_results


def _build_anndata_pair(
    real: ad.AnnData | str,
    pred: ad.AnnData | str,
    control_pert: str,
    pert_col: str,
    allow_discrete: bool = False,
    feature_names: Sequence[str] | None = None,
):
    if isinstance(real, str):
        logger.info(f"Reading real anndata from {real}")
        real = ad.read_h5ad(real)
    if isinstance(pred, str):
        logger.info(f"Reading pred anndata from {pred}")
        pred = ad.read_h5ad(pred)

    if feature_names is not None:
        _apply_feature_names(real, feature_names, which="real")
        _apply_feature_names(pred, feature_names, which="pred")

    # Cast float16 to float32 since NUMBA (used by pdex) does not support float16
    _cast_float16_to_float32(real, which="real")
    _cast_float16_to_float32(pred, which="pred")

    # Validate that the input is normalized and log-transformed
    _convert_to_normlog(real, which="real", allow_discrete=allow_discrete)
    _convert_to_normlog(pred, which="pred", allow_discrete=allow_discrete)

    # Build the anndata pair
    return PerturbationAnndataPair(
        real=real, pred=pred, control_pert=control_pert, pert_col=pert_col
    )


def _convert_to_normlog(
    adata: ad.AnnData,
    which: str | None = None,
    allow_discrete: bool = False,
):
    """Performs a norm-log conversion if the input is integer data (inplace).

    Will skip if the input is not integer data.
    """
    if guess_is_lognorm(adata=adata, validate=not allow_discrete):
        logger.info(
            "Input is found to be log-normalized already - skipping transformation."
        )
        return  # Input is already log-normalized

    # User specified that they want to allow discrete data
    if allow_discrete:
        if which:
            logger.info(
                f"Discovered integer data for {which}. Configuration set to allow discrete. "
                "Make sure this is intentional."
            )
        else:
            logger.info(
                "Discovered integer data. Configuration set to allow discrete. "
                "Make sure this is intentional."
            )
        return  # proceed without conversion

    # Convert the data to norm-log
    if which:
        logger.info(f"Discovered integer data for {which}. Converting to norm-log.")
    sc.pp.normalize_total(adata=adata, inplace=True)  # normalize to median
    sc.pp.log1p(adata)  # log-transform (log1p)


def _resolve_feature_names(
    feature_names: FeatureNamesInput | None,
) -> list[str] | None:
    if feature_names is None:
        return None
    if isinstance(feature_names, (str, os.PathLike)):
        path = os.fspath(feature_names)
        if not isinstance(path, str):
            raise TypeError("--feature-names path must resolve to a string path")
        names = _load_feature_names(path)
    else:
        names = list(feature_names)
    return _validate_feature_names(names)


def _load_feature_names(path: str) -> list[str]:
    feature_path = Path(path)
    suffix = feature_path.suffix.lower()
    if suffix == ".npy":
        array = np.load(feature_path, allow_pickle=True)
        if array.ndim != 1:
            raise ValueError(
                f"--feature-names file must contain a one-dimensional array: {feature_path}"
            )
        return [str(value) for value in array.tolist()]
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return _load_delimited_feature_names(feature_path, sep=sep)

    with open(feature_path) as handle:
        return [line.strip() for line in handle if line.strip()]


def _load_delimited_feature_names(path: Path, sep: str) -> list[str]:
    frame = pd.read_csv(path, sep=sep)
    columns_by_normalized_name = {
        str(column).strip().lower(): column for column in frame.columns
    }
    for column_name in _FEATURE_NAME_COLUMNS:
        if column_name in columns_by_normalized_name:
            column = columns_by_normalized_name[column_name]
            return frame[column].astype(str).tolist()

    frame_no_header = pd.read_csv(path, sep=sep, header=None)
    if frame_no_header.shape[1] == 0:
        raise ValueError(f"--feature-names file is empty: {path}")
    if frame_no_header.shape[1] > 1:
        raise ValueError(
            "--feature-names delimited files with multiple columns must include "
            f"one of these columns: {', '.join(_FEATURE_NAME_COLUMNS)}"
        )
    return frame_no_header.iloc[:, 0].astype(str).tolist()


def _validate_feature_names(names: Sequence[str]) -> list[str]:
    normalized = ["" if pd.isna(name) else str(name).strip() for name in names]
    if not normalized:
        raise ValueError("--feature-names must contain at least one feature name")
    empty_positions = [idx for idx, name in enumerate(normalized) if not name]
    if empty_positions:
        raise ValueError(
            f"--feature-names contains empty names at positions: {empty_positions[:5]}"
        )
    duplicates = pd.Series(normalized).value_counts()
    duplicates = duplicates[duplicates > 1]
    if not duplicates.empty:
        preview = ", ".join(map(str, duplicates.index[:5]))
        raise ValueError(f"--feature-names contains duplicate names: {preview}")
    return normalized


def _apply_feature_names(
    adata: ad.AnnData,
    feature_names: Sequence[str],
    which: str,
) -> None:
    if adata.n_vars != len(feature_names):
        raise ValueError(
            f"--feature-names length ({len(feature_names)}) does not match "
            f"{which} AnnData feature dimension ({adata.n_vars})"
        )
    adata.var.index = pd.Index(feature_names, dtype="str")


def _maybe_remap_de_features(
    frame: pl.DataFrame,
    feature_names: Sequence[str] | None,
) -> pl.DataFrame:
    if feature_names is None or "feature" not in frame.columns:
        return frame

    frame = frame.with_columns(pl.col("feature").cast(pl.Utf8))
    unique_features = frame.select(pl.col("feature").unique())["feature"].to_list()
    feature_name_set = set(feature_names)
    unknown_feature_names = [
        feature for feature in unique_features if feature not in feature_name_set
    ]
    if not unknown_feature_names:
        return frame

    try:
        feature_indices = [int(feature) for feature in unique_features]
    except ValueError as error:
        preview = ", ".join(map(str, unknown_feature_names[:5]))
        raise ValueError(
            "DE feature values do not match --feature-names and are not numeric "
            f"zero-based feature indices. Examples: {preview}"
        ) from error

    n_features = len(feature_names)
    out_of_range = [idx for idx in feature_indices if idx < 0 or idx >= n_features]
    if out_of_range:
        raise ValueError(
            "DE feature indices are outside the range covered by --feature-names. "
            f"Examples: {out_of_range[:5]}"
        )

    mapping = pl.DataFrame(
        {
            "feature": [str(idx) for idx in range(n_features)],
            "_cell_eval_feature_name": list(feature_names),
        }
    )
    return (
        frame.join(mapping, on="feature", how="left")
        .with_columns(
            pl.coalesce(["_cell_eval_feature_name", "feature"]).alias("feature")
        )
        .drop("_cell_eval_feature_name")
    )


def _ensure_polars_dataframe(frame: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return pl.from_pandas(frame)
    return frame


def _build_de_comparison(
    anndata_pair: PerturbationAnndataPair | None = None,
    de_pred: pl.DataFrame | str | None = None,
    de_real: pl.DataFrame | str | None = None,
    num_threads: int = 1,
    allow_discrete: bool = False,
    outdir: str | None = None,
    prefix: str | None = None,
    pdex_kwargs: dict[str, Any] | None = None,
    feature_names: Sequence[str] | None = None,
):
    return initialize_de_comparison(
        real=_load_or_build_de(
            mode="real",
            de_path=de_real,
            anndata_pair=anndata_pair,
            num_threads=num_threads,
            allow_discrete=allow_discrete,
            outdir=outdir,
            prefix=prefix,
            pdex_kwargs=pdex_kwargs or {},
            feature_names=feature_names,
        ),
        pred=_load_or_build_de(
            mode="pred",
            de_path=de_pred,
            anndata_pair=anndata_pair,
            num_threads=num_threads,
            allow_discrete=allow_discrete,
            outdir=outdir,
            prefix=prefix,
            pdex_kwargs=pdex_kwargs or {},
            feature_names=feature_names,
        ),
    )


def _build_pdex_kwargs(
    reference: str,
    groupby: str,
    threads: int,
    allow_discrete: bool,
    pdex_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pdex_kwargs = pdex_kwargs or {}
    if "reference" not in pdex_kwargs:
        pdex_kwargs["reference"] = reference
    if "groupby" not in pdex_kwargs:
        pdex_kwargs["groupby"] = groupby
    if "threads" not in pdex_kwargs:
        pdex_kwargs["threads"] = threads
    if "is_log1p" not in pdex_kwargs:
        if allow_discrete:
            pdex_kwargs["is_log1p"] = False
        else:
            pdex_kwargs["is_log1p"] = True
    return pdex_kwargs


def _load_or_build_de(
    mode: Literal["pred", "real"],
    de_path: pl.DataFrame | str | None = None,
    anndata_pair: PerturbationAnndataPair | None = None,
    num_threads: int = 1,
    outdir: str | None = None,
    prefix: str | None = None,
    allow_discrete: bool = False,
    pdex_kwargs: dict[str, Any] | None = None,
    feature_names: Sequence[str] | None = None,
) -> pl.DataFrame:
    if de_path is None:
        if anndata_pair is None:
            raise ValueError("anndata_pair must be provided if de_path is not provided")
        logger.info(f"Computing DE for {mode} data")
        pdex_kwargs = _build_pdex_kwargs(
            reference=anndata_pair.control_pert,
            groupby=anndata_pair.pert_col,
            threads=num_threads,
            allow_discrete=allow_discrete,
            pdex_kwargs=pdex_kwargs or {},
        )
        logger.info(f"Using the following pdex kwargs: {pdex_kwargs}")
        frame = _ensure_polars_dataframe(
            pdex(
                adata=anndata_pair.real if mode == "real" else anndata_pair.pred,
                mode="ref",
                **pdex_kwargs,
            )
        )
        frame = _maybe_remap_de_features(frame, feature_names)
        if outdir is not None:
            if prefix is not None:
                prefix = prefix.replace(
                    "/", "-"
                )  # some prefixes (e.g. HepG2/C3A) may have slashes in them
            pathname = f"{mode}_de.csv" if not prefix else f"{prefix}_{mode}_de.csv"
            logger.info(f"Writing {mode} DE results to: {pathname}")
            frame.write_csv(os.path.join(outdir, pathname))

        return frame
    elif isinstance(de_path, str):
        logger.info(f"Reading {mode} DE results from {de_path}")
        if pdex_kwargs:
            logger.warning("pdex_kwargs are ignored when reading from a CSV file")
        return _maybe_remap_de_features(
            pl.read_csv(
                de_path,
                schema_overrides={
                    "target": pl.Utf8,
                    "feature": pl.Utf8,
                },
            ),
            feature_names,
        )
    elif isinstance(de_path, pl.DataFrame):
        if pdex_kwargs:
            logger.warning("pdex_kwargs are ignored when reading from a CSV file")
        return _maybe_remap_de_features(de_path, feature_names)
    elif isinstance(de_path, pd.DataFrame):
        if pdex_kwargs:
            logger.warning("pdex_kwargs are ignored when reading from a CSV file")
        return _maybe_remap_de_features(pl.from_pandas(de_path), feature_names)
    else:
        raise TypeError(f"Unexpected type for de_path: {type(de_path)}")
