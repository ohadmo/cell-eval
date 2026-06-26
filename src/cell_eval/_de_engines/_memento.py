import logging
from collections.abc import Sequence
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse

logger = logging.getLogger(__name__)

TREATMENT_COL = "_cell_eval_memento_treatment"
CAPTURE_RATE_COL = "_cell_eval_memento_capture_rate"


def run_memento_de(
    *,
    adata: ad.AnnData,
    reference: str,
    groupby: str,
    threads: int,
    kwargs: dict[str, Any],
) -> pl.DataFrame:
    try:
        import memento  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Memento DE requires the optional dependency `memento-de`. "
            "Install it with `pip install memento-de` or `pip install cell-eval[memento]`."
        ) from exc

    config = _MementoConfig.from_kwargs(kwargs)
    obs = cast(pd.DataFrame, adata.obs)
    if groupby not in obs.columns:
        raise ValueError(f"Column '{groupby}' not found in adata.obs")

    groups = _get_groups(cast(pd.Series, obs[groupby]), reference)
    if not groups:
        raise ValueError(
            f"No perturbation groups found outside reference '{reference}'"
        )

    results: list[pd.DataFrame] = []
    for target in groups:
        logger.info("Computing Memento DE for %s vs %s", target, reference)
        subset = _build_binary_subset(
            adata=adata,
            groupby=groupby,
            reference=reference,
            target=target,
            config=config,
        )

        label_columns = list(
            dict.fromkeys(
                [TREATMENT_COL, *config.replicate_cols, *config.covariate_cols]
            )
        )
        memento.setup_memento(
            subset,
            q_column=CAPTURE_RATE_COL,
            filter_mean_thresh=config.filter_mean_thresh,
            trim_percent=config.trim_percent,
            shrinkage=config.shrinkage,
            num_bins=config.num_bins,
            min_cell_count=config.min_cell_count,
            estimator_type=config.estimator_type,
        )
        memento.create_groups(subset, label_columns=label_columns)
        memento.compute_1d_moments(
            subset,
            min_perc_group=config.min_perc_group,
            gene_list=config.gene_list,
        )

        sample_meta = memento.get_groups(subset)
        treatment = sample_meta[[TREATMENT_COL]]
        covariate = _build_covariate_frame(
            sample_meta=sample_meta,
            replicate_cols=config.replicate_cols,
            covariate_cols=config.covariate_cols,
        )

        memento.ht_1d_moments(
            subset,
            treatment=treatment,
            covariate=covariate,
            num_boot=config.num_boot,
            verbose=config.verbose,
            num_cpus=threads,
            approx=config.approx,
            resample_rep=config.resample_rep,
        )
        target_result = memento.get_1d_ht_result(subset)
        target_result["target"] = target
        results.append(target_result)

    if not results:
        raise ValueError("Memento did not return any DE results")

    return _standardize_memento_results(pd.concat(results, ignore_index=True))


class _MementoConfig:
    def __init__(
        self,
        *,
        counts_layer: str | None = None,
        input_is_log1p: bool = False,
        capture_rate: float | None = None,
        capture_rate_col: str | None = None,
        num_boot: int = 5000,
        filter_mean_thresh: float = 0.07,
        trim_percent: float = 0.1,
        shrinkage: float = 0.5,
        num_bins: int = 30,
        min_cell_count: int = 10,
        min_perc_group: float = 0.9,
        estimator_type: str = "hyper_relative",
        replicate_cols: Sequence[str] = (),
        covariate_cols: Sequence[str] = (),
        gene_list: Sequence[str] | None = None,
        approx: str = "norm",
        resample_rep: bool = False,
        verbose: int = 1,
    ) -> None:
        self.counts_layer = counts_layer
        self.input_is_log1p = input_is_log1p
        self.capture_rate = capture_rate
        self.capture_rate_col = capture_rate_col
        self.num_boot = num_boot
        self.filter_mean_thresh = filter_mean_thresh
        self.trim_percent = trim_percent
        self.shrinkage = shrinkage
        self.num_bins = num_bins
        self.min_cell_count = min_cell_count
        self.min_perc_group = min_perc_group
        self.estimator_type = estimator_type
        self.replicate_cols = list(replicate_cols)
        self.covariate_cols = list(covariate_cols)
        self.gene_list = list(gene_list) if gene_list is not None else None
        self.approx = approx
        self.resample_rep = resample_rep
        self.verbose = verbose

        if self.capture_rate is None and self.capture_rate_col is None:
            raise ValueError(
                "Memento requires either `capture_rate` or `capture_rate_col`."
            )
        if self.capture_rate is not None and not (0 < self.capture_rate < 1):
            raise ValueError("Memento `capture_rate` must be between 0 and 1.")

    @classmethod
    def from_kwargs(cls, kwargs: dict[str, Any]) -> "_MementoConfig":
        remaining = dict(kwargs)
        gene_list = remaining.pop("gene_list", None)
        config = cls(
            counts_layer=remaining.pop("counts_layer", None),
            input_is_log1p=remaining.pop("input_is_log1p", False),
            capture_rate=remaining.pop("capture_rate", None),
            capture_rate_col=remaining.pop("capture_rate_col", None),
            num_boot=remaining.pop("num_boot", 5000),
            filter_mean_thresh=remaining.pop("filter_mean_thresh", 0.07),
            trim_percent=remaining.pop("trim_percent", 0.1),
            shrinkage=remaining.pop("shrinkage", 0.5),
            num_bins=remaining.pop("num_bins", 30),
            min_cell_count=remaining.pop("min_cell_count", 10),
            min_perc_group=remaining.pop("min_perc_group", 0.9),
            estimator_type=remaining.pop("estimator_type", "hyper_relative"),
            replicate_cols=_split_columns(remaining.pop("replicate_cols", ())),
            covariate_cols=_split_columns(remaining.pop("covariate_cols", ())),
            gene_list=_split_columns(gene_list) if gene_list is not None else None,
            approx=remaining.pop("approx", "norm"),
            resample_rep=remaining.pop("resample_rep", False),
            verbose=remaining.pop("verbose", 1),
        )
        if remaining:
            raise ValueError(
                "Unexpected memento kwargs: {}".format(", ".join(sorted(remaining)))
            )
        return config


def _split_columns(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(value)


def _get_groups(labels: pd.Series, reference: str) -> list[str]:
    groups = pd.Series(labels).dropna().astype(str)
    groups = groups[(groups != "") & (groups != reference)].unique().tolist()
    return sorted(groups)


def _build_binary_subset(
    *,
    adata: ad.AnnData,
    groupby: str,
    reference: str,
    target: str,
    config: _MementoConfig,
) -> ad.AnnData:
    full_obs = cast(pd.DataFrame, adata.obs)
    labels = full_obs[groupby].astype(str)
    mask = labels.isin([reference, target]).to_numpy()
    obs = full_obs.loc[mask].copy()
    obs[TREATMENT_COL] = (obs[groupby].astype(str) == target).astype(int)
    obs[CAPTURE_RATE_COL] = _get_capture_rate(obs, config)

    required_cols = [*config.replicate_cols, *config.covariate_cols]
    missing_cols = [col for col in required_cols if col not in obs.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in adata.obs: {missing_cols}")

    matrix = _extract_matrix(adata, mask, config)
    return ad.AnnData(X=matrix, obs=obs, var=cast(pd.DataFrame, adata.var).copy())


def _get_capture_rate(obs: pd.DataFrame, config: _MementoConfig) -> np.ndarray | float:
    if config.capture_rate_col is None:
        if config.capture_rate is None:
            raise ValueError("Memento requires `capture_rate` when no column is set.")
        return float(config.capture_rate)
    if config.capture_rate_col not in obs.columns:
        raise ValueError(f"Column '{config.capture_rate_col}' not found in adata.obs")
    capture_rate = obs[config.capture_rate_col].astype(float).to_numpy()
    if np.any((capture_rate <= 0) | (capture_rate >= 1)):
        raise ValueError("All Memento capture-rate values must be between 0 and 1.")
    return capture_rate


def _extract_matrix(
    adata: ad.AnnData, mask: np.ndarray, config: _MementoConfig
) -> sparse.csr_matrix:
    source: Any = adata.layers[config.counts_layer] if config.counts_layer else adata.X
    if source is None:
        raise ValueError("AnnData object does not contain an expression matrix")

    matrix = source[mask, :]
    to_memory = getattr(matrix, "to_memory", None)
    if callable(to_memory):
        matrix = to_memory()
    if sparse.issparse(matrix):
        result = matrix.tocsr(copy=True)
    else:
        result = sparse.csr_matrix(np.asarray(matrix))

    if config.input_is_log1p:
        logger.warning(
            "Using expm1-transformed log1p values as Memento input. Prefer raw counts "
            "via `counts_layer` when available."
        )
        result.data = np.expm1(result.data)

    return result.astype(np.float64)


def _build_covariate_frame(
    *,
    sample_meta: pd.DataFrame,
    replicate_cols: Sequence[str],
    covariate_cols: Sequence[str],
) -> pd.DataFrame | None:
    cols = list(covariate_cols) or list(replicate_cols)
    if not cols:
        return None

    pieces = []
    for col in cols:
        series = sample_meta[col]
        if pd.api.types.is_numeric_dtype(series):
            pieces.append(series.astype(float).to_frame())
        else:
            pieces.append(pd.get_dummies(series.astype("category"), prefix=col))
    return pd.concat(pieces, axis=1)


def _standardize_memento_results(result: pd.DataFrame) -> pl.DataFrame:
    result = result.copy()
    result["feature"] = result["gene"]
    result["p_value"] = result["de_pval"]
    result["fdr"] = result.groupby("target", sort=False)["p_value"].transform(
        _benjamini_hochberg
    )
    result["log2_fold_change"] = result["de_coef"] / np.log(2)
    result["abs_log2_fold_change"] = result["log2_fold_change"].abs()
    if "dv_pval" in result.columns:
        result["dv_fdr"] = result.groupby("target", sort=False)["dv_pval"].transform(
            _benjamini_hochberg
        )

    columns = [
        "target",
        "feature",
        "log2_fold_change",
        "abs_log2_fold_change",
        "p_value",
        "fdr",
        "de_coef",
        "de_se",
        "de_pval",
        "dv_coef",
        "dv_se",
        "dv_pval",
        "dv_fdr",
    ]
    return pl.from_pandas(result[[col for col in columns if col in result.columns]])


def _benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    fdr = np.ones_like(p, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return fdr

    valid_p = np.clip(p[valid], 0.0, 1.0)
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    n = ranked.size
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    unsorted = np.empty_like(adjusted)
    unsorted[order] = adjusted
    fdr[valid] = unsorted
    return fdr
