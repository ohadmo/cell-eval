"""Array metrics module."""

from logging import getLogger
from typing import Any, Callable, Literal, Sequence, cast

import anndata as ad
import networkit as nk
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.metrics as skm
from scipy.sparse import issparse
from scipy.stats import pearsonr
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from .._types import PerturbationAnndataPair

logger = getLogger(__name__)

LEIDEN_ITERATIONS = 2
LEIDEN_RANDOM_STATE = 0
LEIDEN_RANDOMIZE = True
LEIDEN_THREADS = 1


def pearson_delta(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute Pearson correlation between mean differences from control."""
    return _generic_evaluation(
        data,
        pearsonr,
        use_delta=True,
        embed_key=embed_key,
    )


def mse(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean squared error of each perturbation from control."""
    return _generic_evaluation(
        data, skm.mean_squared_error, use_delta=False, embed_key=embed_key
    )


def mae(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean absolute error of each perturbation from control."""
    return _generic_evaluation(
        data, skm.mean_absolute_error, use_delta=False, embed_key=embed_key
    )


def mse_delta(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean squared error of each perturbation-control delta."""
    return _generic_evaluation(
        data, skm.mean_squared_error, use_delta=True, embed_key=embed_key
    )


def mae_delta(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean absolute error of each perturbation-control delta."""
    return _generic_evaluation(
        data, skm.mean_absolute_error, use_delta=True, embed_key=embed_key
    )


def edistance(
    data: PerturbationAnndataPair,
    embed_key: str | None = None,
    metric: str = "euclidean",
    **kwargs,
) -> float:
    """Compute Euclidean distance of each perturbation-control delta."""

    def _edistance(
        x: np.ndarray,
        y: np.ndarray,
        metric: str = "euclidean",
        precomp_sigma_y: float | None = None,
        **kwargs,
    ) -> float:
        sigma_x = skm.pairwise_distances(x, metric=metric, **kwargs).mean()
        sigma_y = (
            precomp_sigma_y
            if precomp_sigma_y is not None
            else skm.pairwise_distances(y, metric=metric, **kwargs).mean()
        )
        delta = skm.pairwise_distances(x, y, metric=metric, **kwargs).mean()
        return 2 * delta - sigma_x - sigma_y

    d_real = np.zeros(data.perts.size)
    d_pred = np.zeros(data.perts.size)

    # Precompute sigma for control data (reused by all perturbations)
    logger.info("Precomputing sigma for control data (real)")
    precomp_sigma_real = skm.pairwise_distances(
        data.ctrl_matrix(which="real", embed_key=embed_key), metric=metric, **kwargs
    ).mean()

    logger.info("Precomputing sigma for control data (pred)")
    precomp_sigma_pred = skm.pairwise_distances(
        data.ctrl_matrix(which="pred", embed_key=embed_key), metric=metric, **kwargs
    ).mean()

    for idx, delta in enumerate(data.iter_cell_arrays(embed_key=embed_key)):
        d_real[idx] = _edistance(
            delta.pert_real,
            delta.ctrl_real,
            precomp_sigma_y=precomp_sigma_real,
            metric=metric,
            **kwargs,
        )
        d_pred[idx] = _edistance(
            delta.pert_pred,
            delta.ctrl_pred,
            precomp_sigma_y=precomp_sigma_pred,
            metric=metric,
            **kwargs,
        )

    return pearsonr(d_real, d_pred).correlation


def discrimination_score(
    data: PerturbationAnndataPair,
    metric: str = "l1",
    embed_key: str | None = None,
    exclude_target_gene: bool = True,
) -> dict[str, float]:
    """Base implementation for discrimination score computation.

    Best score is 1.0 - worst score is 0.0.

    Args:
        data: PerturbationAnndataPair containing real and predicted data
        embed_key: Key for embedding data in obsm, None for expression data
        metric: Metric for distance calculation (e.g., "l1", "l2", see `scipy.metrics.pairwise.distance_metrics`)
        exclude_target_gene: Whether to exclude target gene from calculation

    Returns:
        Dictionary mapping perturbation names to normalized ranks
    """
    if metric == "l1" or metric == "manhattan" or metric == "cityblock":
        # Ignore the embedding key for L1
        embed_key = None

    # Compute perturbation effects for all perturbations
    real_effects = np.vstack(
        [
            d.perturbation_effect(which="real", abs=False)
            for d in data.iter_bulk_arrays(embed_key=embed_key)
        ]
    )
    pred_effects = np.vstack(
        [
            d.perturbation_effect(which="pred", abs=False)
            for d in data.iter_bulk_arrays(embed_key=embed_key)
        ]
    )

    norm_ranks = {}
    for p_idx, p in enumerate(data.perts):
        # Determine which features to include in the comparison
        if exclude_target_gene and not embed_key:
            # For expression data, exclude the target gene
            include_mask = np.flatnonzero(data.genes != p)
        else:
            # For embedding data or when not excluding target gene, use all features
            include_mask = np.ones(real_effects.shape[1], dtype=bool)

        # Compute distances to all real effects
        distances = skm.pairwise_distances(
            real_effects[
                :, include_mask
            ],  # compare to all real effects across perturbations
            pred_effects[p_idx, include_mask].reshape(
                1, -1
            ),  # select pred effect for current perturbation
            metric=metric,
        ).flatten()

        # Sort by distance (ascending - lower distance = better match)
        sorted_indices = np.argsort(distances)

        # Find rank of the correct perturbation
        p_index = np.flatnonzero(data.perts == p)[0]
        rank = np.flatnonzero(sorted_indices == p_index)[0]

        # Normalize rank by total number of perturbations
        norm_rank = rank / data.perts.size
        norm_ranks[str(p)] = 1 - norm_rank

    return norm_ranks


def _generic_evaluation(
    data: PerturbationAnndataPair,
    func: Callable[[np.ndarray, np.ndarray], float],
    use_delta: bool = False,
    embed_key: str | None = None,
) -> dict[str, float]:
    """Generic evaluation function for anndata pair."""
    res = {}
    for bulk_array in data.iter_bulk_arrays(embed_key=embed_key):
        if use_delta:
            x = bulk_array.perturbation_effect(which="pred", abs=False)
            y = bulk_array.perturbation_effect(which="real", abs=False)
        else:
            x = bulk_array.pert_pred
            y = bulk_array.pert_real

        result = func(x, y)
        if isinstance(result, tuple):
            result = result[0]

        res[bulk_array.key] = float(result)

    return res


def _networkit_graph_from_adjacency(adjacency: Any) -> Any:
    """Convert a Scanpy neighbor graph to a weighted undirected NetworKit graph."""
    n_nodes = adjacency.shape[0]
    graph = nk.Graph(n_nodes, weighted=True, directed=False)

    # Keep explicit insertion instead of GraphFromCoo: with NetworKit 11.2.1,
    # SciPy sparse matrices in coordinate format produced unit weights locally,
    # while tuple input segfaulted under Python 3.14/SciPy 1.17. Only adding
    # edges whose source node index is smaller than the target node index avoids
    # duplicate undirected edges from Scanpy's symmetric connectivity matrix.
    if issparse(adjacency):
        coo_adjacency = adjacency.tocoo()
        for src, dst, weight in zip(
            coo_adjacency.row, coo_adjacency.col, coo_adjacency.data
        ):
            if src < dst and weight != 0:
                graph.addEdge(int(src), int(dst), float(weight))
        return graph

    dense_adjacency = np.asarray(adjacency)
    sources, targets = np.nonzero(np.triu(dense_adjacency, k=1))
    for src, dst in zip(sources, targets):
        weight = dense_adjacency[src, dst]
        if weight != 0:
            graph.addEdge(int(src), int(dst), float(weight))

    return graph


def _networkit_leiden(adjacency: Any, resolution: float) -> np.ndarray:
    """Run Leiden clustering using NetworKit"""
    n_nodes = adjacency.shape[0]
    if n_nodes == 0:
        return np.array([], dtype=np.int64)

    graph = _networkit_graph_from_adjacency(adjacency)
    if graph.numberOfEdges() == 0:
        return np.arange(n_nodes, dtype=np.int64)

    previous_threads = nk.getMaxNumberOfThreads()
    nk.setNumberOfThreads(LEIDEN_THREADS)
    try:
        nk.setSeed(LEIDEN_RANDOM_STATE, False)
        leiden = nk.community.ParallelLeiden(
            graph,
            randomize=LEIDEN_RANDOMIZE,
            iterations=LEIDEN_ITERATIONS,
            gamma=resolution,
        )
        leiden.run()
        partition = leiden.getPartition()
    finally:
        nk.setNumberOfThreads(previous_threads)
    labels = np.array(
        [partition.subsetOf(node_idx) for node_idx in range(n_nodes)], dtype=np.int64
    )
    _, normalized_labels = np.unique(labels, return_inverse=True)
    return normalized_labels


# TODO: clean up this implementation
class ClusteringAgreement:
    """Compute clustering agreement between real and predicted perturbation centroids."""

    def __init__(
        self,
        embed_key: str | None = None,
        real_resolution: float = 1.0,
        pred_resolutions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0),
        metric: Literal["ami", "nmi", "ari"] = "ami",
        n_neighbors: int = 15,
    ) -> None:
        self.embed_key = embed_key
        self.real_resolution = real_resolution
        self.pred_resolutions = pred_resolutions
        self.metric = metric
        self.n_neighbors = n_neighbors

    @staticmethod
    def _score(
        labels_real: Sequence[int],
        labels_pred: Sequence[int],
        metric: Literal["ami", "nmi", "ari"],
    ) -> float:
        if metric == "ami":
            return adjusted_mutual_info_score(labels_real, labels_pred)
        if metric == "nmi":
            return normalized_mutual_info_score(labels_real, labels_pred)
        if metric == "ari":
            return (adjusted_rand_score(labels_real, labels_pred) + 1) / 2
        raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def _cluster_leiden(
        adata: ad.AnnData,
        resolution: float,
        n_neighbors: int = 15,
    ) -> np.ndarray:
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(
                adata, n_neighbors=min(n_neighbors, adata.n_obs - 1), use_rep="X"
            )
        return _networkit_leiden(adata.obsp["connectivities"], resolution=resolution)

    @staticmethod
    def _labels_by_category(
        adata: ad.AnnData,
        category_key: str,
        labels: np.ndarray,
    ) -> pd.Series:
        obs = cast(pd.DataFrame, adata.obs)
        return pd.Series(labels, index=pd.Index(obs[category_key], name=category_key))

    @staticmethod
    def _centroid_ann(
        adata: ad.AnnData,
        category_key: str,
        control_pert: str,
        embed_key: str | None = None,
    ) -> ad.AnnData:
        # Isolate the features
        feats = adata.obsm.get(embed_key, adata.X)  # type: ignore

        # Convert to float if not already
        if feats.dtype != np.dtype("float64"):
            feats = feats.astype(np.float64)

        # Densify if required
        if issparse(feats):
            feats = feats.toarray()

        cats = cast(pd.Series, adata.obs[category_key]).values
        uniq, inv = np.unique(cats, return_inverse=True)
        centroids = np.zeros((uniq.size, feats.shape[1]), dtype=feats.dtype)

        for i, cat in enumerate(uniq):
            mask = cats == cat
            if np.any(mask):
                centroids[i] = feats[mask].mean(axis=0)

        adc = ad.AnnData(X=centroids)
        adc.obs[category_key] = uniq
        return adc[adc.obs[category_key] != control_pert]

    def __call__(self, data: PerturbationAnndataPair) -> float:
        cats_sorted = sorted([c for c in data.perts if c != data.control_pert])

        # 2. build centroids
        ad_real_cent = self._centroid_ann(
            adata=data.real,
            category_key=data.pert_col,
            control_pert=data.control_pert,
            embed_key=self.embed_key,
        )
        ad_pred_cent = self._centroid_ann(
            adata=data.pred,
            category_key=data.pert_col,
            control_pert=data.control_pert,
            embed_key=self.embed_key,
        )

        # 3. cluster real once
        real_labels = (
            self._labels_by_category(
                ad_real_cent,
                data.pert_col,
                self._cluster_leiden(
                    ad_real_cent, self.real_resolution, self.n_neighbors
                ),
            )
            .loc[cats_sorted]
            .to_numpy()
        )

        # 4. sweep predicted resolutions
        best_score = 0.0
        for r in self.pred_resolutions:
            pred_labels = (
                self._labels_by_category(
                    ad_pred_cent,
                    data.pert_col,
                    self._cluster_leiden(ad_pred_cent, r, self.n_neighbors),
                )
                .loc[cats_sorted]
                .to_numpy()
            )
            score = self._score(real_labels, pred_labels, self.metric)
            best_score = max(best_score, score)

        return float(best_score)
