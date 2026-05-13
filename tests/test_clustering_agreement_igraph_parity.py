import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from cell_eval import MetricsEvaluator
from cell_eval._types import PerturbationAnndataPair
from cell_eval.metrics import ClusteringAgreement

pytest.importorskip("igraph")


class IgraphReferenceClusteringAgreement(ClusteringAgreement):
    @staticmethod
    def _cluster_leiden(
        adata: ad.AnnData,
        resolution: float,
        key_added: str,
        n_neighbors: int = 15,
    ) -> None:
        if key_added in adata.obs:
            return
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(
                adata, n_neighbors=min(n_neighbors, adata.n_obs - 1), use_rep="X"
            )
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key_added,
            flavor="igraph",
            n_iterations=2,
            random_state=0,
        )


PERT_COL = "perturbation"
CONTROL_PERT = "control"
PERTURBATIONS = [
    CONTROL_PERT,
    "ACLY",
    "ANXA6",
    "ARPC2",
    "COX6C",
    "CSK",
    "CTSV",
    "DOT1L",
    "DPH2",
    "EHMT1",
    "FUBP1",
]

# Each row is a pseudobulk over the first 3 cells for one perturbation, using
# the first 8 features from the saved STATE real/pred AnnData pair.
REAL_X = np.array(
    [
        [
            0.0,
            0.0,
            0.60480696,
            1.0147281,
            0.45686433,
            1.5789233,
            0.15786041,
            0.23511563,
        ],
        [0.0, 0.0, 0.3299731, 0.9338847, 0.0, 2.2003114, 0.10383952, 0.31972015],
        [
            0.0,
            0.0,
            0.33602008,
            1.6207587,
            0.20242812,
            2.0947647,
            0.20242812,
            0.42307013,
        ],
        [0.0, 0.0, 0.6058275, 1.2358193, 0.44331926, 1.6750681, 0.2847381, 0.26776075],
        [
            0.0,
            0.0,
            0.59900653,
            1.2323949,
            0.32047153,
            1.8378658,
            0.20493835,
            0.53996617,
        ],
        [0.0, 0.0, 0.6796385, 1.2621685, 0.15937002, 1.9017699, 0.0, 0.64992213],
        [0.0, 0.0, 0.25770748, 1.466327, 0.25220138, 1.5658166, 0.18205424, 0.44245934],
        [
            0.0,
            0.0,
            0.24362756,
            1.1761581,
            0.36843127,
            1.8905183,
            0.20891976,
            0.14359324,
        ],
        [0.0, 0.0, 0.17034812, 1.757152, 0.37964264, 1.7587336, 0.19226222, 0.09632262],
        [0.0, 0.0, 0.3833941, 1.3776573, 0.0, 1.5562577, 0.0, 0.0],
        [0.0, 0.0, 1.1842427, 1.2781069, 0.44109604, 1.8756988, 0.0, 0.25006843],
    ],
    dtype=np.float32,
)

PRED_X = np.array(
    [
        [
            0.12574497,
            1.0215758,
            0.00384475,
            0.14187452,
            0.0,
            0.34104082,
            4.2961254,
            0.0,
        ],
        [0.07325882, 0.68531537, 0.0, 0.12940465, 0.0, 0.2618303, 4.479837, 0.0],
        [
            0.07348129,
            0.96166223,
            0.1246824,
            0.02984748,
            0.0,
            0.31466314,
            4.6967735,
            0.0,
        ],
        [0.0, 0.99174285, 0.02410149, 0.00793363, 0.0, 0.00915282, 4.3635025, 0.0],
        [0.0980356, 1.1640447, 0.0, 0.10976028, 0.02083437, 0.14262633, 4.609952, 0.0],
        [0.02744179, 0.985369, 0.01125427, 0.02376619, 0.01002621, 0.0, 4.308384, 0.0],
        [0.06706298, 1.4986506, 0.09679828, 0.02010665, 0.0, 0.238944, 4.682041, 0.0],
        [0.31009912, 0.9791632, 0.0, 0.0, 0.02095211, 0.05203755, 4.5111127, 0.0],
        [
            0.01052931,
            1.2221023,
            0.03932361,
            0.17964269,
            0.01602912,
            0.11803366,
            4.4680724,
            0.0,
        ],
        [
            0.01341854,
            1.0703925,
            0.13870947,
            0.03706174,
            0.0,
            0.08551833,
            4.8320613,
            0.0,
        ],
        [0.0, 0.8109128, 0.27211696, 0.0, 0.05988158, 0.0540864, 4.7460966, 0.0],
    ],
    dtype=np.float32,
)


def _inline_state_pair() -> PerturbationAnndataPair:
    obs = pd.DataFrame(
        {PERT_COL: PERTURBATIONS},
        index=pd.Index([f"cell_{idx}" for idx in range(len(PERTURBATIONS))]),
    )
    var = pd.DataFrame(
        index=pd.Index([f"feature_{idx}" for idx in range(REAL_X.shape[1])])
    )
    evaluator = MetricsEvaluator(
        adata_pred=ad.AnnData(X=PRED_X, obs=obs.copy(), var=var.copy()),
        adata_real=ad.AnnData(X=REAL_X, obs=obs.copy(), var=var.copy()),
        control_pert=CONTROL_PERT,
        pert_col=PERT_COL,
        skip_de=True,
    )
    return evaluator.anndata_pair


def _score(
    metric_cls: type[ClusteringAgreement],
    data: PerturbationAnndataPair,
) -> float:
    return metric_cls(
        real_resolution=1.0,
        pred_resolutions=(0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0),
        metric="ami",
        n_neighbors=15,
    )(data)


def test_networkit_leiden_matches_igraph_reference_on_inline_state_fixture() -> None:
    data = _inline_state_pair()
    igraph_score = _score(IgraphReferenceClusteringAgreement, data)
    networkit_score = _score(ClusteringAgreement, data)

    assert igraph_score == pytest.approx(0.15800415800415743)
    assert networkit_score == pytest.approx(igraph_score)
