import argparse as ap
import importlib.metadata
import logging
import os

from .. import KNOWN_PROFILES
from ._const import (
    DEFAULT_CTRL,
    DEFAULT_GSEA_GENE_SETS,
    DEFAULT_OUTDIR,
    DEFAULT_PERT_COL,
    DEFAULT_PROGENY_LICENSE,
    DEFAULT_PROGENY_METHOD,
    DEFAULT_PROGENY_ORGANISM,
    DEFAULT_PROGENY_THR_PADJ,
    DEFAULT_PROGENY_TOP,
)

logger = logging.getLogger(__name__)


def _parse_progeny_top(value: str) -> int | float:
    if value.lower() in {"all", "inf", "infinity"}:
        return float("inf")
    try:
        top = int(value)
    except ValueError as error:
        raise ap.ArgumentTypeError(
            "--progeny-top must be a positive integer or one of: all, inf"
        ) from error
    if top <= 0:
        raise ap.ArgumentTypeError("--progeny-top must be greater than 0")
    return top


def parse_args_run(parser: ap.ArgumentParser):
    """
    CLI for evaluation
    """
    parser.add_argument(
        "-ap",
        "--adata-pred",
        type=str,
        help="Path to the predicted adata object to evaluate",
        required=True,
    )
    parser.add_argument(
        "-ar",
        "--adata-real",
        type=str,
        help="Path to the real adata object to evaluate against",
        required=True,
    )
    parser.add_argument(
        "-dp",
        "--de-pred",
        type=str,
        help="Path to the predicted DE results "
        f"(computed with pdex from adata-pred if not provided and saved to {DEFAULT_OUTDIR}/pred_de.csv)",
        required=False,
    )
    parser.add_argument(
        "-dr",
        "--de-real",
        type=str,
        help="Path to the real DE results "
        f"(computed with pdex from adata-real if not provided and saved to {DEFAULT_OUTDIR}/real_de.csv)",
        required=False,
    )
    parser.add_argument(
        "--control-pert",
        type=str,
        default=DEFAULT_CTRL,
        help="Name of the control perturbation [default: %(default)s]",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        default=DEFAULT_PERT_COL,
        help="Name of the column designated perturbations [default: %(default)s]",
    )
    parser.add_argument(
        "--celltype-col",
        type=str,
        help="Name of the column designated celltype to split results by (optional)",
    )
    parser.add_argument(
        "--embed-key",
        type=str,
        default=None,
        help="Key for embedded data (.obsm) in the AnnData object used in some metrics (evaluated over .X otherwise)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help="Output directory to write to [default: %(default)s]",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads to use for parallel processing [default: %(default)s]",
    )
    parser.add_argument(
        "--allow-discrete",
        action="store_true",
        help="Allow discrete data to be evaluated (usually expected to be norm-logged inputs)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        help="Profile of metrics to compute [default: %(default)s]",
        choices=KNOWN_PROFILES,
    )
    parser.add_argument(
        "--gsea-gene-sets",
        type=str,
        default=DEFAULT_GSEA_GENE_SETS,
        help=(
            "Gene sets for GSEA metrics. Use 'hallmark' for Decoupler Hallmark, "
            "or provide a .gmt/CSV/TSV path with source,target columns "
            "[default: %(default)s]"
        ),
    )
    parser.add_argument(
        "--gsea-rank-by",
        type=str,
        default="log2_fold_change",
        choices=["log2_fold_change", "signed_pvalue", "signed_fdr"],
        help="DE statistic used to rank genes for GSEA metrics [default: %(default)s]",
    )
    parser.add_argument(
        "--gsea-times",
        type=int,
        default=1000,
        help="Number of random permutations for GSEA NES normalization; must be > 1 [default: %(default)s]",
    )
    parser.add_argument(
        "--gsea-tmin",
        type=int,
        default=5,
        help="Minimum number of genes per pathway after filtering [default: %(default)s]",
    )
    parser.add_argument(
        "--gsea-seed",
        type=int,
        default=42,
        help="Random seed for GSEA permutations [default: %(default)s]",
    )
    parser.add_argument(
        "--progeny-method",
        type=str,
        default=DEFAULT_PROGENY_METHOD,
        choices=["ulm", "mlm", "waggr", "zscore"],
        help="Decoupler method used to infer PROGENy pathway activities [default: %(default)s]",
    )
    parser.add_argument(
        "--progeny-rank-by",
        type=str,
        default="log2_fold_change",
        choices=["log2_fold_change", "signed_pvalue", "signed_fdr"],
        help="DE statistic used to infer PROGENy pathway activities [default: %(default)s]",
    )
    parser.add_argument(
        "--progeny-organism",
        type=str,
        default=DEFAULT_PROGENY_ORGANISM,
        help="Organism used when loading Decoupler PROGENy [default: %(default)s]",
    )
    parser.add_argument(
        "--progeny-top",
        type=_parse_progeny_top,
        default=DEFAULT_PROGENY_TOP,
        help="Top PROGENy genes per pathway, or 'all'/'inf' [default: %(default)s]",
    )
    parser.add_argument(
        "--progeny-thr-padj",
        type=float,
        default=DEFAULT_PROGENY_THR_PADJ,
        help="Adjusted p-value threshold for PROGENy interactions [default: %(default)s]",
    )
    parser.add_argument(
        "--progeny-license",
        type=str,
        default=DEFAULT_PROGENY_LICENSE,
        choices=["academic", "commercial", "nonprofit"],
        help="OmniPath license used when loading PROGENy [default: %(default)s]",
    )
    parser.add_argument(
        "--progeny-tmin",
        type=int,
        default=5,
        help="Minimum number of PROGENy target genes per pathway after filtering [default: %(default)s]",
    )
    parser.add_argument(
        "--skip-metrics",
        type=str,
        help="Metrics to skip (comma-separated for multiple) (see docs for more details)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(
            version=importlib.metadata.version("cell_eval")
        ),
    )


def build_outdir(outdir: str):
    if os.path.exists(outdir):
        logger.warning(
            f"Output directory {outdir} already exists, potential overwrite occurring"
        )
    os.makedirs(outdir, exist_ok=True)


def run_evaluation(args: ap.Namespace):
    import anndata as ad

    from cell_eval import MetricsEvaluator
    from cell_eval.utils import split_anndata_on_celltype

    skip_metrics = args.skip_metrics.split(",") if args.skip_metrics else None

    # Set metric config for optional metric inputs.
    metric_kwargs = {}
    if args.embed_key is not None:
        metric_kwargs.update(
            {
                "discrimination_score_l2": {"embed_key": args.embed_key},
                "discrimination_score_cosine": {"embed_key": args.embed_key},
                "pearson_edistance": {"n_jobs": args.num_threads},
            }
        )

    gsea_metric_enabled = args.profile == "vcc" and (
        skip_metrics is None or "gsea_nes_spearman" not in skip_metrics
    )
    progeny_metric_enabled = args.profile == "vcc" and (
        skip_metrics is None or "progeny_activity_spearman" not in skip_metrics
    )

    if gsea_metric_enabled:
        gsea_gene_set_path = (
            None
            if args.gsea_gene_sets == DEFAULT_GSEA_GENE_SETS
            else args.gsea_gene_sets
        )
        metric_kwargs["gsea_nes_spearman"] = {
            "gene_set_path": gsea_gene_set_path,
            "rank_by": args.gsea_rank_by,
            "times": args.gsea_times,
            "tmin": args.gsea_tmin,
            "seed": args.gsea_seed,
        }

    if progeny_metric_enabled:
        metric_kwargs["progeny_activity_spearman"] = {
            "method": args.progeny_method,
            "rank_by": args.progeny_rank_by,
            "organism": args.progeny_organism,
            "top": args.progeny_top,
            "thr_padj": args.progeny_thr_padj,
            "license": args.progeny_license,
            "tmin": args.progeny_tmin,
        }

    if args.celltype_col is not None:
        real = ad.read_h5ad(args.adata_real)
        pred = ad.read_h5ad(args.adata_pred)

        real_split = split_anndata_on_celltype(real, args.celltype_col)
        pred_split = split_anndata_on_celltype(pred, args.celltype_col)

        assert len(real_split) == len(pred_split), (
            f"Number of celltypes in real and pred anndata must match: {len(real_split)} != {len(pred_split)}"
        )

        for ct in real_split.keys():
            real_ct = real_split[ct]
            pred_ct = pred_split[ct]

            evaluator = MetricsEvaluator(
                adata_pred=pred_ct,
                adata_real=real_ct,
                de_pred=args.de_pred,
                de_real=args.de_real,
                control_pert=args.control_pert,
                pert_col=args.pert_col,
                num_threads=args.num_threads,
                outdir=args.outdir,
                allow_discrete=args.allow_discrete,
                prefix=ct,
                skip_de=args.profile == "pds",
            )
            evaluator.compute(
                profile=args.profile,
                metric_configs=metric_kwargs,
                skip_metrics=skip_metrics,
                basename="results.csv",
            )

    else:
        evaluator = MetricsEvaluator(
            adata_pred=args.adata_pred,
            adata_real=args.adata_real,
            de_pred=args.de_pred,
            de_real=args.de_real,
            control_pert=args.control_pert,
            pert_col=args.pert_col,
            num_threads=args.num_threads,
            outdir=args.outdir,
            allow_discrete=args.allow_discrete,
            skip_de=args.profile == "pds",
        )
        evaluator.compute(
            profile=args.profile,
            metric_configs=metric_kwargs,
            skip_metrics=skip_metrics,
            basename="results.csv",
        )
