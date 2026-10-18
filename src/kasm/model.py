"""Model — a dataset bound to a kernel configuration.

``Model`` is where a declarative ``KernelOptions`` meets real columns. It
validates that every column the configuration names exists and is usable, and
it turns the configuration into the per-gene arrays the numerical reducers
consume: the variant scaling ``d``, the burden weight matrix, and the diagonal
kernel weights. The reducers themselves never see a ``Model`` — they take
an LDM and NumPy arrays.

The model is stateful by design: computed sufficient statistics live on it
(stage 3), which is what lets one configuration be fitted many ways.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl

from kasm import fitting
from kasm._logging import logger
from kasm.dataset import Dataset
from kasm.kernels import KernelOptions
from kasm.suffstats import (
    JKSuffStats,
    SuffStats,
    compute_jk_suffstats,
    compute_suffstats,
)


class Model:
    """A dataset bound to a kernel configuration.

    Parameters
    ----------
    dataset : Dataset
    kernel_options : KernelOptions
    """

    def __init__(self, dataset: Dataset, kernel_options: KernelOptions):
        self.dataset = dataset
        self.options = kernel_options
        self.layout = kernel_options.layout

        _validate_columns(dataset, kernel_options)
        self._maf_masks = _maf_masks(dataset, kernel_options)
        self.n_unbinned = _count_unbinned(dataset, kernel_options, self._maf_masks)
        self.suffstats: SuffStats | None = None
        self.jk_suffstats: JKSuffStats | None = None
        self.burden_1d_wgts: dict[str, float] | None = None
        logger.info(
            "event=model_created genes=%d variants=%d kernels=%d burden_kernels=%d",
            dataset.n_genes,
            dataset.n_variants,
            self.layout.n_kernels,
            self.layout.n_burden,
        )

    # -- sufficient statistics -----------------------------------------------

    def compute_suffstats(self, extra_enrich_cols=None) -> Model:
        """Retain the base, requested burden extensions, and configured covar extensions.

        Parameters
        ----------
        extra_enrich_cols : list[str] or None
            Finite enrichment columns to reduce in the same LD pass. Select one
            later with ``fit_random_effects(extra_enrich_col=...)``.
        """
        requested = tuple(dict.fromkeys(extra_enrich_cols or ()))
        self.suffstats = compute_suffstats(self, extra_enrich_cols=requested or None)
        self.jk_suffstats = None
        return self

    def with_burden_extension(self, column: str) -> Model:
        """Return an augmented model from one retained burden-column extension.

        ``compute_suffstats(extra_enrich_cols=...)`` reduces every requested
        extension alongside the base model in one LD pass. This method selects
        one of those ordinary augmented models without reloading LD. It does
        not assign enrichment semantics to the column.
        """
        base = self._require_suffstats()
        if column in (self.options.burden_cols or ()):
            raise ValueError(f"burden column {column!r} is already in the model")
        if column not in base.burden_extensions:
            raise ValueError(
                f"burden extension {column!r} is unavailable; call "
                f"compute_suffstats(extra_enrich_cols=[{column!r}]) first"
            )
        extended = self._with_burden_column(column)
        extended.suffstats = base.with_burden_extension(column)
        return extended

    def _with_burden_column(self, column: str) -> Model:
        """Construct an unfitted copy with one appended burden column."""
        if self.options.kernel_groups:
            groups = {
                name: dict(value) for name, value in self.options.kernel_groups.items()
            }
            if len(groups) != 1:
                raise ValueError("enrichment requires a single kernel group")
            group = next(iter(groups.values()))
            group["burden_cols"] = [*group["burden_cols"], column]
            return Model(
                self.dataset,
                replace(self.options, burden_cols=None, kernel_groups=groups),
            )
        return Model(
            self.dataset,
            replace(
                self.options,
                burden_cols=[*(self.options.burden_cols or ()), column],
            ),
        )

    def compute_jk_suffstats(
        self,
        *,
        exclude_genes,
        n_jk_blocks=100,
        gene_blocks=None,
    ) -> JKSuffStats:
        """Retain one aggregate as ``jk_suffstats`` without replacing ``suffstats``.

        Fits reuse this geometry. Recomputing per-gene statistics invalidates
        the aggregate; callers replacing them directly must clear it explicitly.
        Explicit fit geometry must match the aggregate.
        """
        base = self._require_suffstats()
        if base.genes != tuple(self.genes):
            raise ValueError("sufficient statistics genes do not match the model")
        self.jk_suffstats = compute_jk_suffstats(
            base,
            exclude_genes=exclude_genes,
            n_jk_blocks=n_jk_blocks,
            gene_annot=self.gene_annot_matrix(),
            sample_size=self.dataset.sample_size,
            gene_annot_names=tuple(self.options.gene_annot_cols or ("constant",)),
            kernel_options=self.options.to_dict(),
            layout=self.layout,
            gene_blocks=gene_blocks,
        )
        return self.jk_suffstats

    def _get_jk_suffstats(
        self, *, exclude_genes=None, n_jk_blocks=None, gene_blocks=None
    ):
        """Use the single prepared geometry, or prepare it from explicit support."""
        if self.jk_suffstats is None:
            if exclude_genes is None:
                raise ValueError(
                    "exclude_genes is required before preparing JK statistics"
                )
            return self.compute_jk_suffstats(
                exclude_genes=exclude_genes,
                n_jk_blocks=min(100, len(self.genes))
                if n_jk_blocks is None
                else n_jk_blocks,
                gene_blocks=gene_blocks,
            )
        self._require_suffstats()
        stats = self.jk_suffstats
        stats.validate_jk_shape(
            exclude_genes=exclude_genes,
            n_jk_blocks=n_jk_blocks,
            gene_blocks=gene_blocks,
        )
        if stats.genes != self.suffstats.genes or stats.layout != self.layout:
            raise ValueError("stored JK geometry does not match the Model")
        return stats

    def _require_suffstats(self):
        if self.suffstats is None:
            raise ValueError(
                "no sufficient statistics; call compute_suffstats() or "
                "assign loaded statistics to model.suffstats first"
            )
        if not isinstance(self.suffstats, SuffStats):
            raise TypeError(
                "model.suffstats must be per-gene SuffStats; pass JKSuffStats directly to fitting"
            )
        return self.suffstats

    # -- per-gene tests ------------------------------------------------------

    def compute_gene_pvalue(self, method: str = "burden_1d") -> pl.DataFrame:
        """Per-gene burden test with an explicit gene column.

        ``method="burden_1d"`` learns one weight vector across the whole gene
        set, projects each gene's burden components onto it, and tests that
        scalar. The weights are retained as ``burden_1d_wgts`` so that fixed
        burden estimation reuses the burden projection rather than relearning
        it from the significant genes alone. ``method="burden_full"`` instead
        tests every retained burden component jointly.
        """
        if method not in ("burden_1d", "burden_full"):
            raise ValueError(
                f"unknown method {method!r}; expected 'burden_1d' or 'burden_full'"
            )
        suffstats = self._require_suffstats()
        burden_beta = suffstats.burden_beta
        burden_cov = suffstats.burden_cov()
        if method == "burden_full":
            self.burden_1d_wgts = None
            statistics = fitting.burden_full_statistics(
                burden_beta, burden_cov, self.dataset.sample_size
            )
            return pl.DataFrame({"gene": suffstats.genes, **statistics})
        burden_labels = self.layout.burden_labels
        burden_1d_wgts = fitting.optimal_1d_weights(burden_beta, burden_cov)
        self.burden_1d_wgts = dict(zip(burden_labels, burden_1d_wgts))
        statistics = fitting.burden_1d_statistics(
            burden_beta, burden_cov, burden_1d_wgts, self.dataset.sample_size
        )
        return pl.DataFrame({"gene": suffstats.genes, **statistics})

    # -- shape ---------------------------------------------------------------

    @property
    def n_kernels(self) -> int:
        return self.layout.n_kernels

    @property
    def genes(self) -> list[str]:
        return self.dataset.genes

    # -- per-gene arrays for the reducers ------------------------------------

    def variant_scale(self, gene: str | None = None) -> np.ndarray:
        """Diagonal scaling ``d`` for a gene, or for every variant.

        ``d = sqrt(2 p (1 - p))`` under ``variant_variance_from_maf``, else
        ``d = 1 / (se sqrt(N))``. The two agree only when the standard errors
        were generated under the MAF convention. They define different
        observation geometries, although coefficients for the covariance of
        unscaled per-allele causal effects can be transported between them.
        """
        if self.options.variant_variance_from_maf:
            maf = self.dataset.variant_values(self.options.maf_col, gene)
            return np.sqrt(2.0 * maf * (1.0 - maf))
        se = self.dataset.variant_values(self.dataset.se_col, gene)
        return 1.0 / (se * np.sqrt(self.dataset.sample_size))

    def burden_wgts(self, gene: str | None = None) -> np.ndarray:
        """Burden weights ``U``, shape ``(m, n_burden)``.

        Column ``k`` is ``burden_labels[k]``: the annotation column, zeroed
        outside its MAF bin.
        """
        return self._design(self.layout.burden_indices, gene)

    def burden_partition_codes(self, gene: str | None = None) -> np.ndarray | None:
        """Integer burden-partition membership, factored independently by gene."""
        column = self.options.within_gene_group_col
        if column is None:
            return None
        if gene is not None:
            values = self.dataset.variant_df(gene)[column]
            levels = values.unique(maintain_order=True)
            return np.asarray(
                values.replace_strict(
                    levels, pl.Series(range(len(levels))), return_dtype=pl.Int64
                ),
                dtype=np.intp,
            )
        codes = np.empty(self.dataset.n_variants, dtype=np.intp)
        for name in self.dataset.genes:
            codes[self.dataset.gene_slice(name)] = self.burden_partition_codes(name)
        return codes

    def diag_kernel_weights(self, gene: str | None = None) -> np.ndarray:
        """Diagonal kernel weights, shape ``(m, n_diagonal)``.

        Column ``k`` is ``layout.diagonal_indices[k]``. The global independent kernel
        is all ones; annotated independent and overdispersion kernels use their
        named variant columns.
        """
        return self._design(self.layout.diagonal_indices, gene)

    def overdispersion_weights(self, gene: str | None = None) -> np.ndarray:
        """Overdispersion kernel weights, shape ``(m, n_overdispersion)``."""
        return self._design(self.layout.overdispersion_indices, gene)

    def gene_annot_matrix(self) -> np.ndarray:
        """Gene covariates, shape ``(n_genes, n_annot)``, in dataset gene order.

        A single all-ones column when the configuration names none, so that
        downstream code always has a gene-level design matrix.

        Return one writable allocation without changing Dataset's frozen buffers.
        """
        if not self.options.gene_annot_cols:
            return np.ones((self.dataset.n_genes, 1))
        return np.column_stack(
            [
                self.dataset.gene_values(column)
                for column in self.options.gene_annot_cols
            ]
        )

    def _design(self, kernel_indices, gene: str | None) -> np.ndarray:
        """Stack each kernel's annotation column, masked to its MAF bin.

        Callers that walk every gene should build the whole design once and
        slice it, rather than asking for one gene at a time: the columns are
        re-extracted on each call.
        """
        if gene is None:
            rows, m = slice(None), self.dataset.n_variants
        else:
            rows = self.dataset.gene_slice(gene)
            m = rows.stop - rows.start

        columns = []
        for index in kernel_indices:
            kernel_id = self.layout.kernel_ids[index]
            values = np.asarray(
                self.dataset.variant_values(kernel_id.col, gene), dtype=float
            )
            if kernel_id.maf_label is not None:
                values = values * self._maf_masks[kernel_id.maf_label][rows]
            columns.append(values)
        return np.column_stack(columns) if columns else np.zeros((m, 0))

    def __repr__(self) -> str:
        return (
            f"Model(n_genes={self.dataset.n_genes}, "
            f"n_kernels={self.n_kernels}, "
            f"n_burden={self.layout.n_burden})"
        )


def _validate_columns(dataset: Dataset, options: KernelOptions) -> None:
    """Every column the configuration names must exist and be usable."""
    needed = [
        *(options.burden_cols or ()),
        *options.independent_cols,
        *(
            options.layout.kernel_ids[i].col
            for i in options.layout.overdispersion_indices
        ),
    ]
    if options.maf_col is not None:
        needed.append(options.maf_col)
    for col in dict.fromkeys(needed):
        values = dataset.variant_values(col)  # raises KeyError if absent
        if col in options.independent_cols and (values < 0).any():
            raise ValueError(f"independent column {col!r} must be nonnegative")

    for kernel_group, group_options in (options.kernel_groups or {}).items():
        independent_col = group_options["independent_col"]
        if independent_col is None:
            continue
        support = dataset.variant_values(independent_col)
        if not np.isin(support, (0.0, 1.0)).all():
            raise ValueError(
                f"kernel group {kernel_group!r} independent column {independent_col!r} "
                "must be binary"
            )
        outside = support == 0
        for col in group_options["burden_cols"]:
            if np.any(dataset.variant_values(col)[outside] != 0):
                raise ValueError(
                    f"kernel group {kernel_group!r} burden column {col!r} must be zero "
                    f"outside {independent_col!r}"
                )

    if options.maf_col is not None:
        maf = dataset.variant_values(options.maf_col)
        if (maf < 0).any() or (maf > 1).any():
            raise ValueError(f"{options.maf_col!r} must lie in [0, 1]")
        if options.variant_variance_from_maf and ((maf <= 0).any() or (maf >= 1).any()):
            raise ValueError(
                f"{options.maf_col!r} must lie strictly between 0 and 1 "
                "when variant_variance_from_maf=True"
            )

    if options.gene_annot_cols:
        annot = fitting.validate_gene_annotations(
            dataset.gene_values(list(options.gene_annot_cols))
        )
        rank = np.linalg.matrix_rank(annot)
        if rank < annot.shape[1]:
            raise ValueError(
                f"gene_annot_cols are not full rank ({rank} < {annot.shape[1]}); "
                "gene-level parameters are not identifiable"
            )
    if options.within_gene_group_col is not None:
        column = options.within_gene_group_col
        frame = dataset.variant_df()
        if column not in frame.columns:
            raise KeyError(f"variant_df has no column {column!r}")
        if frame[column].null_count():
            raise ValueError(
                f"within-gene group column {column!r} must not contain missing values"
            )


def _maf_masks(dataset: Dataset, options: KernelOptions) -> dict[str, np.ndarray]:
    """Float 0/1 membership mask per MAF bin, over all variants."""
    if not options.maf_bins:
        return {}
    maf = dataset.variant_values(options.maf_col)
    return {
        name: ((low <= maf) & (maf < high)).astype(np.float64)
        for name, (low, high) in options.maf_bins.items()
    }


def _count_unbinned(dataset, options, masks) -> int:
    """Variants in no MAF bin. They contribute to no stratified kernel."""
    if not masks:
        return 0
    covered = np.zeros(dataset.n_variants, dtype=bool)
    for mask in masks.values():
        covered |= mask > 0
    unbinned = int((~covered).sum())
    if unbinned == dataset.n_variants:
        raise ValueError(
            f"no variant falls in any MAF bin; check {options.maf_col!r} against "
            f"maf_bins {options.maf_bins}"
        )
    return unbinned
