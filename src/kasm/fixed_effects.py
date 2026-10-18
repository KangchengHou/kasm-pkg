"""Fixed-effect estimators: hsq of the significant gene set.

"Fixed" means these genes were selected for being significant, so they are not
exchangeable and a jackknife over them would understate the uncertainty. Each
gene's quadratic form is debiased by the sampling noise its own rank
contributes, and the standard error is analytic.

Every estimator here requires ``include_genes``: the significant-gene set is an input,
never inferred.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.linalg import pinvh

from kasm import fitting

# Relative tolerance for pseudo-inverse rank determination. The rank enters the
# debiasing directly, so it is a modelling choice rather than a numerical detail.
PINV_RTOL = 1e-6


@dataclass(frozen=True)
class FixedFit:
    """Fixed-effect result owning per-gene moments and their analysis context."""

    hsq: float
    hsq_se: float
    per_gene: pl.DataFrame
    method: str
    n_genes: int
    sample_size: float
    _model: object
    _maf_per_gene: pl.DataFrame | None = None
    burden_per_gene: pl.DataFrame | None = None
    _extra_enrich_col: str | None = None
    _burden_1d_wgts: object = None
    _burden_indices: object = None
    _maf_error: str | None = None

    @property
    def point(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "hsq": self.hsq,
                    "hsq_se": self.hsq_se,
                    "n_genes": self.n_genes,
                    "method": self.method,
                }
            ]
        )

    @property
    def total_rank(self) -> float:
        return float(self.per_gene["rank"].sum())

    @property
    def maf_components(self) -> pl.DataFrame:
        """Exact additive allocation of fitted hsq across MAF bins."""
        if self._extra_enrich_col is not None:
            raise TypeError("MAF components are unavailable for enrichment fits")
        if self._maf_error is not None:
            raise ValueError(self._maf_error)
        components = self._maf_per_gene
        names = tuple(self._model.options.maf_bins)
        result = pl.DataFrame(
            {
                "maf_bin": names,
                "hsq": [
                    components.filter(pl.col("maf_bin") == name)["hsq"].sum()
                    for name in names
                ],
            }
        )
        if not np.isclose(
            result["hsq"].sum(),
            self.hsq,
            rtol=1e-9,
            atol=1e-11,
        ):
            raise RuntimeError("MAF components do not recover fitted hsq")
        return result

    def decompose(self, columns) -> pl.DataFrame:
        """Allocate fitted hsq across categorical variant columns."""
        if self._extra_enrich_col is not None:
            raise TypeError("decompose is unavailable for enrichment fits")
        if isinstance(columns, str):
            columns = (columns,)
        return _decompose_fixed_effects(
            self._model,
            self,
            columns,
            burden_1d_wgts=self._burden_1d_wgts,
            burden_indices=self._burden_indices,
        )


def _apply_low_rank(design, inverse, values):
    """Apply a symmetric low-rank operator without materializing it."""
    return design @ (inverse @ (design.T @ values))


def _fit_fixed_effects(
    model,
    *,
    include_genes,
    method: str = "burden",
    burden_1d_wgts=None,
    burden_indices=None,
    extra_enrich_col: str | None = None,
) -> FixedFit:
    """Hsq carried by ``include_genes``, treated as fixed effects.

    Parameters
    ----------
    model : Model
        ``method="burden_1d"`` requires retained or explicitly supplied
        burden projection weights. Every method computes
        the selected genes' fixed moments directly from their LD.
    include_genes : iterable of str
        The gene set. Required.
    method : {"burden_1d", "burden", "indep"}
        ``"burden_1d"`` projects each gene onto the retained one-dimensional
        burden projection weights. ``"burden"`` jointly fits the full burden design
        and is the default. ``"indep"`` is the saturated per-gene fit, using
        the pseudo-inverse of the gene's own scaled LD.

        This is orthogonal to ``KernelOptions(independent_col=...)``, which adds an
        identity *kernel* to a random-effect model.
    burden_1d_wgts : Mapping or None
        Burden projection weights indexed by burden label. Required for
        ``method="burden_1d"`` unless retained on ``model`` by
        ``compute_gene_pvalue``.
    burden_indices : iterable of int or None
        Expanded burden-design columns to include with ``method="burden"``.
        When supplied, the quadratic forms are computed directly from the
        selected design columns and each gene's LD. This is the lightweight
        seam used for nested fixed-effect estimands; it does not build random
        kernel sufficient statistics.
    extra_enrich_col : str or None
        Finite annotation used for the canonical ``[X, X * annotation]``
        enrichment design.
    """
    if include_genes is None:
        raise ValueError("include_genes is required")
    if method not in ("burden_1d", "burden", "indep"):
        raise ValueError(
            f"unknown method {method!r}; expected 'burden_1d', 'burden', or 'indep'"
        )

    if burden_1d_wgts is not None and method != "burden_1d":
        raise ValueError("burden_1d_wgts is only supported by method='burden_1d'")
    if burden_indices is not None and method != "burden":
        raise ValueError("burden_indices is only supported by method='burden'")
    if extra_enrich_col is not None:
        if method != "burden":
            raise ValueError("extra_enrich_col requires method='burden'")
        if burden_1d_wgts is not None or burden_indices is not None:
            raise ValueError(
                "extra_enrich_col cannot be combined with burden_1d_wgts or "
                "burden_indices"
            )
        from kasm.enrichment import _validate_burden_annotation

        _validate_burden_annotation(model, extra_enrich_col)
    if burden_indices is not None:
        burden_indices = np.asarray(tuple(burden_indices), dtype=np.intp)
        if burden_indices.ndim != 1 or burden_indices.size == 0:
            raise ValueError("burden_indices must select at least one column")
        if (
            np.any(burden_indices < 0)
            or np.any(burden_indices >= model.layout.n_burden)
            or len(np.unique(burden_indices)) != len(burden_indices)
        ):
            raise ValueError("burden_indices contains invalid or repeated columns")

    genes = _select_genes(model.dataset.genes, include_genes)
    burden_1d_wgts = (
        _validated_burden_1d_wgts(model, burden_1d_wgts)
        if method == "burden_1d"
        else None
    )
    quadforms, ranks, maf_components, maf_error, burden_moments = _fixed_gene_moments(
        model,
        genes,
        method=method,
        burden_1d_wgts=burden_1d_wgts,
        burden_indices=burden_indices,
        extra_enrich_col=extra_enrich_col,
    )

    sample_size = model.dataset.sample_size
    hsq, error = fitting.fixed_hsq_from_quadforms(quadforms, ranks, sample_size)
    return FixedFit(
        hsq=hsq,
        hsq_se=error,
        per_gene=_per_gene_df(genes, quadforms, ranks, sample_size),
        method=method,
        n_genes=len(genes),
        sample_size=sample_size,
        _model=model,
        _extra_enrich_col=extra_enrich_col,
        _burden_1d_wgts=(
            None
            if burden_1d_wgts is None
            else dict(zip(model.layout.burden_labels, burden_1d_wgts))
        ),
        _burden_indices=burden_indices,
        _maf_per_gene=maf_components,
        _maf_error=maf_error,
        burden_per_gene=(
            None
            if burden_moments is None
            else _per_gene_df(
                genes, burden_moments[0], burden_moments[1], sample_size
            ).with_columns(
                pl.Series("variance", burden_moments[2][:, 0]),
                pl.Series("total_variance", burden_moments[2][:, 1]),
                pl.Series("total_covariance", burden_moments[2][:, 2]),
            )
        ),
    )


def _select_genes(available_genes, include_genes):
    """Validate and retain requested genes in canonical model order."""
    requested = set(include_genes)
    genes = [gene for gene in available_genes if gene in requested]
    missing = requested - set(available_genes)
    if missing:
        raise KeyError(
            f"{len(missing)} genes are not in this model: {sorted(missing)[:5]}"
        )
    return genes


def _per_gene_df(genes, quadforms, ranks, sample_size):
    usable = ranks < sample_size
    hsq = np.full(len(ranks), np.nan)
    hsq[usable] = (sample_size * quadforms[usable] - ranks[usable]) / (
        sample_size - ranks[usable]
    )
    return pl.DataFrame(
        {
            "gene": genes,
            "quadform": quadforms,
            "rank": ranks,
            "hsq": hsq,
        },
    )


def _validated_burden_1d_wgts(model, burden_1d_wgts):
    if burden_1d_wgts is None:
        burden_1d_wgts = model.burden_1d_wgts
    if burden_1d_wgts is None:
        raise ValueError(
            "method='burden_1d' needs the burden projection weights; call "
            "compute_gene_pvalue(method='burden_1d') first or pass "
            "burden_1d_wgts"
        )
    if not isinstance(burden_1d_wgts, Mapping):
        raise TypeError("burden_1d_wgts must be a Mapping")
    labels = tuple(model.layout.burden_labels)
    missing = set(labels) - set(burden_1d_wgts)
    extra = set(burden_1d_wgts) - set(labels)
    if missing or extra:
        raise ValueError(
            "burden_1d_wgts labels mismatch: "
            f"missing={sorted(missing)}, extras={sorted(extra)}"
        )
    burden_1d_wgts = np.asarray(
        [burden_1d_wgts[label] for label in labels], dtype=float
    )
    if not np.isfinite(burden_1d_wgts).all():
        raise ValueError("burden_1d_wgts must be finite")
    return burden_1d_wgts


def _maf_memberships(model, genes):
    """Variant-to-bin indicators when configured bins partition every gene."""
    if model.options.maf_col is None or not model.options.maf_bins:
        raise ValueError("fixed MAF decomposition requires maf_col and maf_bins")
    memberships = []
    for gene in genes:
        maf = model.dataset.variant_values(model.options.maf_col, gene)
        if not np.isfinite(maf).all():
            raise ValueError(
                f"{model.options.maf_col!r} must not contain missing values"
            )
        membership = np.column_stack(
            [
                (low <= maf) & (maf < high)
                for low, high in model.options.maf_bins.values()
            ]
        )
        if not np.all(membership.sum(axis=1) == 1):
            raise ValueError(
                f"maf_bins must be an exhaustive, disjoint partition for gene {gene!r}"
            )
        memberships.append(membership)
    return tuple(model.options.maf_bins), memberships


def _fixed_gene_moments(
    model,
    genes,
    *,
    method,
    burden_1d_wgts=None,
    burden_indices=None,
    extra_enrich_col=None,
):
    """Compute fixed, MAF, and enrichment moments in one gene/LD pass."""
    dataset = model.dataset
    beta_tilde_all = dataset.beta / (dataset.se**2 * dataset.sample_size)
    quadforms = np.zeros(len(genes))
    ranks = np.zeros(len(genes))
    annotated_quadforms = annotated_ranks = None
    enrichment_covariances = None
    maf_error = None
    if extra_enrich_col is not None:
        from kasm.enrichment import _nested_burden_designs

        annotated_quadforms = np.zeros(len(genes))
        annotated_ranks = np.zeros(len(genes))
        enrichment_covariances = np.zeros((len(genes), 3))
        maf_names, memberships, maf_values = (), None, None
    else:
        try:
            maf_names, memberships = _maf_memberships(model, genes)
            maf_values = np.zeros((len(genes), len(maf_names)))
        except ValueError as error:
            maf_names, memberships, maf_values = (), None, None
            maf_error = str(error)

    for position, gene in enumerate(genes):
        rows = dataset.gene_slice(gene)
        annotated_design = None
        if extra_enrich_col is not None:
            design, annotated_design = _nested_burden_designs(
                model, gene, extra_enrich_col
            )
            burden_wgts = None
        else:
            design, burden_wgts = _fixed_design(
                model,
                gene,
                method=method,
                burden_1d_wgts=burden_1d_wgts,
                burden_indices=burden_indices,
            )

        scaled_ldm = dataset.load_ldm(gene).scale(model.variant_scale(gene))
        propagated = scaled_ldm @ design
        covariance = design.T @ propagated
        inverse, rank = _fixed_inverse(
            model,
            gene,
            method,
            burden_1d_wgts,
            burden_wgts,
            scaled_ldm,
            covariance,
        )
        projection = design @ inverse
        beta_tilde = beta_tilde_all[rows]
        fitted = projection @ (design.T @ beta_tilde)
        quadforms[position] = float(beta_tilde @ fitted)
        ranks[position] = float(rank)
        if annotated_design is not None:
            annotated_propagated = scaled_ldm @ annotated_design
            annotated_covariance = annotated_design.T @ annotated_propagated
            annotated_inverse, annotated_rank = pinvh(
                annotated_covariance, return_rank=True, rtol=PINV_RTOL
            )
            annotated_fitted = annotated_design @ (
                annotated_inverse @ (annotated_design.T @ beta_tilde)
            )
            annotated_quadforms[position] = float(beta_tilde @ annotated_fitted)
            annotated_ranks[position] = float(annotated_rank)
            total_operator = (design, inverse)
            annotated_operator = (annotated_design, annotated_inverse)
            enrichment_covariances[position] = (
                _quadratic_form_covariance(
                    beta_tilde,
                    scaled_ldm,
                    annotated_operator,
                    annotated_operator,
                    dataset.sample_size,
                    annotated_rank,
                    annotated_rank,
                ),
                _quadratic_form_covariance(
                    beta_tilde,
                    scaled_ldm,
                    total_operator,
                    total_operator,
                    dataset.sample_size,
                    rank,
                    rank,
                ),
                _quadratic_form_covariance(
                    beta_tilde,
                    scaled_ldm,
                    annotated_operator,
                    total_operator,
                    dataset.sample_size,
                    annotated_rank,
                    rank,
                ),
            )
        if memberships is not None and rank < dataset.sample_size:
            leverage = np.sum(projection * propagated, axis=1)
            contribution = (dataset.sample_size * beta_tilde * fitted - leverage) / (
                dataset.sample_size - rank
            )
            maf_values[position] = memberships[position].T @ contribution

    maf_components = None
    if maf_values is not None:
        maf_components = pl.DataFrame(
            {
                "gene": [gene for gene in genes for _ in maf_names],
                "maf_bin": list(maf_names) * len(genes),
                "hsq": maf_values.ravel().tolist(),
            },
            schema={
                "gene": pl.String,
                "maf_bin": pl.String,
                "hsq": pl.Float64,
            },
        )
    burden_moments = (
        None
        if annotated_quadforms is None
        else (annotated_quadforms, annotated_ranks, enrichment_covariances)
    )
    return quadforms, ranks, maf_components, maf_error, burden_moments


def _fixed_design(model, gene, *, method, burden_1d_wgts, burden_indices):
    """Expanded fixed-effect design and its logical burden basis."""
    rows = model.dataset.gene_slice(gene)
    if method == "indep":
        return np.eye(rows.stop - rows.start), None
    burden_wgts = model.burden_wgts(gene)
    design = (
        (burden_wgts @ burden_1d_wgts)[:, None]
        if method == "burden_1d"
        else burden_wgts[
            :,
            np.arange(model.layout.n_burden)
            if burden_indices is None
            else burden_indices,
        ]
    )
    partition = model.burden_partition_codes(gene)
    if partition is not None:
        design = np.column_stack(
            [
                design[:, column] * (partition == group)
                for column in range(design.shape[1])
                for group in range(partition.max() + 1)
            ]
        )
    return design, burden_wgts


def _fixed_inverse(
    model,
    gene,
    method,
    burden_1d_wgts,
    burden_wgts,
    scaled_ldm,
    covariance,
):
    """Pseudo-inverse with the burden-projection variance threshold."""
    if method == "burden_1d" and model.burden_partition_codes(gene) is None:
        burden_covariance = burden_wgts.T @ (scaled_ldm @ burden_wgts)
        variance = float(covariance[0, 0])
        usable = variance > fitting.VARIANCE_RTOL * float(
            np.diag(burden_covariance) @ np.square(burden_1d_wgts)
        )
        return (
            np.array([[1.0 / variance]]) if usable else np.zeros((1, 1)),
            int(usable),
        )
    return pinvh(covariance, return_rank=True, rtol=PINV_RTOL)


def _decompose_fixed_effects(
    model,
    result,
    columns,
    *,
    burden_1d_wgts=None,
    burden_indices=None,
) -> pl.DataFrame:
    """Allocate one fitted fixed model across exhaustive categorical columns."""
    columns = tuple(dict.fromkeys(columns))
    if not columns:
        raise ValueError("decompose requires at least one column")
    fitted = result.per_gene.filter(pl.col("rank") < result.sample_size)
    genes = fitted["gene"].to_list()
    if not genes:
        raise ValueError("decompose requires at least one estimable fitted gene")
    if result.method == "burden_1d":
        burden_1d_wgts = _validated_burden_1d_wgts(model, burden_1d_wgts)
    frame = model.dataset.variant_df().filter(pl.col("gene").is_in(genes))
    levels = {}
    for column in columns:
        if column not in frame.columns:
            raise KeyError(f"variant_df has no column {column!r}")
        if frame[column].null_count():
            raise ValueError(f"decomposition column {column!r} contains null values")
        levels[column] = frame[column].unique().sort().to_list()
        if len(levels[column]) < 2:
            raise ValueError(
                f"decomposition column {column!r} requires at least two groups"
            )

    accumulators = {
        column: {
            "hsq": np.zeros(len(values)),
            "covariance": np.zeros((len(values), len(values))),
            "n_variants": np.zeros(len(values), dtype=np.int64),
            "scale": np.zeros(len(values)),
        }
        for column, values in levels.items()
    }
    total_scale = 0.0
    sample_size = model.dataset.sample_size
    beta_tilde_all = model.dataset.beta / (model.dataset.se**2 * sample_size)
    for gene in genes:
        rows = model.dataset.gene_slice(gene)
        design, burden_wgts = _fixed_design(
            model,
            gene,
            method=result.method,
            burden_1d_wgts=burden_1d_wgts,
            burden_indices=burden_indices,
        )
        scaled_ldm = model.dataset.load_ldm(gene).scale(model.variant_scale(gene))
        propagated = scaled_ldm @ design
        covariance = design.T @ propagated
        inverse, rank = _fixed_inverse(
            model,
            gene,
            result.method,
            burden_1d_wgts,
            burden_wgts,
            scaled_ldm,
            covariance,
        )
        operator = (design, inverse)
        beta_tilde = beta_tilde_all[rows]
        fitted = _apply_low_rank(*operator, beta_tilde)
        leverage = np.sum((design @ inverse) * propagated, axis=1)
        per_variant = (sample_size * beta_tilde * fitted - leverage) / (
            sample_size - rank
        )
        scale = model.variant_scale(gene) ** 2
        total_scale += float(scale.sum())
        gene_frame = model.dataset.variant_df(gene)
        for column, values in levels.items():
            observed = gene_frame[column].to_numpy()
            masks = [observed == value for value in values]
            accumulator = accumulators[column]
            accumulator["hsq"] += np.asarray(
                [per_variant[mask].sum() for mask in masks]
            )
            accumulator["covariance"] += _allocation_covariance(
                beta_tilde,
                scaled_ldm,
                operator,
                masks,
                sample_size,
                rank,
            )
            accumulator["n_variants"] += np.asarray([mask.sum() for mask in masks])
            accumulator["scale"] += np.asarray([scale[mask].sum() for mask in masks])

    rows = []
    for column, values in levels.items():
        accumulator = accumulators[column]
        covariance = accumulator["covariance"]
        covariance_sum = float(covariance.sum())
        target_variance = result.hsq_se**2
        if covariance_sum > 0.0:
            covariance *= target_variance / covariance_sum
        total_variance = _nonnegative_variance(float(covariance.sum()), "total hsq")
        total = float(accumulator["hsq"].sum())
        for position, value in enumerate(values):
            hsq = float(accumulator["hsq"][position])
            variance = _nonnegative_variance(
                float(covariance[position, position]), "group hsq"
            )
            total_covariance = float(covariance[position].sum())
            proportion = float(accumulator["scale"][position] / total_scale)
            if total == 0.0 or proportion == 0.0:
                share = share_se = enrichment = enrichment_se = np.nan
            else:
                share = hsq / total
                share_variance = _nonnegative_variance(
                    variance / total**2
                    + hsq**2 * total_variance / total**4
                    - 2.0 * hsq * total_covariance / total**3,
                    "hsq share",
                )
                share_se = float(np.sqrt(share_variance))
                enrichment = share / proportion
                enrichment_se = share_se / proportion
            rows.append(
                {
                    "source": f"partition:{column}",
                    "group": str(value),
                    "method": result.method,
                    "n_genes": len(genes),
                    "n_variants": int(accumulator["n_variants"][position]),
                    "hsq": hsq,
                    "hsq_se": float(np.sqrt(variance)),
                    "total_hsq": total,
                    "total_hsq_se": float(np.sqrt(total_variance)),
                    "hsq_total_covariance": total_covariance,
                    "heterozygosity_proportion": proportion,
                    "hsq_share": share,
                    "hsq_share_se": share_se,
                    "enrichment": enrichment,
                    "enrichment_se": enrichment_se,
                }
            )
    return pl.DataFrame(rows)


def _allocation_covariance(beta_tilde, residual, operator, masks, sample_size, rank):
    """Covariance of additive groups under the fixed-fit SE convention."""
    design, inverse = operator
    masks = [np.asarray(mask, dtype=float) for mask in masks]
    masked_designs = [mask[:, None] * design for mask in masks]
    propagated = [residual @ masked for masked in masked_designs]
    cross = [inverse @ (design.T @ projected) for projected in propagated]
    common = inverse @ (design.T @ (residual @ design)) @ inverse
    # Expanding the two symmetrized factors gives four traces in equal pairs.
    trace_terms = 0.5 * np.asarray(
        [
            [
                np.einsum("ij,ji->", cross[row], cross[column])
                + np.einsum("ij,ji->", common, left.T @ right)
                for column, right in enumerate(propagated)
            ]
            for row, left in enumerate(masked_designs)
        ]
    )
    fitted = _apply_low_rank(*operator, beta_tilde)
    projected = np.stack(
        [
            (mask * fitted + _apply_low_rank(*operator, mask * beta_tilde)) / 2.0
            for mask in masks
        ]
    )
    signal_terms = projected @ (residual @ projected.T)
    total_quadform = float(beta_tilde @ fitted)
    total_hsq = (sample_size * total_quadform - rank) / (sample_size - rank)
    covariance = 2.0 * trace_terms / sample_size**2
    signal_total = float(signal_terms.sum())
    if signal_total > 0.0 and total_hsq > 0.0:
        covariance += 4.0 * total_hsq / sample_size * signal_terms / signal_total
    return (covariance + covariance.T) / 2.0


def _quadratic_form_covariance(
    beta_tilde,
    residual,
    left_operator,
    right_operator,
    sample_size,
    left_rank,
    right_rank,
):
    """Covariance of two debiased fixed estimators without noise-as-signal."""
    if left_rank >= sample_size or right_rank >= sample_size:
        return 0.0
    cross = left_operator[0].T @ (residual @ right_operator[0])
    trace_term = float(np.trace(left_operator[1] @ cross @ right_operator[1] @ cross.T))
    left_projected = _apply_low_rank(*left_operator, beta_tilde)
    right_projected = _apply_low_rank(*right_operator, beta_tilde)
    left_signal = float(left_projected @ (residual @ left_projected))
    right_signal = float(right_projected @ (residual @ right_projected))
    left_quadform = float(beta_tilde @ left_projected)
    right_quadform = float(beta_tilde @ right_projected)
    left_variance = _fixed_quadform_variance(left_quadform, left_rank, sample_size)
    right_variance = _fixed_quadform_variance(right_quadform, right_rank, sample_size)
    correlation = 0.0
    if left_signal > 0.0 and right_signal > 0.0:
        correlation = float(
            left_projected
            @ (residual @ right_projected)
            / np.sqrt(left_signal * right_signal)
        )
    elif trace_term != 0.0:
        left_cross = left_operator[0].T @ (residual @ left_operator[0])
        right_cross = right_operator[0].T @ (residual @ right_operator[0])
        left_trace = float(
            np.trace(left_operator[1] @ left_cross @ left_operator[1] @ left_cross)
        )
        right_trace = float(
            np.trace(right_operator[1] @ right_cross @ right_operator[1] @ right_cross)
        )
        if left_trace > 0.0 and right_trace > 0.0:
            correlation = trace_term / np.sqrt(left_trace * right_trace)
    return float(
        np.sqrt(left_variance * right_variance) * np.clip(correlation, -1.0, 1.0)
    )


def _fixed_quadform_variance(quadform, rank, sample_size):
    """Reference fixed-effect variance for one quadratic-form estimator."""
    hsq = (sample_size * quadform - rank) / (sample_size - rank)
    variance = (
        (sample_size / (sample_size - rank)) ** 2
        * (2.0 * rank * (1.0 - hsq) / sample_size + 4.0 * hsq)
        * (1.0 - hsq)
        / sample_size
    )
    return max(0.0, float(variance))


def _nonnegative_variance(value, name):
    tolerance = 1e-12 * max(1.0, abs(value))
    if value < -tolerance:
        raise ValueError(f"materially negative {name} variance: {value}")
    return max(0.0, value)
