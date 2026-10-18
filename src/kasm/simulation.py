"""LD-aware simulation of causal burden effects and marginal summary statistics.

Simulation is split where the difficulty is. ``simulate_burden_effects`` draws
the structured part — a gene's burden coefficients, from a covariance with
cross-annotation terms and gene-level scaling — and knows nothing about LD
propagation. Diagonal components, an identity kernel or overdispersion, are a
scaled standard normal per variant, so they are left to the caller, which is
also what lets the caller give them a wider variant set than the burden model
has. ``simulate_sumstats`` maps any causal vector through ``D^-1 R D`` and adds
correlated estimation noise, and knows nothing about kernels.

Both simulating functions draw from the global NumPy random state, so a
replicate is made reproducible by seeding with ``np.random.seed`` beforehand.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import lapack

from kasm._logging import logger

_LOG_EVERY = 2000


def mvn_sample(covariance: np.ndarray, n_sim: int) -> np.ndarray:
    """Draw ``n_sim`` columns from ``N(0, covariance)`` by eigendecomposition.

    Negative eigenvalues are clamped to zero rather than rejected. A per-gene
    causal covariance recombines shared kernel coefficients with that gene's
    annotation weights, so it can leave the positive semidefinite cone even
    when every fitted coefficient block is valid on its own; clamping simulates
    the nearest valid covariance instead of failing.
    """
    values, vectors = np.linalg.eigh(covariance)
    return vectors @ (
        np.sqrt(np.maximum(values, 0.0))[:, None]
        * np.random.standard_normal((len(values), n_sim))
    )


def simulate_burden_effects(
    model,
    fit,
    *,
    n_sim: int = 1,
    gene_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Draw causal per-allele burden effects, shape ``(n_variants, n_sim)``.

    Only burden kernels are generated. Identity and overdispersion kernels are
    a scaled standard normal per variant, so the caller adds them directly.

    The simulation model must use ``d = sqrt(2p(1-p))``. The fit may instead
    have used ``d = 1/(se sqrt(N))``: its coefficients parameterize the
    covariance of unscaled per-allele causal effects and are transported
    without rescaling. Transporting them does not preserve hsq;
    analysis workflows own any realized-variance normalization.

    Parameters
    ----------
    n_sim : int
        Number of replicates drawn in one call, sharing each gene's numerical
        work within the batch.
    gene_scale : ndarray or None
        Nonnegative per-gene variance multiplier, shaped ``(n_genes,)`` or
        ``(n_genes, n_sim)``.
    """
    if not model.options.variant_variance_from_maf:
        raise ValueError("simulation requires variant_variance_from_maf=True")
    if gene_scale is not None:
        gene_scale = np.asarray(gene_scale, dtype=float)
        if gene_scale.ndim == 1:
            gene_scale = gene_scale[:, None]
        if gene_scale.shape[0] != model.dataset.n_genes or gene_scale.shape[1] not in (
            1,
            n_sim,
        ):
            raise ValueError("gene_scale must be shaped (n_genes,) or (n_genes, n_sim)")

    source = dict(fit.kernel_options)
    target = model.options.to_dict()
    source.pop("variant_variance_from_maf")
    target.pop("variant_variance_from_maf")
    if source != target:
        raise ValueError(
            "fitted model kernel definitions do not match the simulation model"
        )
    layout = model.layout
    labels = [kernel_id.label for kernel_id in layout.kernel_ids]
    missing = set(labels) - set(fit.kernel_labels)
    if missing:
        raise ValueError(
            f"fitted model does not cover simulation kernels {sorted(missing)}"
        )
    # Each kernel is scaled by its own group's gene coefficient, so a fit whose
    # pLoF and missense kernels scale differently with gene constraint
    # generates what it describes. Non-heritable kernels are left unscaled.
    positions = [fit.kernel_labels.index(label) for label in labels]
    groups = fit.kernel_groups[positions]
    scale = fit.gene_scale(model)
    kernel_coef = np.asarray(fit.kernel_coef)[positions] * np.where(
        np.array([kid.is_heritable for kid in layout.kernel_ids])[None, :],
        scale[:, np.clip(groups, 0, None)],
        1.0,
    )

    beta = np.zeros((model.dataset.n_variants, n_sim))
    started = time.perf_counter()
    for position, gene in enumerate(model.dataset.genes):
        # The burden covariance is small — one row per (annotation, MAF bin) —
        # so its causal effects are drawn there and projected onto variants,
        # rather than forming an (n_variants, n_burden) factor per gene.
        covariance = np.zeros((layout.n_burden, layout.n_burden))
        for column, kernel in enumerate(layout.burden_indices):
            covariance[column, column] = kernel_coef[position, kernel]
        for kernel, (left, right) in zip(layout.cross_indices, layout.cross_pairs):
            covariance[left, right] = kernel_coef[position, kernel]
            covariance[right, left] = kernel_coef[position, kernel]
        burden_wgts = model.burden_wgts(gene)
        partition = model.burden_partition_codes(gene)
        if partition is None:
            values = burden_wgts @ mvn_sample(covariance, n_sim)
        else:
            # Non-overlapping partitions, typically exons, draw independently.
            values = sum(
                (burden_wgts * (partition == code)[:, None])
                @ mvn_sample(covariance, n_sim)
                for code in range(partition.max() + 1)
            )
        if gene_scale is not None:
            values = values * np.sqrt(gene_scale[position])
        beta[model.dataset.gene_slice(gene)] = values
        if (position + 1) % _LOG_EVERY == 0:
            logger.info(
                "simulate_burden_effects: %d/%d genes in %.1fs",
                position + 1,
                model.dataset.n_genes,
                time.perf_counter() - started,
            )
    logger.info(
        "simulate_burden_effects: %d genes x %d replicates in %.1fs",
        model.dataset.n_genes,
        n_sim,
        time.perf_counter() - started,
    )

    return beta


def simulate_sumstats(model, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Propagate causal effects to marginal estimates and add sampling noise.

    The marginal estimate is ``D^-1 R D beta`` plus noise with covariance
    ``diag(se) R diag(se)``, where ``se = 1/sqrt(N d^2)``. Any causal vector is
    accepted, so one component's own summary statistics cost nothing extra.

    Each gene's correlation matrix is factorized once per call and released, so
    the peak memory is one gene rather than the whole genome. Draw every
    replicate of a batch in one call — that is what amortizes the
    factorization.

    Returns
    -------
    tuple of ndarray
        ``(betahat, se)``. ``betahat`` matches the shape of *beta*; ``se`` is
        always ``(n_variants,)``.
    """
    if not model.options.variant_variance_from_maf:
        raise ValueError("simulation requires variant_variance_from_maf=True")
    values = np.asarray(beta, dtype=float)
    flat = values.ndim == 1
    if flat:
        values = values[:, None]
    se = 1.0 / (np.sqrt(model.dataset.sample_size) * model.variant_scale())
    betahat = np.zeros(values.shape)
    started = time.perf_counter()
    genes = model.dataset.genes
    deficient = largest = 0
    for position, gene in enumerate(genes, start=1):
        rows = model.dataset.gene_slice(gene)
        ldm = model.dataset.load_ldm(gene)
        scale = model.variant_scale(gene)
        scaled = scale[:, None] * values[rows]
        marginal = (ldm @ scaled) / scale[:, None]
        correlation = ldm.toarray()
        largest = max(largest, len(correlation))
        # An LD correlation matrix is routinely singular: variants in perfect
        # LD, or absent from the reference panel. A pivoted Cholesky is the
        # positive semidefinite factorization, so there is no failure case to
        # branch on, and it drops the null directions at LAPACK's own rank
        # tolerance instead of an ad-hoc eigenvalue cutoff.
        triangular, pivots, rank, _ = lapack.dpstrf(correlation, lower=1, overwrite_a=1)
        deficient += rank < len(triangular)
        factor = np.zeros((len(triangular), rank))
        factor[pivots - 1] = np.tril(triangular)[:, :rank]
        betahat[rows] = (
            marginal
            + (factor @ np.random.standard_normal((factor.shape[1], values.shape[1])))
            * se[rows][:, None]
        )
        if position % _LOG_EVERY == 0:
            logger.info(
                "simulate_sumstats: %d/%d genes in %.1fs",
                position,
                len(genes),
                time.perf_counter() - started,
            )
    logger.info(
        "simulate_sumstats: %d genes x %d replicates in %.1fs "
        "(largest gene %d variants, %d rank-deficient)",
        len(genes),
        values.shape[1],
        time.perf_counter() - started,
        largest,
        deficient,
    )
    return (betahat[:, 0] if flat else betahat), se.copy()
