"""Numerical machinery shared by the random- and fixed-effect estimators.

Nothing here knows about ``Dataset``, ``Model``, or labeled tables: every function takes
arrays and returns arrays.

Stage 4 contributes the burden projection: a one-dimensional summary of a
gene's burden components, and the weight vector that makes that summary as
powerful as possible across a significant-gene set.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

import clarabel
import numpy as np
import scipy.sparse as sps
from scipy import linalg, optimize, stats

from kasm._logging import logger

# A projected variance is treated as absent when it falls this far below the
# scale of the projection itself, `sum_i w_i^2 G_ii`. The floor has to be
# relative: real burden covariances have entries of order 1e-4 to 1e-1, so an
# absolute floor near zero would accept a variance of 1e-14 — pure roundoff
# from a rank-deficient gene — and turn it into an enormous z-score. Several
# real genes have rank-deficient burden covariance, so this is live geometry.
VARIANCE_RTOL = 1e-10
ANNOTATION_BLOCK_KEYS = (
    "pt0",
    "pt1",
    "tr0",
    "tr1",
    "qf0",
    "qf1",
    "ga1",
    "ng",
    "nv",
)


def _variance_scale(burden_cov: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Natural scale of ``w'Gw`` per gene: ``sum_i w_i^2 G_ii``.

    Bounds the projected variance from above up to the conditioning of ``G``,
    so it is the right yardstick for deciding that a variance is roundoff.
    """
    return np.einsum("gii,i->g", burden_cov, weights * weights)


def burden_1d_statistics(
    burden_beta: np.ndarray,
    burden_cov: np.ndarray,
    weights: np.ndarray,
    sample_size: float,
) -> dict:
    """Per-gene burden test through a fixed one-dimensional projection.

    Under the null, ``b = U' beta_tilde`` has covariance ``G / N`` for
    ``G = U' R_s U``: with ``d = 1/(se sqrt(N))``, ``D diag(se) = I / sqrt(N)``,
    so ``Var(beta_tilde) = D R D / N = R_s / N``. The projected statistic
    ``w'b / sqrt(w'Gw)`` is therefore a z-score once scaled by ``sqrt(N)``.

    Parameters
    ----------
    burden_beta : ndarray, shape (n_genes, n_burden)
    burden_cov : ndarray, shape (n_genes, n_burden, n_burden)
    weights : ndarray, shape (n_burden,)
    sample_size : float

    Returns
    -------
    dict of arrays, each shape (n_genes,): ``beta``, ``variance``, ``gamma``,
    ``zscore``, ``pvalue``. Genes whose projected variance is indistinguishable
    from roundoff get NaN rather than a huge z-score.
    """
    weights = np.asarray(weights, dtype=float)
    projected = burden_beta @ weights
    variance = np.einsum("i,gij,j->g", weights, burden_cov, weights)

    usable = variance > VARIANCE_RTOL * _variance_scale(burden_cov, weights)
    gamma = np.full(len(projected), np.nan)
    gamma[usable] = projected[usable] / np.sqrt(variance[usable])
    zscore = gamma * np.sqrt(sample_size)
    pvalue = 2.0 * stats.norm.sf(np.abs(zscore))

    return {
        "beta": projected,
        "variance": variance,
        "gamma": gamma,
        "zscore": zscore,
        "pvalue": pvalue,
    }


def burden_full_statistics(
    burden_beta: np.ndarray,
    burden_cov: np.ndarray,
    sample_size: float,
) -> dict:
    """Per-gene test of every retained burden component jointly."""
    chisq = np.zeros(len(burden_beta))
    dof = np.zeros(len(burden_beta))
    for position, (score, covariance) in enumerate(zip(burden_beta, burden_cov)):
        inverse, rank = linalg.pinvh(covariance, return_rank=True, rtol=1e-6)
        chisq[position] = float(score @ inverse @ score * sample_size)
        dof[position] = float(rank)
    return {"chisq": chisq, "dof": dof, "pvalue": stats.chi2.sf(chisq, dof)}


def _objective_and_gradient(weights, burden_beta, burden_cov, scale=1.0):
    """Negated total chi-square and its exact gradient.

    ``f(w) = sum_g (w'b_g)^2 / (w'G_g w)``, the total per-gene burden
    chi-square. Its gradient is written out rather than differentiated
    numerically or by an autodiff dependency:

        d/dw = sum_g 2 t_g / d_g * (b_g - t_g / d_g * G_g w)

    for ``t_g = b_g'w`` and ``d_g = w'G_g w``.

    ``scale`` multiplies both, so the optimizer's absolute tolerances see an
    O(1) problem whatever the units of the data.
    """
    projected = burden_beta @ weights
    covariance_weights = burden_cov @ weights  # (n_genes, n_burden)
    variance = covariance_weights @ weights  # (n_genes,)
    # Relative floor: keeps the ratio bounded for near-singular directions
    # instead of letting it reach 1e297 and overflow when squared.
    variance = np.maximum(
        variance, VARIANCE_RTOL * _variance_scale(burden_cov, weights)
    )

    ratio = projected / variance
    value = float(projected @ ratio)
    gradient = 2.0 * (ratio @ burden_beta - (ratio * ratio) @ covariance_weights)
    return -value * scale, -gradient * scale


def _generalized_eigenvector_start(burden_beta, burden_cov):
    """Closed-form start: the pooled-covariance solution.

    If every gene shared one covariance, the objective would be the Rayleigh
    quotient of ``sum_g b_g b_g'`` against it, whose maximizer is the leading
    generalized eigenvector. Genes do not share a covariance, so this is a
    starting point rather than an answer — but a principled one.
    """
    scatter = burden_beta.T @ burden_beta
    pooled = burden_cov.sum(axis=0)
    try:
        eigenvalues, eigenvectors = linalg.eigh(scatter, pooled)
        return eigenvectors[:, int(np.argmax(eigenvalues))]
    except (linalg.LinAlgError, ValueError):
        eigenvalues, eigenvectors = np.linalg.eigh(linalg.pinvh(pooled) @ scatter)
        return np.real(eigenvectors[:, int(np.argmax(np.real(eigenvalues)))])


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Scale to unit maximum magnitude, oriented so the first entry is positive.

    The objective is invariant to scaling and sign, so a representative has to
    be chosen for the answer to be reproducible and comparable.
    """
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all():
        raise RuntimeError("optimizer returned non-finite weights")
    scale = np.abs(weights).max()
    if scale == 0.0:
        raise RuntimeError("optimizer returned an all-zero weight vector")
    anchor = weights[0]
    if anchor == 0.0:
        nonzero = np.flatnonzero(weights)
        anchor = weights[nonzero[0]] if nonzero.size else 1.0
    return weights / (scale if anchor > 0 else -scale)


def optimal_1d_weights(
    burden_beta: np.ndarray,
    burden_cov: np.ndarray,
    *,
    n_random_starts: int = 16,
    seed: int = 0,
) -> np.ndarray:
    """Burden weights maximizing total per-gene chi-square.

    Maximizes ``sum_g (w'b_g)^2 / (w'G_g w)`` over the significant-gene set. The
    objective is scale- and sign-invariant, so the result is normalized.

    Deterministic: the starting points are the closed-form pooled solution, the
    unit vector, and a fixed seeded set of random directions.

    Two numerical points, both of which cost real accuracy if skipped. The
    objective is rescaled to O(1) before optimizing, because L-BFGS-B's
    convergence tests are absolute and the raw objective can be O(1e-4) — the
    answer would otherwise depend on the units of the input. And the search
    genuinely needs many starts: on ill-conditioned problems six starts can miss
    the optimum by several percent, and because this objective is flat near its
    maximum a small objective gap can mean a nearly orthogonal weight vector.
    For a small significant-gene set the maximizer is weakly identified whatever
    the search.

    Parameters
    ----------
    burden_beta : ndarray, shape (n_genes, n_burden)
    burden_cov : ndarray, shape (n_genes, n_burden, n_burden)
    n_random_starts, seed : int
        Extra starting points and the seed that fixes them.

    Returns
    -------
    ndarray, shape (n_burden,)
    """
    burden_beta = np.asarray(burden_beta, dtype=float)
    burden_cov = np.asarray(burden_cov, dtype=float)
    n_burden = burden_beta.shape[1]
    if n_burden == 0:
        raise ValueError("no burden columns to weight")
    # Reject non-finite inputs here rather than letting every start fail
    # obscurely: a NaN on a diagonal would otherwise be silently filtered out
    # as an uninformative gene.
    if not np.isfinite(burden_beta).all() or not np.isfinite(burden_cov).all():
        raise ValueError("burden statistics contain non-finite values")

    # "Any burden variance" means a nonzero diagonal. The reference tests the
    # sum of all entries, which is the variance of the unweighted-sum
    # projection and can vanish for a gene whose covariance does not.
    informative = np.einsum("gii->g", burden_cov) > 0
    if not informative.any():
        raise ValueError("no gene has any burden variance")
    burden_beta, burden_cov = burden_beta[informative], burden_cov[informative]

    if n_burden == 1:
        return np.ones(1)

    rng = np.random.default_rng(seed)
    starts = [
        _generalized_eigenvector_start(burden_beta, burden_cov),
        np.ones(n_burden),
        *(rng.standard_normal(n_burden) for _ in range(n_random_starts)),
    ]

    # Put the objective on an O(1) scale so absolute tolerances mean something.
    reference = abs(
        _objective_and_gradient(np.ones(n_burden), burden_beta, burden_cov)[0]
    )
    scale = 1.0 / reference if reference > 0 else 1.0

    results, failures = [], []
    for index, start in enumerate(starts):
        if not np.isfinite(start).all() or not np.any(start):
            failures.append(f"start {index}: degenerate")
            continue
        result = optimize.minimize(
            _objective_and_gradient,
            normalize_weights(start),
            args=(burden_beta, burden_cov, scale),
            jac=True,
            method="L-BFGS-B",
            options={"ftol": 1e-14, "gtol": 1e-12, "maxiter": 1000},
        )
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            failures.append(f"start {index}: non-finite result")
            continue
        results.append(result)

    if not results:
        raise RuntimeError(
            "optimal_1d_weights failed to converge from any start "
            f"({'; '.join(failures)})"
        )

    results.sort(key=lambda result: float(result.fun))
    return normalize_weights(results[0].x)


def total_chisq(burden_beta, burden_cov, weights) -> float:
    """``sum_g (w'b_g)^2 / (w'G_g w)`` — the objective, for diagnostics."""
    return -_objective_and_gradient(
        np.asarray(weights, dtype=float),
        np.asarray(burden_beta, dtype=float),
        np.asarray(burden_cov, dtype=float),
    )[0]


def total_chisq_gradient(burden_beta, burden_cov, weights) -> np.ndarray:
    """Gradient of :func:`total_chisq`, for convergence checks."""
    return -_objective_and_gradient(
        np.asarray(weights, dtype=float),
        np.asarray(burden_beta, dtype=float),
        np.asarray(burden_cov, dtype=float),
    )[1]


# ---------------------------------------------------------------------------
# Jackknife blocks
# ---------------------------------------------------------------------------


def assign_jackknife_blocks(n_genes: int, n_blocks: int) -> np.ndarray:
    """Block label per gene, as contiguous runs over the gene order.

    Each analysis assigns its own genes. Joint and meta-analysis code pairs
    delete replicates by block position; gene membership need not agree across
    analyses.

    This deliberately differs from the reference, which splits gene order after
    filtering. Every jackknife standard error therefore differs from the
    published tables, by design.
    """
    if n_genes <= 0:
        raise ValueError("no genes to block")
    if n_blocks < 1:
        raise ValueError("n_jk_blocks must be at least 1")
    sizes = [
        len(part) for part in np.array_split(np.arange(n_genes), min(n_blocks, n_genes))
    ]
    return np.repeat(np.arange(len(sizes)), sizes)


def jackknife_se(replicates: np.ndarray) -> np.ndarray:
    """Delete-one-block jackknife standard error.

    ``sqrt((B - 1) / B * sum_b (x_b - mean)^2)`` over leave-one-block-out
    estimates. The reference omits the ``1/B``, which inflates the error by
    ``sqrt(B / (B - 1))`` — 0.5% at 100 blocks but 15% at 4, so the textbook
    form is used here.
    """
    replicates = np.asarray(replicates, dtype=float)
    n_blocks = replicates.shape[0]
    if n_blocks < 2:
        return np.full(replicates.shape[1:], np.nan)
    deviations = replicates - replicates.mean(axis=0)
    return np.sqrt((n_blocks - 1) / n_blocks * (deviations**2).sum(axis=0))


# ---------------------------------------------------------------------------
# The moment system
# ---------------------------------------------------------------------------


def _gram_solve(
    system: np.ndarray, right: np.ndarray, *, least_squares: bool = False
) -> tuple[np.ndarray, int]:
    """Solve a Gram system in unit-diagonal coordinates."""
    scale = np.sqrt(np.diag(system))
    normalized = system / np.outer(scale, scale)
    normalized_right = right / scale
    rank = int(np.linalg.matrix_rank(normalized))
    if least_squares:
        solution = np.linalg.lstsq(normalized, normalized_right, rcond=None)[0]
    else:
        if rank < len(system):
            return np.zeros_like(normalized_right), rank
        condition = np.linalg.cond(normalized)
        if condition > 1e10:
            logger.warning(
                "event=near_singular_moment_system condition_number=%.6g", condition
            )
        solution = np.linalg.solve(normalized, normalized_right)
    return solution / scale, rank


def moment_solve(
    pair_trace: np.ndarray,
    trace: np.ndarray,
    quadform: np.ndarray,
    sample_size: float,
    *,
    least_squares: bool = False,
) -> tuple[np.ndarray, float]:
    """Solve the method-of-moments system for one aggregate.

    The estimating equations are ``E[b' K_i b] = sum_k theta_k
    tr(K_i R_s K_k R_s) + tr(K_i R_s) / N``, i.e.

        pair_trace @ theta = quadform - trace / N

    with the phenotype normalization ``sum_k theta_k tr_k + residual = 1``
    fixing the residual. Kernels that are structurally empty for this aggregate
    have a zero diagonal and are dropped rather than making the system
    singular.

    ``least_squares`` takes the minimum-norm solution when the system is
    singular.  This is appropriate for an interior alternating-fit solve, whose
    answer is only a starting point for the other half of the alternation.  A
    reported variance-component solve must remain full-rank.

    Returns ``(theta, residual)``.
    """
    right = quadform - trace / sample_size
    theta = np.zeros(len(right))
    live = np.diag(pair_trace) != 0
    if live.any():
        index = np.flatnonzero(live)
        system = pair_trace[np.ix_(index, index)]
        if least_squares:
            theta[index], _ = _gram_solve(system, right[index], least_squares=True)
            return theta, float(1.0 - trace @ theta)
        # ``pair_trace`` is a Gram matrix whose columns can have very different
        # units. Rank must not change when a kernel is rescaled, so classify the
        # corresponding unit-diagonal matrix rather than the raw system.
        solution, rank = _gram_solve(system, right[index])
        if rank < len(index):
            raise np.linalg.LinAlgError(
                f"the moment system is singular ({len(index)} live kernels, "
                f"rank {rank}). The usual cause is two kernels that are "
                "indistinguishable in this configuration: an identity kernel "
                "together with MAF-stratified overdispersion is collinear unless "
                "the gene coefficient varies, so it needs gene_annot_cols with "
                "nonconstant fitted coefficients."
            )
        theta[index] = solution
    return theta, float(1.0 - trace @ theta)


def annotation_weighted_sums(values, gene_annot, block_of_gene, n_blocks):
    """Sum gene-major arrays and their binary-annotation contractions by block."""
    shape = values.shape[1:]
    plain = np.zeros((n_blocks, *shape))
    weighted = np.zeros((n_blocks, *shape, gene_annot.shape[1]))
    for block in range(n_blocks):
        rows = np.flatnonzero(block_of_gene == block)
        if not rows.size:
            continue
        selected = values[rows]
        plain[block] = selected.sum(axis=0)
        weighted[block] = (
            selected.reshape(len(rows), -1).T @ gene_annot[rows]
        ).reshape(*shape, gene_annot.shape[1])
    return plain, weighted


def annotation_block_sums(
    pair_trace, trace, quadform, gene_annot, block_of_gene, n_blocks
):
    """Sum gene-major pt/tr/qf moments without changing their trailing shapes.

    Packed base triangles and rectangular extension rows use the same gene-axis
    contractions. No packing, unpacking, support counting or totals belong here.
    Binary, mutually exclusive annotations need only the zero- and one-power
    sums: their two-power sums are identical. Explicit n_blocks preserves empty
    canonical blocks after exclusion.
    """
    gene_annot = validate_gene_annotations(gene_annot)
    if gene_annot.ndim != 2:
        raise ValueError("gene annotations must be a matrix")
    blocks = {}
    for prefix, values in (("pt", pair_trace), ("tr", trace), ("qf", quadform)):
        blocks[prefix + "0"], blocks[prefix + "1"] = annotation_weighted_sums(
            values, gene_annot, block_of_gene, n_blocks
        )
    return blocks


def empty_annotation_blocks(n_blocks: int, n_kernels: int, n_annot: int) -> dict:
    """Allocate a complete block-level moment system for extension assembly."""
    return {
        "pt0": np.zeros((n_blocks, n_kernels, n_kernels)),
        "pt1": np.zeros((n_blocks, n_kernels, n_kernels, n_annot)),
        "tr0": np.zeros((n_blocks, n_kernels)),
        "tr1": np.zeros((n_blocks, n_kernels, n_annot)),
        "qf0": np.zeros((n_blocks, n_kernels)),
        "qf1": np.zeros((n_blocks, n_kernels, n_annot)),
        "ga1": np.zeros((n_blocks, n_annot)),
        "ng": np.zeros(n_blocks),
        "nv": np.zeros(n_blocks),
    }


def validate_gene_annotations(gene_annot: np.ndarray) -> np.ndarray:
    """Validate a binary, mutually exclusive gene-annotation design."""
    gene_annot = np.asarray(gene_annot)
    if gene_annot.ndim not in (1, 2):
        raise ValueError("gene annotations must be a vector or matrix")
    if not np.isfinite(gene_annot).all():
        raise ValueError("gene annotations must be finite")
    if not np.all((gene_annot == 0) | (gene_annot == 1)):
        raise ValueError("gene annotations must contain exact binary 0/1 values")
    annotation_count = gene_annot.sum(axis=-1)
    if np.any(annotation_count > 1):
        raise ValueError(
            "gene annotations must be mutually exclusive (at most one nonzero per gene)"
        )
    return gene_annot


def require_nonempty_blocks(block_of_gene: np.ndarray, n_blocks: int) -> np.ndarray:
    """Return every canonical block, rejecting blocks emptied by exclusion."""
    counts = np.bincount(block_of_gene, minlength=n_blocks)
    empty = np.flatnonzero(counts == 0)
    if len(empty):
        raise ValueError(
            "every jackknife block must contain a retained gene; "
            f"empty blocks: {empty.tolist()}"
        )
    return np.arange(n_blocks, dtype=np.intp)


def _contract_scaled_totals(totals, scales, heritable, overdispersion):
    """Contract kernel moments given one annotation scale per heritable kernel."""
    n_kernels = totals["tr0"].shape[0]
    pair_trace = np.zeros((n_kernels, n_kernels))
    trace = np.zeros(n_kernels)
    quadform = np.zeros(n_kernels)
    if len(heritable):
        pair_trace[np.ix_(heritable, heritable)] = np.einsum(
            "ia,ija,ja->ij",
            scales,
            totals["pt1"][np.ix_(heritable, heritable)],
            scales,
        )
        trace[heritable] = np.einsum("ia,ia->i", totals["tr1"][heritable], scales)
        quadform[heritable] = np.einsum("ia,ia->i", totals["qf1"][heritable], scales)
    if len(overdispersion):
        overdispersion_block = np.ix_(overdispersion, overdispersion)
        pair_trace[overdispersion_block] = totals["pt0"][overdispersion_block]
        trace[overdispersion] = totals["tr0"][overdispersion]
        quadform[overdispersion] = totals["qf0"][overdispersion]
    if len(heritable) and len(overdispersion):
        cross = np.einsum(
            "ia,ija->ij",
            scales,
            totals["pt1"][np.ix_(heritable, overdispersion)],
        )
        pair_trace[np.ix_(heritable, overdispersion)] = cross
        pair_trace[np.ix_(overdispersion, heritable)] = cross.T
    return pair_trace, trace, quadform


def grouped_scaled_totals(
    totals: dict,
    gene_coef: np.ndarray,
    kernel_group: np.ndarray,
    overdispersion: np.ndarray,
):
    """Contract annotation moments with a separate gene scale per kernel group.

    ``kernel_group`` records ownership for each kernel. ``overdispersion``
    explicitly identifies kernels that do not receive gene scaling. This is the generic
    array-level form needed when several biological kernel families have
    independently fitted gene coefficients in one model.
    """
    kernel_groups = np.asarray(kernel_group, dtype=np.intp)
    eta = np.asarray(gene_coef, dtype=float)
    overdispersion = np.asarray(overdispersion, dtype=np.intp)
    n_k = totals["tr0"].shape[0]
    if kernel_groups.shape != (n_k,) or eta.ndim != 2:
        raise ValueError("kernel_group/gene_coef geometry does not match totals")
    heritable = np.setdiff1d(np.arange(len(kernel_groups)), overdispersion)
    if len(heritable) and kernel_groups[heritable].max() >= eta.shape[1]:
        raise ValueError("kernel_group refers to an absent gene-coefficient group")
    if np.any(kernel_groups[heritable] < 0):
        raise ValueError("every ungrouped kernel must be an overdispersion kernel")
    scales = eta[:, kernel_groups[heritable]].T
    return _contract_scaled_totals(totals, scales, heritable, overdispersion)


@dataclass
class RandomEffectsInput:
    """Array-only inputs for one model in a pooled random-effects fit."""

    totals: dict
    kernel_group: np.ndarray
    overdispersion: np.ndarray
    sample_size: float
    kernel_coef_loc: np.ndarray
    weight: float = 1.0
    psd_kernel_basis: np.ndarray | None = None
    psd_annotation_patterns: np.ndarray | None = None


def gene_coef_system(
    item: RandomEffectsInput, local_theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Assemble the grouped gene-coefficient moment system for one model.

    Overdispersion kernels that share a parameter location share one unconstrained
    system component. The returned index arrays identify the independently
    mean-one, nonnegative gene-coefficient groups.
    """
    totals = item.totals
    kernel_groups = np.asarray(item.kernel_group, dtype=np.intp)
    overdispersion = np.asarray(item.overdispersion, dtype=np.intp)
    kernel_coef_loc = np.asarray(item.kernel_coef_loc, dtype=np.intp)
    local_theta = np.asarray(local_theta, dtype=float)
    n_a = totals["tr1"].shape[1]
    n_kernel_groups = int(kernel_groups.max()) + 1
    heritable = np.setdiff1d(np.arange(len(kernel_groups)), overdispersion)
    overdispersion_parameters = np.unique(kernel_coef_loc[overdispersion])
    overdispersion_index = {
        value: i for i, value in enumerate(overdispersion_parameters)
    }
    size = n_a * n_kernel_groups + len(overdispersion_parameters)
    left, trace, quad = np.zeros((size, size)), np.zeros(size), np.zeros(size)
    # Heritable kernels contribute per annotation, so each kernel-group pair
    # block of `left` is diagonal: block (g, h) gets sum_{i in g, j in h}
    # theta_i theta_j pt1[i, j] on its diagonal.
    weights = local_theta[heritable]
    membership = np.eye(n_kernel_groups)[kernel_groups[heritable]]
    n_heritable = n_a * n_kernel_groups
    trace[:n_heritable] = (
        membership.T @ (weights[:, None] * totals["tr1"][heritable])
    ).ravel()
    quad[:n_heritable] = (
        membership.T @ (weights[:, None] * totals["qf1"][heritable])
    ).ravel()
    pair = totals["pt1"][np.ix_(heritable, heritable)] * (
        weights[:, None, None] * weights[None, :, None]
    )
    rows = np.arange(n_kernel_groups)[:, None] * n_a + np.arange(n_a)
    left[rows[:, None, :], rows[None, :, :]] = np.einsum(
        "ig,ija,jh->gha", membership, pair, membership
    )
    for i in overdispersion:
        p = n_a * n_kernel_groups + overdispersion_index[kernel_coef_loc[i]]
        trace[p] += totals["tr0"][i]
        quad[p] += totals["qf0"][i]
        for j in overdispersion:
            left[
                p, n_a * n_kernel_groups + overdispersion_index[kernel_coef_loc[j]]
            ] += totals["pt0"][i, j]
        for j in heritable:
            sj = slice(kernel_groups[j] * n_a, (kernel_groups[j] + 1) * n_a)
            cross = local_theta[j] * totals["pt1"][j, i]
            left[sj, p] += cross
            left[p, sj] += cross
    constrained_groups = [
        np.arange(kernel_group * n_a, (kernel_group + 1) * n_a)
        for kernel_group in range(n_kernel_groups)
    ]
    return left, trace, quad, constrained_groups


START_FLOOR = 1e-3


def gene_coef_starts(shapes: list[tuple[int, int]]) -> list[list[np.ndarray]]:
    """Deterministic multistart set for nonnegative, mean-one gene coefficients.

    The joint kernel/gene-coefficient problem is bilinear and, in weakly
    informative fits, has optima on the simplex boundary (some annotation
    coefficients exactly zero). The set is the uniform start plus, for each
    kernel group in turn (others uniform), equal weights on every annotation
    pair and triple (``START_FLOOR`` elsewhere), applied to that group in every
    pooled model at once; groups with at most three annotations use every proper
    nonempty subset. For five constraint quintiles this is 21 starts per group,
    independent of the number of pooled models.
    """
    uniform = [np.ones(shape) for shape in shapes]
    starts = [uniform]
    n_groups = max((n_g for _, n_g in shapes), default=0)
    for group in range(n_groups):
        sizes_a = sorted({n_a for n_a, n_g in shapes if n_g > group and n_a >= 2})
        for n_a in sizes_a:
            sizes = (2, 3) if n_a >= 4 else range(1, n_a)
            for subset in (c for size in sizes for c in combinations(range(n_a), size)):
                column = np.full(n_a, START_FLOOR)
                column[list(subset)] = 1.0
                column /= column.mean()
                values = [value.copy() for value in uniform]
                for value, (shape_a, shape_g) in zip(values, shapes, strict=True):
                    if shape_g > group and shape_a == n_a:
                        value[:, group] = column
                starts.append(values)
    return starts


def moment_objective(inputs: list[RandomEffectsInput], results: list[dict]) -> float:
    """Pooled profiled moment objective ``sum w (0.5 t'Lt - r't)`` of a fit."""
    total = 0.0
    for item, fit in zip(inputs, results, strict=True):
        pair, trace, quad = grouped_scaled_totals(
            item.totals, fit["gene_coef"], item.kernel_group, item.overdispersion
        )
        theta = fit["theta"]
        right = quad - trace / item.sample_size
        total += item.weight * float(0.5 * theta @ pair @ theta - right @ theta)
    return total


def alternating_fit(
    inputs: list[RandomEffectsInput],
    *,
    init_gene_coef: list[np.ndarray] | None = None,
    fix_gene_coef: bool = False,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> list[dict]:
    """Fit one or more models in one pooled alternating MoM lifecycle.

    Profiling the free residual in the augmented system leaves the Gram
    matrix ``pair_trace`` and target ``quadform - trace / N``. This remains
    true under constraints on kernel/gene coefficients, not on the residual.
    One fix_gene_coef flag applies to every model and group. Without a supplied
    start, estimated gene coefficients are fitted from every start in
    :func:`gene_coef_starts` and the lowest pooled moment objective is kept
    (ties to the earliest start); a supplied start runs once.
    Each model contributes this profiled objective with its explicit weight;
    the augmented system's arbitrary N-squared scale must not enter pooling.
    """
    if init_gene_coef is None and not fix_gene_coef and inputs:
        shapes = [
            (
                item.totals["tr1"].shape[1],
                int(np.max(np.asarray(item.kernel_group))) + 1,
            )
            for item in inputs
        ]
        starts = gene_coef_starts(shapes)
        if len(starts) > 1:
            best, best_value = None, np.inf
            for index, start in enumerate(starts):
                try:
                    results = _alternating_fit_once(
                        inputs,
                        init_gene_coef=start,
                        fix_gene_coef=False,
                        max_iter=max_iter,
                        tol=tol,
                    )
                except (np.linalg.LinAlgError, RuntimeError):
                    if index == 0:
                        raise
                    continue
                value = moment_objective(inputs, results)
                if best is None or value < best_value - 1e-12 * abs(best_value):
                    best, best_value = results, value
            for fit in best:
                fit["objective"] = best_value
                fit["n_starts"] = len(starts)
            return best
    results = _alternating_fit_once(
        inputs,
        init_gene_coef=init_gene_coef,
        fix_gene_coef=fix_gene_coef,
        max_iter=max_iter,
        tol=tol,
    )
    value = moment_objective(inputs, results)
    for fit in results:
        fit["objective"] = value
        fit["n_starts"] = 1
    return results


def _alternating_fit_once(
    inputs: list[RandomEffectsInput],
    *,
    init_gene_coef: list[np.ndarray] | None = None,
    fix_gene_coef: bool = False,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> list[dict]:
    """One alternating MoM fit from one start (uniform when not supplied)."""
    if not inputs:
        raise ValueError("inputs must not be empty")
    if init_gene_coef is not None and len(init_gene_coef) != len(inputs):
        raise ValueError("init_gene_coef must have one array per input")

    prepared = []
    all_kernel_coef_loc = []
    for position, item in enumerate(inputs):
        kernel_groups = np.asarray(item.kernel_group, dtype=np.intp)
        overdispersion = np.asarray(item.overdispersion, dtype=np.intp)
        kernel_coef_loc = np.asarray(item.kernel_coef_loc, dtype=np.intp)
        n_kernels = item.totals["tr0"].shape[0]
        n_annotations = item.totals["tr1"].shape[1]
        if not np.isfinite(item.sample_size) or item.sample_size <= 0:
            raise ValueError("sample_size must be positive and finite")
        if kernel_groups.shape != (n_kernels,) or kernel_coef_loc.shape != (n_kernels,):
            raise ValueError(
                "kernel_group/kernel_coef_loc geometry does not match totals"
            )
        if (
            np.any(kernel_coef_loc < 0)
            or not np.isfinite(item.weight)
            or item.weight <= 0
        ):
            raise ValueError(
                "parameter locations must be nonnegative and weight positive"
            )
        n_kernel_groups = (
            int(kernel_groups.max()) + 1 if np.any(kernel_groups >= 0) else 0
        )
        if n_kernel_groups < 1 or np.any(np.delete(kernel_groups, overdispersion) < 0):
            raise ValueError(
                "kernel groups must partition heritable and overdispersion kernels"
            )
        basis, patterns = item.psd_kernel_basis, item.psd_annotation_patterns
        if (basis is None) != (patterns is None):
            raise ValueError(
                "psd kernel basis and annotation patterns must be supplied together"
            )
        if basis is not None:
            basis, patterns = np.asarray(basis, float), np.asarray(patterns, float)
            if (
                basis.shape[:1] != (n_kernels,)
                or basis.ndim != 3
                or basis.shape[1] != basis.shape[2]
            ):
                raise ValueError("psd_kernel_basis must match n_kernels and be square")
            if (
                patterns.ndim != 2
                or patterns.shape[1] != n_annotations
                or not len(patterns)
            ):
                raise ValueError("psd_annotation_patterns must match n_annotations")
            if not np.isfinite(basis).all() or not np.isfinite(patterns).all():
                raise ValueError("PSD constraint inputs must be finite")
            if not np.allclose(basis, basis.swapaxes(1, 2), rtol=0, atol=1e-12):
                raise ValueError("every PSD kernel basis matrix must be symmetric")
        start = None if init_gene_coef is None else init_gene_coef[position]
        if fix_gene_coef and start is None:
            raise ValueError("fixed grouped gene coefficients require init_gene_coef")
        eta = (
            np.ones((n_annotations, n_kernel_groups))
            if start is None
            else np.asarray(start, float).copy()
        )
        if (
            eta.shape != (n_annotations, n_kernel_groups)
            or not np.isfinite(eta).all()
            or np.any(eta < 0)
        ):
            raise ValueError(
                f"init_gene_coef must have finite nonnegative shape {(n_annotations, n_kernel_groups)}"
            )
        if not fix_gene_coef:
            means = eta.mean(axis=0)
            if np.any(np.abs(means) <= np.finfo(float).eps):
                raise np.linalg.LinAlgError("grouped gene coefficient has zero scale")
            eta /= means
        prepared.append(
            (item, kernel_groups, overdispersion, kernel_coef_loc, basis, patterns, eta)
        )
        all_kernel_coef_loc.append(kernel_coef_loc)
    kernel_coef_loc = np.unique(np.concatenate(all_kernel_coef_loc))
    if not np.array_equal(kernel_coef_loc, np.arange(len(kernel_coef_loc))):
        raise ValueError("kernel_coef_loc must form one global contiguous mapping")
    n_parameters = len(kernel_coef_loc)

    def solve_theta(values):
        left, right = np.zeros((n_parameters, n_parameters)), np.zeros(n_parameters)
        moments = []
        for (
            item,
            kernel_groups,
            overdispersion,
            loc,
            _basis,
            _patterns,
            _,
        ), eta in zip(prepared, values, strict=True):
            pair, trace, quad = grouped_scaled_totals(
                item.totals, eta, kernel_groups, overdispersion
            )
            np.add.at(left, (loc[:, None], loc[None, :]), item.weight * pair)
            np.add.at(right, loc, item.weight * (quad - trace / item.sample_size))
            moments.append(trace)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise RuntimeError("pooled kernel system is non-finite")
        if np.any(np.diag(left) < 0):
            raise np.linalg.LinAlgError("pooled kernel system has negative curvature")
        theta = np.zeros(n_parameters)
        live = np.diag(left) > 0
        if not live.any():
            raise np.linalg.LinAlgError(
                "the pooled moment system has no live parameters"
            )
        if live.any():
            index = np.flatnonzero(live)
            solution, rank = _gram_solve(left[np.ix_(index, index)], right[index])
            if rank < len(index):
                raise np.linalg.LinAlgError(
                    f"the pooled moment system is rank deficient "
                    f"({len(index)} live parameters, rank {rank})"
                )
            theta[index] = solution

        if any(entry[4] is not None for entry in prepared):
            scale = np.ones(n_parameters)
            scale[live] = np.sqrt(np.diag(left)[live])
            inverse = 1 / scale
            psd_blocks = []
            for (
                _item,
                kernel_groups,
                overdispersion,
                loc,
                basis,
                patterns,
                _,
            ), eta in zip(prepared, values, strict=True):
                if basis is not None:
                    heritable = np.setdiff1d(
                        np.arange(len(kernel_groups)), overdispersion
                    )
                    for annotation in patterns:
                        block = np.zeros((n_parameters, *basis[heritable[0]].shape))
                        for k in heritable:
                            block[loc[k]] += (
                                inverse[loc[k]]
                                * (annotation @ eta[:, kernel_groups[k]])
                                * basis[k]
                            )
                        psd_blocks.append(block)
            solution, status = _psd_constrained_qp(
                left * np.outer(inverse, inverse),
                -(right * inverse),
                np.flatnonzero(~live),
                psd_blocks,
            )
            if solution is None:
                raise RuntimeError(
                    f"constrained pooled kernel solve failed: status={status}"
                )
            theta = solution * inverse
        return theta, moments

    def solve_gene(entry, global_theta):
        item, kernel_groups, _overdispersion, loc, _basis, _patterns, _eta = entry
        n_a, n_kernel_groups = item.totals["tr1"].shape[1], int(kernel_groups.max()) + 1
        left, trace, quad, constrained_groups = gene_coef_system(
            item, global_theta[loc]
        )
        # Profiling the augmented residual block leaves this Schur complement.
        solved = _constrained_grouped_simplex_solve(
            left,
            quad - trace / item.sample_size,
            constrained_groups,
        )
        return solved[: n_a * n_kernel_groups].reshape(n_kernel_groups, n_a).T

    values = [entry[-1] for entry in prepared]
    converged, shift, iteration = False, np.inf, 0
    if fix_gene_coef:
        converged, shift, iteration = True, 0.0, 0
    else:
        for iteration in range(1, max_iter + 1):
            theta, _ = solve_theta(values)
            updated = [
                solve_gene(entry, theta) if eta.shape[0] > 1 else eta
                for entry, eta in zip(prepared, values, strict=True)
            ]
            shift = max(
                float(np.max(np.abs(new - old)))
                for new, old in zip(updated, values, strict=True)
            )
            values = updated
            logger.debug(
                "event=fit_iteration method=random iteration=%d shift=%.6g",
                iteration,
                shift,
            )
            if shift < tol:
                converged = True
                break
    theta, traces = solve_theta(values)
    results = []
    for entry, eta, trace in zip(prepared, values, traces, strict=True):
        item, kernel_groups, _overdispersion, loc, basis, patterns, _ = entry
        local_theta = theta[loc]
        if (
            not np.isfinite(local_theta).all()
            or not np.isfinite(eta).all()
            or not np.isfinite(local_theta * trace).all()
        ):
            raise RuntimeError("random fit returned non-finite parameters or variance")
        eigenvalues = np.empty(0)
        if basis is not None:
            covariances = [
                sum(
                    local_theta[k] * (annotation @ eta[:, kernel_groups[k]]) * basis[k]
                    for k in np.setdiff1d(
                        np.arange(len(kernel_groups)), _overdispersion
                    )
                )
                for annotation in patterns
            ]
            eigenvalues = np.asarray(
                [np.linalg.eigvalsh(value).min() for value in covariances]
            )
            scale = max(
                1.0, *(float(np.linalg.norm(value, ord=2)) for value in covariances)
            )
            if eigenvalues.min() < -1e-7 * scale:
                raise RuntimeError(
                    "kernel solve returned a substantively non-PSD covariance"
                )
        results.append(
            {
                "theta": local_theta,
                "residual": float(1 - trace @ local_theta),
                "gene_coef": eta,
                "trace": trace,
                "n_iter": iteration,
                "converged": converged,
                "convergence_delta": float(shift),
                "psd_min_eigenvalues": eigenvalues,
            }
        )
    return results


def _psd_constrained_qp(quadratic, linear, fixed_zero, psd_blocks):
    """Minimize ``0.5 x'Px + q'x`` with ``x[fixed_zero] = 0`` and PSD constraints.

    Each PSD block is an ``(n, d, d)`` array ``B`` constraining
    ``sum_i x_i B[i]`` to be positive semidefinite. Returns ``(x, status)``;
    ``x`` is None unless Clarabel reports a (near-)optimal point.
    """
    n = len(linear)
    rows, cones = [], []
    if len(fixed_zero):
        rows.append(np.eye(n)[fixed_zero])
        cones.append(clarabel.ZeroConeT(len(fixed_zero)))
    for block in psd_blocks:
        # Clarabel's svec: upper triangle by column, off-diagonals times sqrt(2).
        lower_row, lower_col = np.tril_indices(block.shape[1])
        weight = np.where(lower_row == lower_col, 1.0, np.sqrt(2.0))
        rows.append(-(block[:, lower_col, lower_row] * weight).T)
        cones.append(clarabel.PSDTriangleConeT(block.shape[1]))
    constraints = np.vstack(rows)
    options = clarabel.DefaultSettings()
    options.verbose = False
    options.max_iter = 200
    try:
        result = clarabel.DefaultSolver(
            sps.csc_matrix(np.triu(quadratic)),
            np.asarray(linear, float),
            sps.csc_matrix(constraints),
            np.zeros(len(constraints)),
            cones,
            options,
        ).solve()
    except (KeyboardInterrupt, SystemExit):
        raise
    # Clarabel reports invalid input as a Rust panic deriving from BaseException.
    except BaseException as error:  # noqa: BLE001
        return None, f"error: {error}"
    status = str(result.status)
    if status not in {"Solved", "AlmostSolved"}:
        return None, status
    return np.asarray(result.x, dtype=float), status


def _active_set_qp(quadratic, linear, equality, equality_rhs, nonnegative, start):
    """Exact solve of a small convex QP by the primal active-set method.

    Minimizes ``0.5 x'Qx + c'x`` subject to ``Ex = b`` and ``x[nonnegative] >= 0``
    from a feasible ``start``. Each iteration solves the equality-constrained
    subproblem on the free coordinates; directions of zero curvature are
    followed to the nearest bound. Active bounds are exactly zero.
    """
    n = len(linear)
    x = np.array(start, dtype=float)
    bounded = np.zeros(n, dtype=bool)
    bounded[nonnegative] = True
    active = bounded & (x <= 0)
    x[active] = 0.0
    n_equality = len(equality_rhs)
    scale = max(1.0, np.abs(quadratic).max(), np.abs(linear).max())
    for _ in range(20 * n + 50):
        free = np.flatnonzero(~active)
        gradient = quadratic @ x + linear
        k = len(free)
        kkt = np.zeros((k + n_equality, k + n_equality))
        kkt[:k, :k] = quadratic[free][:, free]
        kkt[:k, k:] = equality[:, free].T
        kkt[k:, :k] = equality[:, free]
        rhs = np.zeros(k + n_equality)
        rhs[:k] = -gradient[free]
        try:
            solution = np.linalg.solve(kkt, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(kkt, rhs)[0]
        tiny = 1e-12 * (1 + np.abs(x).max())
        step = np.zeros(n)
        step[free] = solution[:k]
        nu = solution[k:]
        if np.abs(kkt @ solution - rhs).max() > 1e-10 * scale:
            # Zero curvature with descent: move along the projected gradient
            # within the free null space until a bound blocks.
            null = linalg.null_space(
                np.vstack([quadratic[free][:, free], equality[:, free]])
            )
            descent = -null @ (null.T @ gradient[free])
            if np.abs(descent).max() > tiny:
                step[free] = descent
                nu = None
        if nu is not None and np.abs(step).max() <= tiny:
            multiplier = gradient + equality.T @ nu
            candidates = np.flatnonzero(active)
            if not len(candidates) or multiplier[candidates].min() >= -1e-10 * scale:
                return x
            active[candidates[np.argmin(multiplier[candidates])]] = False
            continue
        blocking = np.flatnonzero(bounded & ~active & (step < 0))
        ratios = -x[blocking] / step[blocking]
        if nu is None and not len(blocking):
            raise RuntimeError("active-set QP is unbounded")
        alpha = 1.0 if nu is not None else np.inf
        if len(blocking) and ratios.min() < alpha:
            alpha = ratios.min()
            x = x + alpha * step
            hit = blocking[np.argmin(ratios)]
            x[hit] = 0.0
            active[hit] = True
        else:
            x = x + alpha * step
    raise RuntimeError("active-set QP did not converge")


def _constrained_grouped_simplex_solve(
    pair_trace: np.ndarray,
    right: np.ndarray,
    constrained_groups,
) -> np.ndarray:
    """Solve a moment system with nonnegative, mean-one coefficient groups."""
    pair_trace = np.asarray(pair_trace, dtype=float)
    right = np.asarray(right, dtype=float)
    if pair_trace.shape != (len(right), len(right)):
        raise ValueError("constrained system dimensions do not match")
    kernel_groups = [
        np.asarray(kernel_group, dtype=np.intp) for kernel_group in constrained_groups
    ]
    if not kernel_groups or any(
        not len(kernel_group) for kernel_group in kernel_groups
    ):
        raise ValueError("at least one nonempty constrained group is required")
    constrained = np.concatenate(kernel_groups)
    if len(np.unique(constrained)) != len(constrained):
        raise ValueError("constrained groups must not overlap")
    if np.any(constrained < 0) or np.any(constrained >= len(right)):
        raise ValueError("constrained group index is out of bounds")
    diagonal = np.diag(pair_trace)
    if not np.isfinite(pair_trace).all() or not np.isfinite(right).all():
        raise RuntimeError("constrained gene-coefficient system is non-finite")
    if np.any(diagonal < 0):
        raise RuntimeError("constrained gene-coefficient system has negative curvature")
    if any(not np.any(diagonal[kernel_group] > 0) for kernel_group in kernel_groups):
        raise RuntimeError(
            "constrained gene-coefficient group has no annotation support"
        )
    # Simplex coefficients have natural scale one; normalize the objective by
    # the largest curvature and its live columns to unit diagonal.
    strength = diagonal[constrained].max()
    pair_trace = pair_trace / strength
    right = right / strength
    diagonal = np.diag(pair_trace)
    live = np.flatnonzero(diagonal > 0)
    scale = np.sqrt(diagonal[live])
    position = np.full(len(right), -1, dtype=np.intp)
    position[live] = np.arange(len(live))
    # Coordinates without curvature are fixed at zero; each group's live
    # coordinates share its mean-one total, starting uniform.
    equality = np.zeros((len(kernel_groups), len(live)))
    start = np.zeros(len(live))
    for row, kernel_group in enumerate(kernel_groups):
        members = position[kernel_group][position[kernel_group] >= 0]
        equality[row, members] = 1.0 / scale[members]
        start[members] = scale[members] * len(kernel_group) / len(members)
    normalized = pair_trace[np.ix_(live, live)] / np.outer(scale, scale)
    coefficients = _active_set_qp(
        normalized,
        -right[live] / scale,
        equality,
        np.array([len(kernel_group) for kernel_group in kernel_groups], float),
        position[constrained][position[constrained] >= 0],
        start,
    )
    solution = np.zeros(len(right))
    solution[live] = coefficients / scale
    if not np.isfinite(solution).all():
        raise RuntimeError(
            "constrained gene-coefficient solve returned non-finite values"
        )
    for kernel_group in kernel_groups:
        if solution[kernel_group].min() < -1e-10:
            raise RuntimeError(
                "constrained gene-coefficient solve violated nonnegativity"
            )
        solution[kernel_group] = np.maximum(solution[kernel_group], 0.0)
        solution[kernel_group] *= len(kernel_group) / solution[kernel_group].sum()
    return solution


# ---------------------------------------------------------------------------
# Fixed-effect hsq
# ---------------------------------------------------------------------------


def fixed_hsq_from_quadforms(
    quadforms: np.ndarray, ranks: np.ndarray, sample_size: float
) -> tuple[float, float]:
    """Combine per-gene quadratic forms into a fixed-effect hsq.

    Each gene contributes ``(N q - r) / (N - r)``: the quadratic form debiased
    by the sampling noise its own rank contributes. The standard error is the
    analytic chi-square form, not a jackknife, because these genes were chosen
    for being significant and a jackknife over them would not be exchangeable.
    """
    quadforms = np.asarray(quadforms, dtype=float)
    ranks = np.asarray(ranks, dtype=float)
    usable = ranks < sample_size
    hsq = float(
        (
            (sample_size * quadforms[usable] - ranks[usable])
            / (sample_size - ranks[usable])
        ).sum()
    )
    total_rank = float(ranks[usable].sum())
    error = float(
        np.sqrt(2.0 * total_rank + 4.0 * sample_size * max(0.0, hsq)) / sample_size
    )
    return hsq, error


def jackknife_blocks(genes, gene_blocks, n_jk_blocks):
    """Resolve optional externally defined block identities in gene order."""
    if gene_blocks is None:
        block_of = assign_jackknife_blocks(len(genes), min(n_jk_blocks, len(genes)))
        return block_of, int(block_of.max()) + 1, False
    if isinstance(gene_blocks, Mapping):
        missing = [gene for gene in genes if gene not in gene_blocks]
        if missing:
            raise KeyError(f"genes are missing jackknife blocks: {missing[:5]}")
        values = [gene_blocks[gene] for gene in genes]
    else:
        values = gene_blocks
    block_of = np.asarray(values)
    if block_of.shape != (len(genes),):
        raise ValueError("gene_blocks must have one aligned label per model gene")
    if not np.issubdtype(block_of.dtype, np.integer):
        raise TypeError("gene_blocks labels must be integers")
    block_of = block_of.astype(np.intp, copy=False)
    if np.any((block_of < 0) | (block_of >= n_jk_blocks)):
        raise ValueError(f"gene_blocks labels must be in [0, {n_jk_blocks})")
    return block_of, n_jk_blocks, True
