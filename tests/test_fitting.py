"""Burden projection: optimal-1D weights and the per-gene burden test."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from conftest import SYNTHETIC_MAF_BINS, MAF_BINS
from scipy import stats

from kasm import KernelOptions, Model
from kasm.fitting import (
    RandomEffectsInput,
    alternating_fit,
    burden_1d_statistics,
    burden_full_statistics,
    gene_coef_starts,
    normalize_weights,
    optimal_1d_weights,
    total_chisq,
    total_chisq_gradient,
)


def fit_moments(totals, kernel_group, overdispersion, sample_size, **options):
    """Concise single-input adapter for the numerical unit tests below."""
    start = options.pop("init_gene_coef", None)
    item_fields = {
        key: options.pop(key)
        for key in list(options)
        if key
        in {
            "psd_kernel_basis",
            "psd_annotation_patterns",
        }
    }
    return alternating_fit(
        [
            RandomEffectsInput(
                totals,
                np.asarray(kernel_group),
                np.asarray(overdispersion),
                sample_size,
                np.arange(len(kernel_group)),
                **item_fields,
            )
        ],
        init_gene_coef=None if start is None else [start],
        **options,
    )[0]


def test_grouped_scaling_fit_recovers_three_independent_gene_patterns():
    rng = np.random.default_rng(928)
    n_k, n_a = 4, 3
    x = rng.normal(size=(300, n_k, n_a))
    x[:, 3, 1:] = 0
    pt1 = np.einsum("ria,rja->ija", x, x)
    totals = {
        "pt0": pt1.sum(axis=-1),
        "pt1": pt1,
        "tr0": np.full(n_k, 2.0),
        "tr1": rng.uniform(0.2, 1, (n_k, n_a)),
        "qf0": np.zeros(n_k),
        "qf1": np.zeros((n_k, n_a)),
        "ga1": np.ones(n_a),
        "ng": float(n_a),
        "nv": 100.0,
    }
    groups = np.array([0, 1, 2, -1])
    eta = np.array([[1.2, 0.5, 0.9], [0.7, 1.4, 1.1], [1.0, 0.8, 0.6]])
    eta *= np.sqrt(n_a / np.sum(eta**2, axis=0))
    theta = np.array([0.3, 0.2, 0.1, 0.05])
    from kasm.fitting import grouped_scaled_totals

    pt, tr, _ = grouped_scaled_totals(totals, eta, groups, np.array([3]))
    rhs = pt @ theta + tr / 1000
    for kernel in range(n_k - 1):
        scale = eta[:, groups[kernel]]
        totals["qf1"][kernel] = rhs[kernel] * scale / (scale @ scale)
    totals["qf0"] = rhs + totals["tr0"] / 1000
    fit = fit_moments(
        totals,
        groups,
        np.array([3]),
        1000.0,
        max_iter=20,
        init_gene_coef=eta,
    )
    assert fit["gene_coef"].shape == (n_a, 3)
    assert np.isfinite(fit["theta"]).all()
    np.testing.assert_allclose(fit["gene_coef"].mean(axis=0), 1.0)

    conditional = fit_moments(
        totals,
        groups,
        np.array([3]),
        1000.0,
        init_gene_coef=fit["gene_coef"],
        fix_gene_coef=True,
    )
    np.testing.assert_allclose(conditional["gene_coef"], fit["gene_coef"])
    assert conditional["converged"]
    assert conditional["n_iter"] == 0

    with pytest.raises(ValueError, match="require init_gene_coef"):
        fit_moments(
            totals,
            groups,
            np.array([3]),
            1000.0,
            fix_gene_coef=True,
        )

    rank_deficient = {key: np.copy(value) for key, value in totals.items()}
    for key in ("pt0", "pt1"):
        rank_deficient[key][1] = rank_deficient[key][0]
        rank_deficient[key][:, 1] = rank_deficient[key][:, 0]
    for key in ("tr0", "tr1", "qf0", "qf1"):
        rank_deficient[key][1] = rank_deficient[key][0]
    for psd in (False, True):
        with pytest.raises(np.linalg.LinAlgError, match="rank deficient"):
            fit_moments(
                rank_deficient,
                np.array([0, 0, 1, -1]),
                np.array([3]),
                1000.0,
                init_gene_coef=np.ones((n_a, 2)),
                fix_gene_coef=True,
                psd_kernel_basis=np.zeros((n_k, 1, 1)) if psd else None,
                psd_annotation_patterns=np.ones((1, n_a)) if psd else None,
            )


def _grouped_totals(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(80, 3, 5))
    x[:, 2, 1:] = 0
    pt1 = np.einsum("ria,rja->ija", x, x)
    totals = {
        "pt0": pt1.sum(axis=-1),
        "pt1": pt1,
        "tr0": np.full(3, 2.0),
        "tr1": rng.uniform(0.2, 1, size=(3, 5)),
        "qf0": np.zeros(3),
        "qf1": np.zeros((3, 5)),
        "ga1": np.ones(5),
        "ng": 5.0,
        "nv": 100.0,
    }
    eta = rng.dirichlet(np.ones(5)) * 5
    theta = np.array([0.3, 0.15, 0.08])
    from kasm.fitting import grouped_scaled_totals

    pt, tr, _ = grouped_scaled_totals(
        totals, eta[:, None], np.array([0, 0, -1]), np.array([2])
    )
    rhs = pt @ theta + tr / 1000
    totals["qf1"] = np.outer(rhs, eta) / (eta @ eta)
    totals["qf0"] = rhs + tr / 1000
    return totals


def test_one_group_scaling_handles_zero_support_and_overdispersion():
    from kasm.fitting import grouped_scaled_totals

    rng = np.random.default_rng(1842)
    n_kernels, n_annotations = 5, 6
    moments = rng.normal(size=(70, n_kernels, n_annotations))
    moments[:, 3:, 1:] = 0
    pair_trace = np.einsum("ria,rja->ija", moments, moments)
    totals = {
        "pt0": pair_trace.sum(axis=-1),
        "pt1": pair_trace,
        "tr0": rng.uniform(0.2, 2, n_kernels),
        "tr1": rng.uniform(0.2, 2, (n_kernels, n_annotations)),
        "qf0": rng.normal(size=n_kernels),
        "qf1": rng.normal(size=(n_kernels, n_annotations)),
    }
    totals["tr1"][3:, 1:] = 0
    totals["qf1"][3:, 1:] = 0
    gene_coef = rng.dirichlet(np.ones(n_annotations)) * n_annotations
    heritable = np.array([0, 1, 2])
    overdispersion = np.array([3, 4])

    grouped = grouped_scaled_totals(
        totals,
        gene_coef[:, None],
        np.array([0, 0, 0, -1, -1]),
        overdispersion,
    )
    assert all(np.isfinite(moment).all() for moment in grouped)
    np.testing.assert_allclose(
        grouped[0][np.ix_(overdispersion, overdispersion)],
        totals["pt0"][np.ix_(overdispersion, overdispersion)],
    )
    np.testing.assert_allclose(
        grouped[1][heritable], totals["tr1"][heritable] @ gene_coef
    )


def test_grouped_nonnegative_gene_coef_are_separately_constrained():
    totals = _grouped_totals(729)
    groups = np.array([0, 1, -1])
    fit = fit_moments(
        totals,
        groups,
        np.array([2]),
        1000.0,
        max_iter=3,
    )
    assert fit["gene_coef"].shape == (5, 2)
    assert np.all(fit["gene_coef"] >= -1e-8)
    np.testing.assert_allclose(fit["gene_coef"].sum(axis=0), 5.0, atol=1e-7)


def test_default_gene_coef_multistart_keeps_lowest_objective():
    totals = _grouped_totals(729)
    groups, overdispersion = np.array([0, 1, -1]), np.array([2])
    default = fit_moments(totals, groups, overdispersion, 1000.0, max_iter=3)
    starts = gene_coef_starts([(5, 2)])
    assert len(starts) == 1 + 2 * 20 and default["n_starts"] == len(starts)
    assert len(gene_coef_starts([(5, 1)] * 73)) == 21
    np.testing.assert_array_equal(starts[0][0], np.ones((5, 2)))
    objectives = [
        fit_moments(
            totals, groups, overdispersion, 1000.0, max_iter=3, init_gene_coef=start[0]
        )["objective"]
        for start in starts
    ]
    assert default["objective"] == min(objectives)
    assert default["objective"] <= objectives[0]


def test_alternating_fit_pools_global_moment_system_without_sample_size_weights():
    def item(pair_trace, right, sample_size, weight):
        totals = {
            "pt0": np.array([[pair_trace]]),
            "pt1": np.array([[[pair_trace]]]),
            "tr0": np.array([2.0]),
            "tr1": np.array([[2.0]]),
            "qf0": np.array([right + 2 / sample_size]),
            "qf1": np.array([[right + 2 / sample_size]]),
        }
        return RandomEffectsInput(
            totals,
            np.array([0]),
            np.array([], dtype=int),
            sample_size,
            np.array([0]),
            weight=weight,
        )

    results = alternating_fit(
        [item(2.0, 1.0, 100.0, 3.0), item(5.0, 4.0, 10_000.0, 0.5)],
        init_gene_coef=[np.ones((1, 1)), np.ones((1, 1))],
        fix_gene_coef=True,
    )
    expected = (3 * 1 + 0.5 * 4) / (3 * 2 + 0.5 * 5)
    np.testing.assert_allclose([result["theta"][0] for result in results], expected)


def test_fit_moments_constrains_full_symmetric_covariance():
    n_kernels = 4
    totals = {
        "pt0": np.eye(n_kernels),
        "pt1": np.eye(n_kernels)[:, :, None],
        "tr0": np.zeros(n_kernels),
        "tr1": np.zeros((n_kernels, 1)),
        "qf0": np.array([1.0, 1.0, 2.0, 0.1]),
        "qf1": np.array([[1.0], [1.0], [2.0], [0.0]]),
        "ga1": np.ones(1),
        "ng": 1.0,
        "nv": 2.0,
    }
    groups = np.array([0, 0, 0, -1])
    basis = np.zeros((n_kernels, 2, 2))
    basis[0, 0, 0] = basis[1, 1, 1] = 1.0
    basis[2, 0, 1] = basis[2, 1, 0] = 1.0

    result = fit_moments(
        totals,
        groups,
        np.array([3]),
        1_000.0,
        psd_kernel_basis=basis,
        psd_annotation_patterns=np.ones((1, 1)),
        max_iter=20,
    )

    covariance = sum(result["theta"][i] * basis[i] for i in range(3))
    assert np.linalg.eigvalsh(covariance).min() >= -1e-7
    assert result["psd_min_eigenvalues"].min() >= -1e-7


def test_fit_moments_rejects_asymmetric_psd_basis():
    totals = _grouped_totals(729)
    basis = np.zeros((3, 2, 2))
    basis[0, 0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        fit_moments(
            totals,
            np.array([0, 1, -1]),
            np.array([2]),
            1_000.0,
            psd_kernel_basis=basis,
            psd_annotation_patterns=np.ones((1, 5)),
        )


@pytest.mark.parametrize(
    ("basis", "patterns", "message"),
    [
        (np.zeros((3, 2, 2)), None, "supplied together"),
        (np.zeros((2, 2, 2)), np.ones((1, 5)), "n_kernels"),
        (np.zeros((3, 2, 2)), np.ones((1, 4)), "n_annotations"),
        (np.full((3, 2, 2), np.nan), np.ones((1, 5)), "finite"),
    ],
)
def test_fit_moments_validates_psd_inputs(basis, patterns, message):
    with pytest.raises(ValueError, match=message):
        fit_moments(
            _grouped_totals(730),
            np.array([0, 1, -1]),
            np.array([2]),
            1_000.0,
            psd_kernel_basis=basis,
            psd_annotation_patterns=patterns,
        )


def burden_options(**overrides) -> KernelOptions:
    settings = dict(
        maf_col="af",
        maf_bins=SYNTHETIC_MAF_BINS,
        burden_cols=["constant", "score", "buried"],
        burden_by_maf=False,
        overdispersion_cols=["constant"],
    )
    settings.update(overrides)
    return KernelOptions(**settings)


def random_burden_inputs(n_genes=40, n_burden=3, seed=0, signal=0.3):
    """Well-conditioned burden components with a known preferred direction.

    Covariances are real Gram matrices, and beta carries signal along
    ``truth`` so the optimizer has something to find.
    """
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n_genes, n_burden, n_burden + 2))
    burden_cov = factors @ factors.swapaxes(1, 2)
    truth = np.array([1.0, -0.5, 0.25])[:n_burden]
    noise = np.linalg.cholesky(burden_cov) @ rng.normal(size=(n_genes, n_burden, 1))
    burden_beta = signal * truth + noise[:, :, 0]
    return burden_beta, burden_cov, truth


# ---------------------------------------------------------------------------
# The gradient
# ---------------------------------------------------------------------------


def test_analytic_gradient_matches_finite_differences():
    """The gradient replaces an autodiff dependency, so it is checked directly."""
    from kasm.fitting import _objective_and_gradient

    burden_beta, burden_cov, _ = random_burden_inputs()
    rng = np.random.default_rng(1)
    for _ in range(5):
        weights = rng.normal(size=burden_beta.shape[1])
        _, gradient = _objective_and_gradient(weights, burden_beta, burden_cov)
        numeric = np.zeros_like(gradient)
        step = 1e-6
        for position in range(len(weights)):
            forward, backward = weights.copy(), weights.copy()
            forward[position] += step
            backward[position] -= step
            numeric[position] = (
                _objective_and_gradient(forward, burden_beta, burden_cov)[0]
                - _objective_and_gradient(backward, burden_beta, burden_cov)[0]
            ) / (2 * step)
        np.testing.assert_allclose(gradient, numeric, rtol=1e-5, atol=1e-6)


def test_objective_is_scale_and_sign_invariant():
    burden_beta, burden_cov, _ = random_burden_inputs()
    weights = np.array([0.7, -0.2, 0.4])
    base = total_chisq(burden_beta, burden_cov, weights)
    for factor in (2.0, 0.1, -1.0, -3.0):
        assert total_chisq(burden_beta, burden_cov, factor * weights) == pytest.approx(
            base
        )


# ---------------------------------------------------------------------------
# The optimizer
# ---------------------------------------------------------------------------


def test_weights_beat_every_alternative_direction():
    """The returned weights must be a maximum, not merely a fixed point."""
    burden_beta, burden_cov, _ = random_burden_inputs()
    weights = optimal_1d_weights(burden_beta, burden_cov)
    best = total_chisq(burden_beta, burden_cov, weights)

    rng = np.random.default_rng(7)
    for _ in range(300):
        candidate = rng.normal(size=len(weights))
        assert total_chisq(burden_beta, burden_cov, candidate) <= best * (1 + 1e-9)

    # Also beats each coordinate direction and the unit vector.
    for candidate in [*np.eye(len(weights)), np.ones(len(weights))]:
        assert total_chisq(burden_beta, burden_cov, candidate) <= best * (1 + 1e-9)


def _across_gradient_norm(burden_beta, burden_cov, weights):
    """Gradient norm after removing the scale-invariant radial direction.

    The objective is scale-invariant, so its gradient is orthogonal to w by
    construction; only the component across that direction must vanish.
    Reported relative to the objective, because an absolute threshold is
    meaningless when the real-data objective is O(1e-4).
    """
    gradient = total_chisq_gradient(burden_beta, burden_cov, weights)
    across = gradient - weights * (gradient @ weights) / (weights @ weights)
    value = total_chisq(burden_beta, burden_cov, weights)
    return np.linalg.norm(across) * np.linalg.norm(weights) / abs(value)


def test_weights_are_a_stationary_point():
    burden_beta, burden_cov, _ = random_burden_inputs()
    weights = optimal_1d_weights(burden_beta, burden_cov)
    assert _across_gradient_norm(burden_beta, burden_cov, weights) < 1e-6


def brute_force_optimum(
    burden_beta, burden_cov, n_starts=40, seed=99, method="Nelder-Mead"
):
    """Independent optimum from many tightly-converged starts.

    Deliberately not the production routine: more starts, and a derivative-free
    method so it shares neither the gradient implementation nor its scaling.
    """
    from scipy import optimize

    tolerances = (
        {"xatol": 1e-12, "fatol": 1e-14, "maxiter": 4000}
        if method == "Nelder-Mead"
        else {"xtol": 1e-12, "ftol": 1e-14, "maxiter": 4000}
    )
    rng = np.random.default_rng(seed)
    best = -np.inf
    for _ in range(n_starts):
        start = normalize_weights(rng.standard_normal(burden_beta.shape[1]))
        result = optimize.minimize(
            lambda w: -total_chisq(burden_beta, burden_cov, w),
            start,
            method=method,
            options=tolerances,
        )
        best = max(best, -float(result.fun))
    return best


def test_objective_matches_an_independent_aggressive_search():
    """The failure mode that matters: a local optimum silently preferred."""
    burden_beta, burden_cov, _ = random_burden_inputs()
    weights = optimal_1d_weights(burden_beta, burden_cov)
    achieved = total_chisq(burden_beta, burden_cov, weights)
    assert achieved >= brute_force_optimum(burden_beta, burden_cov) * (1 - 1e-8)


def test_convergence_does_not_depend_on_the_objective_magnitude():
    """Scaling beta scales the objective, and must not change the direction.

    L-BFGS-B's convergence tests are absolute, so without an internal rescaling
    the same problem converges to different tightness at different magnitudes.
    Real data sits near 1e-3, where the default tolerances stop early.
    """
    burden_beta, burden_cov, _ = random_burden_inputs()
    reference = optimal_1d_weights(burden_beta, burden_cov)
    for factor in (1e-4, 1e-2, 1e3):
        scaled_beta = burden_beta * factor
        rescaled = optimal_1d_weights(scaled_beta, burden_cov)
        np.testing.assert_allclose(rescaled, reference, rtol=1e-7, atol=1e-9)
        # And it is genuinely at the optimum of the rescaled problem.
        assert _across_gradient_norm(scaled_beta, burden_cov, rescaled) < 1e-6


def test_weights_recover_a_planted_direction():
    """With strong signal along one direction, the weights should find it."""
    burden_beta, burden_cov, truth = random_burden_inputs(
        n_genes=400, seed=3, signal=3.0
    )
    weights = optimal_1d_weights(burden_beta, burden_cov)
    cosine = abs(weights @ truth / (np.linalg.norm(weights) * np.linalg.norm(truth)))
    assert cosine > 0.9, cosine


def test_optimization_is_deterministic():
    burden_beta, burden_cov, _ = random_burden_inputs()
    first = optimal_1d_weights(burden_beta, burden_cov)
    second = optimal_1d_weights(burden_beta, burden_cov)
    np.testing.assert_array_equal(first, second)


def test_genes_without_burden_variance_are_ignored():
    """A gene contributing no variance must not change the answer."""
    burden_beta, burden_cov, _ = random_burden_inputs()
    reference = optimal_1d_weights(burden_beta, burden_cov)

    padded_beta = np.vstack([burden_beta, np.zeros((3, burden_beta.shape[1]))])
    padded_cov = np.concatenate(
        [burden_cov, np.zeros((3, *burden_cov.shape[1:]))], axis=0
    )
    np.testing.assert_allclose(
        optimal_1d_weights(padded_beta, padded_cov), reference, rtol=1e-8
    )


def test_roundoff_variance_is_not_significant():
    """A rank-deficient gene projects to ~0 variance; that is not significance.

    With an absolute floor near zero, a variance of 1e-14 against a projected
    beta of 1e-2 yields z ~ 1e8 and p = 0 — a certain false positive.
    """
    burden_beta, burden_cov, _ = random_burden_inputs(n_genes=4, seed=21)
    # Make gene 2 exactly rank deficient, then perturb into pure roundoff.
    values, vectors = np.linalg.eigh(burden_cov[2])
    values[0] = 0.0
    burden_cov[2] = vectors @ np.diag(values) @ vectors.T
    null_direction = vectors[:, 0]
    burden_beta[2] = 1e-2 * null_direction

    statistics = burden_1d_statistics(
        burden_beta, burden_cov, null_direction, sample_size=4e5
    )
    assert statistics["variance"][2] < 1e-12
    assert np.isnan(statistics["pvalue"][2])
    assert not (statistics["pvalue"][2] < 0.05 / len(burden_beta))


def test_small_but_real_variance_is_still_usable():
    """The floor separates roundoff from small; it must not swallow small.

    A projected variance four orders below the projection's scale is
    ill-conditioned, not absent, and must still yield a p-value.
    """
    burden_beta, burden_cov, _ = random_burden_inputs(n_genes=3, seed=31)
    values, vectors = np.linalg.eigh(burden_cov[1])
    scale = float(values.max())
    values[0] = 1e-4 * scale
    burden_cov[1] = vectors @ np.diag(values) @ vectors.T
    weights = vectors[:, 0]

    statistics = burden_1d_statistics(burden_beta, burden_cov, weights, 4e5)
    ratio = statistics["variance"][1] / np.einsum(
        "ii,i->", burden_cov[1], weights * weights
    )
    assert 1e-6 < ratio < 1e-2, ratio
    assert np.isfinite(statistics["pvalue"][1])


def test_non_finite_statistics_are_rejected_up_front():
    """A NaN on a diagonal would otherwise be filtered out as uninformative."""
    burden_beta, burden_cov, _ = random_burden_inputs(n_genes=5, seed=32)
    burden_cov[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        optimal_1d_weights(burden_beta, burden_cov)

    burden_beta, burden_cov, _ = random_burden_inputs(n_genes=5, seed=32)
    burden_beta[2, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        optimal_1d_weights(burden_beta, burden_cov)


def test_pooled_start_is_the_exact_optimum_when_genes_share_a_covariance():
    """Unit-test the closed-form start against its own closed-form truth.

    If every gene has the same covariance the objective is a Rayleigh quotient,
    whose maximizer is the leading generalized eigenvector — so the start is
    the answer, and any other eigenvector is not.
    """
    from kasm.fitting import _generalized_eigenvector_start

    rng = np.random.default_rng(41)
    n_burden = 4
    factor = rng.normal(size=(n_burden, n_burden + 3))
    shared = factor @ factor.T
    burden_beta = rng.normal(size=(30, n_burden))
    burden_cov = np.repeat(shared[None], len(burden_beta), axis=0)

    start = _generalized_eigenvector_start(burden_beta, burden_cov)
    assert total_chisq(burden_beta, burden_cov, start) == pytest.approx(
        total_chisq(
            burden_beta, burden_cov, optimal_1d_weights(burden_beta, burden_cov)
        )
    )
    assert _across_gradient_norm(burden_beta, burden_cov, start) < 1e-8


def test_optimizer_survives_a_near_singular_direction():
    """The floored ratio must not overflow when squared."""
    burden_beta, burden_cov, _ = random_burden_inputs(n_genes=6, seed=22)
    values, vectors = np.linalg.eigh(burden_cov[0])
    values[0] = 0.0
    burden_cov[0] = vectors @ np.diag(values) @ vectors.T
    with np.errstate(over="raise", invalid="raise"):
        weights = optimal_1d_weights(burden_beta, burden_cov)
    assert np.isfinite(weights).all()


def test_rejects_inputs_with_no_information():
    for n_burden in (1, 3):
        burden_beta, burden_cov, _ = random_burden_inputs(n_burden=n_burden)
        with pytest.raises(ValueError, match="no gene has any burden variance"):
            optimal_1d_weights(burden_beta, np.zeros_like(burden_cov))
    with pytest.raises(ValueError, match="no burden columns"):
        optimal_1d_weights(np.zeros((5, 0)), np.zeros((5, 0, 0)))


@pytest.mark.parametrize(
    "weights,expected",
    [
        ([2.0, -1.0], [1.0, -0.5]),
        ([-2.0, 1.0], [1.0, -0.5]),
        ([0.0, -4.0], [0.0, 1.0]),
        ([0.5, 0.25], [1.0, 0.5]),
    ],
)
def test_normalize_weights_picks_one_representative(weights, expected):
    np.testing.assert_allclose(normalize_weights(np.array(weights)), expected)


def test_normalize_weights_rejects_degenerate_input():
    with pytest.raises(RuntimeError, match="all-zero"):
        normalize_weights(np.zeros(3))
    with pytest.raises(RuntimeError, match="non-finite"):
        normalize_weights(np.array([1.0, np.nan]))


# ---------------------------------------------------------------------------
# The test statistic
# ---------------------------------------------------------------------------


def test_statistics_match_the_written_out_formula():
    burden_beta, burden_cov, _ = random_burden_inputs(n_genes=6)
    weights = np.array([1.0, -0.5, 0.25])
    sample_size = 1000.0
    statistics = burden_1d_statistics(burden_beta, burden_cov, weights, sample_size)

    for gene in range(len(burden_beta)):
        projected = float(weights @ burden_beta[gene])
        variance = float(weights @ burden_cov[gene] @ weights)
        gamma = projected / np.sqrt(variance)
        zscore = gamma * np.sqrt(sample_size)
        assert statistics["beta"][gene] == pytest.approx(projected)
        assert statistics["variance"][gene] == pytest.approx(variance)
        assert statistics["gamma"][gene] == pytest.approx(gamma)
        assert statistics["zscore"][gene] == pytest.approx(zscore)
        assert statistics["pvalue"][gene] == pytest.approx(
            2 * stats.norm.sf(abs(zscore))
        )


def test_statistics_are_invariant_to_weight_scaling():
    """The z-score is a ratio, so rescaling the projection cannot move it."""
    burden_beta, burden_cov, _ = random_burden_inputs(n_genes=8)
    weights = np.array([1.0, -0.5, 0.25])
    base = burden_1d_statistics(burden_beta, burden_cov, weights, 500.0)
    scaled = burden_1d_statistics(burden_beta, burden_cov, 3.0 * weights, 500.0)
    np.testing.assert_allclose(base["gamma"], scaled["gamma"])
    np.testing.assert_allclose(base["pvalue"], scaled["pvalue"])


def test_zero_variance_genes_get_nan_not_a_spurious_hit():
    burden_beta, burden_cov, _ = random_burden_inputs(n_genes=4)
    burden_cov[2] = 0.0
    statistics = burden_1d_statistics(
        burden_beta, burden_cov, np.ones(burden_beta.shape[1]), 100.0
    )
    assert np.isnan(statistics["gamma"][2])
    assert np.isnan(statistics["pvalue"][2])
    assert np.isfinite(statistics["pvalue"][[0, 1, 3]]).all()


def test_pvalues_are_calibrated_under_the_null():
    """Null betas drawn from the model's own covariance give uniform p-values.

    This is the check that the sqrt(N) scaling and the covariance convention
    agree: a wrong factor would shift the whole distribution.
    """
    rng = np.random.default_rng(11)
    n_genes, n_burden, sample_size = 20000, 3, 30.0
    factors = rng.normal(size=(n_genes, n_burden, n_burden + 2))
    burden_cov = factors @ factors.swapaxes(1, 2)
    # Generated from the DERIVED covariance G / N, not from the implementation's
    # own convention, and at a small N so that an off-by-one in the scaling
    # would be a 1.7% shift in z rather than invisible.
    chol = np.linalg.cholesky(burden_cov / sample_size)
    burden_beta = (chol @ rng.normal(size=(n_genes, n_burden, 1)))[:, :, 0]

    pvalue = burden_1d_statistics(
        burden_beta, burden_cov, np.array([1.0, -0.5, 0.25]), sample_size
    )["pvalue"]
    assert stats.kstest(pvalue, "uniform").pvalue > 1e-3
    assert abs(pvalue.mean() - 0.5) < 0.02
    assert abs((pvalue < 0.05).mean() - 0.05) < 0.01


# ---------------------------------------------------------------------------
# Through the Model, on real data
# ---------------------------------------------------------------------------


def test_full_burden_test_matches_joint_chisquare(synthetic_dataset):
    model = Model(synthetic_dataset, burden_options())
    model.compute_suffstats()
    suffstats = model.suffstats

    observed = model.compute_gene_pvalue(method="burden_full")
    expected = burden_full_statistics(
        suffstats.burden_beta,
        suffstats.burden_cov(),
        synthetic_dataset.sample_size,
    )

    np.testing.assert_allclose(observed["chisq"], expected["chisq"])
    np.testing.assert_allclose(observed["dof"], expected["dof"])
    np.testing.assert_allclose(observed["pvalue"], expected["pvalue"])
    assert model.burden_1d_wgts is None


def test_model_pvalue_agrees_with_the_projection_by_hand(synthetic_dataset):
    model = Model(synthetic_dataset, burden_options())
    model.compute_suffstats()
    suffstats = model.suffstats
    pvalue_df = model.compute_gene_pvalue()
    burden_1d_wgts = np.array(list(model.burden_1d_wgts.values()))

    for position, gene in enumerate(model.dataset.genes):
        cov = suffstats.burden_cov(gene)
        projected = burden_1d_wgts @ suffstats.burden_beta[position]
        variance = burden_1d_wgts @ cov @ burden_1d_wgts
        if variance <= 0:
            continue
        zscore = projected / np.sqrt(variance) * np.sqrt(model.dataset.sample_size)
        assert pvalue_df["zscore"][position] == pytest.approx(zscore)


def test_single_column_model_reproduces_a_plain_z_test(synthetic_dataset):
    """With one burden column the projection is the column itself."""
    model = Model(synthetic_dataset, burden_options(burden_cols=["constant"]))
    model.compute_suffstats()
    suffstats = model.suffstats
    pvalue_df = model.compute_gene_pvalue()
    np.testing.assert_array_equal(list(model.burden_1d_wgts.values()), [1.0])

    beta = suffstats.burden_beta[:, 0]
    variance = suffstats.burden_cov()[:, 0, 0]
    usable = variance > 0
    expected = 2 * stats.norm.sf(
        np.abs(
            beta[usable]
            / np.sqrt(variance[usable])
            * np.sqrt(model.dataset.sample_size)
        )
    )
    np.testing.assert_allclose(pvalue_df["pvalue"].to_numpy()[usable], expected)


def _real_options(**overrides) -> KernelOptions:
    settings = dict(
        maf_col="AF",
        maf_bins=MAF_BINS,
        gene_annot_cols=[f"oebin{i}" for i in range(1, 6)],
        burden_cols=["constant", "am.pathogenicity", "cadd.EsmScoreMissense"],
        burden_by_maf=True,
        overdispersion_cols=["constant"],
        overdispersion_by_maf=True,
    )
    settings.update(overrides)
    return KernelOptions(**settings)


def test_real_weights_reach_the_optimum_at_real_data_magnitude(test_dataset):
    """Convergence where it is hardest: the real objective is O(1e-3).

    The optimizer's convergence tests are absolute, so a tiny objective is
    exactly where an unscaled, loosely-toleranced search stops early. Checked
    against an independent method rather than against itself.
    """
    model = Model(test_dataset, _real_options())
    model.compute_suffstats()
    suffstats = model.suffstats
    burden_beta = suffstats.burden_beta
    burden_cov = suffstats.burden_cov()

    weights = optimal_1d_weights(burden_beta, burden_cov)
    achieved = total_chisq(burden_beta, burden_cov, weights)
    assert achieved < 1.0, "guard the premise: this test is about a small objective"

    independent = brute_force_optimum(
        burden_beta, burden_cov, n_starts=6, seed=5, method="Powell"
    )
    assert achieved >= independent * (1 - 1e-10), (
        f"under-converged: {achieved:.12e} vs {independent:.12e}"
    )
    assert _across_gradient_norm(burden_beta, burden_cov, weights) < 1e-6


def test_real_signif_genes_rank_ahead_of_the_rest(test_dataset, test_signif_genes):
    """The fixture's reference-significant genes should look significant here.

    Not a parity check: the models differ. But a burden test that put the
    reference's signif_genes in the middle of the pack would be broken.
    """
    model = Model(test_dataset, _real_options())
    model.compute_suffstats()
    pvalue_df = model.compute_gene_pvalue()

    present = [gene for gene in test_signif_genes if gene in pvalue_df["gene"]]
    assert len(present) >= 2
    ranks = pvalue_df.with_columns((pl.col("pvalue").rank() / pl.len()).alias("rank"))
    assert ranks.filter(pl.col("gene").is_in(present))["rank"].median() < 0.5


def test_real_weights_are_reused_not_relearned(test_dataset):
    """Calling twice must give the same weights, so downstream reuse is stable."""
    model = Model(test_dataset, _real_options())
    model.compute_suffstats()
    model.compute_gene_pvalue()
    first = model.burden_1d_wgts.copy()
    model.compute_gene_pvalue()
    assert first == model.burden_1d_wgts
