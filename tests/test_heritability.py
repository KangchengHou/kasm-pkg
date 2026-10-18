"""Jackknife blocking, the moment fit, and the two hsq estimators.

The moment solve is checked against an oracle written straight from the
generative model the moment equations invert:

    beta_tilde = R_s gamma + e,   Cov(gamma) = sum_k theta_k K_k,
                                  Cov(e) = R_s / N

which gives ``E[b' K_i b] = sum_k theta_k tr(K_i R_s K_k R_s) + tr(K_i R_s)/N``
— exactly the estimating equations.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from conftest import MAF_BINS
from polars.testing import assert_frame_equal

from kasm import (
    Dataset,
    KernelOptions,
    Model,
    fit_fixed_effects,
)
from kasm import (
    fit_random_effects as estimate_random_hsq,
)
from kasm.fitting import (
    _constrained_grouped_simplex_solve,
    annotation_block_sums,
    assign_jackknife_blocks,
    jackknife_se,
    moment_solve,
)
from kasm.fixed_effects import _fit_fixed_effects as estimate_fixed_hsq
from kasm.random_effects import RandomFit


@pytest.mark.parametrize(
    ("right", "expected"),
    [
        (np.array([0.5, 1.5]), np.array([0.5, 1.5])),
        (np.array([-1.0, 3.0]), np.array([0.0, 2.0])),
    ],
)
def test_constrained_gene_solve_interior_and_active_bound(right, expected):
    solution = _constrained_grouped_simplex_solve(np.eye(2), right, [np.arange(2)])
    np.testing.assert_allclose(solution, expected, atol=1e-8)


def test_constrained_gene_solve_handles_overdispersion_and_inactive_annotations():
    system = np.array([[4.0, 1.0, 2.0], [1.0, 3.0, -1.0], [2.0, -1.0, 5.0]])
    expected = np.array([0.5, 1.5, 4.0])
    solution = _constrained_grouped_simplex_solve(
        system, system @ expected, [np.arange(2)]
    )
    np.testing.assert_allclose(solution, expected, atol=1e-7)

    inactive = _constrained_grouped_simplex_solve(
        np.diag([1.0, 0.0, 1.0]),
        np.array([2.0, 0.0, 3.0]),
        [np.arange(2)],
    )
    np.testing.assert_allclose(inactive, [2.0, 0.0, 3.0], atol=1e-7)


@pytest.mark.parametrize("strength", [1e-24, 1e-12, 1.0, 1e12, 1e24])
@pytest.mark.parametrize("active_bound", [False, True])
def test_gene_qp_is_invariant_to_objective_scale(strength, active_bound):
    system = np.array([[4.0, 1.0, 2.0], [1.0, 3.0, -1.0], [2.0, -1.0, 5.0]])
    expected = (
        np.array([0.0, 2.0, -0.2]) if active_bound else np.array([0.3, 1.7, -0.2])
    )
    # Independent KKT oracle: the sole active inequality has multiplier one;
    # the overdispersion parameter is signed and the equality multiplier is zero.
    right = system @ expected - (np.array([1.0, 0.0, 0.0]) if active_bound else 0)
    result = _constrained_grouped_simplex_solve(
        strength * system, strength * right, [np.arange(2)]
    )
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-8)


def test_constrained_profiled_objective_matches_augmented_residual_solve():
    import cvxpy

    pair_trace = np.array([[3.0, 0.4], [0.4, 2.0]])
    trace = np.array([0.7, 1.1])
    quadform = np.array([-0.2, 2.4])
    sample_size = 50.0
    profiled = _constrained_grouped_simplex_solve(
        pair_trace, quadform - trace / sample_size, [np.arange(2)]
    )

    coefficients = cvxpy.Variable(2)
    residual = cvxpy.Variable()
    augmented = np.block(
        [
            [
                (pair_trace + np.outer(trace, trace) / sample_size) * sample_size**2,
                trace[:, None] * sample_size,
            ],
            [trace[None, :] * sample_size, np.array([[sample_size]])],
        ]
    )
    value = cvxpy.hstack([coefficients, residual])
    right = np.r_[quadform * sample_size**2, sample_size]
    problem = cvxpy.Problem(
        cvxpy.Minimize(0.5 * cvxpy.quad_form(value, augmented) - right @ value),
        [coefficients >= 0, cvxpy.sum(coefficients) == 2],
    )
    problem.solve(solver=cvxpy.CLARABEL)
    np.testing.assert_allclose(profiled, coefficients.value, atol=1e-8)


@pytest.mark.parametrize("seed", range(20))
def test_simplex_solve_matches_an_independent_conic_solver(seed):
    """Random grouped systems with active bounds, free and curvature-free rows."""
    import cvxpy

    rng = np.random.default_rng(seed)
    n_groups, n_a, n_free = rng.integers(1, 4), rng.integers(1, 6), rng.integers(0, 3)
    n = n_groups * n_a + n_free
    factor = rng.normal(size=(n, n + 2)) * rng.uniform(0.1, 10, size=n)[:, None]
    pair_trace = factor @ factor.T
    dead = rng.random(n) < 0.15
    dead[np.arange(0, n_groups * n_a, n_a)] = False  # every group keeps support
    pair_trace[dead] = pair_trace[:, dead] = 0.0
    right = rng.normal(size=n) * np.sqrt(np.diag(pair_trace)) * 3
    groups = [np.arange(g * n_a, (g + 1) * n_a) for g in range(n_groups)]

    solution = _constrained_grouped_simplex_solve(pair_trace, right, groups)

    value = cvxpy.Variable(n)
    constraints = [value[dead] == 0] if dead.any() else []
    for group in groups:
        constraints += [value[group] >= 0, cvxpy.sum(value[group]) == len(group)]
    cvxpy.Problem(
        cvxpy.Minimize(
            0.5 * cvxpy.quad_form(value, cvxpy.psd_wrap(pair_trace)) - right @ value
        ),
        constraints,
    ).solve(solver=cvxpy.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12)

    def objective(x):
        return 0.5 * x @ pair_trace @ x - right @ x

    for group in groups:
        assert solution[group].min() >= 0
        np.testing.assert_allclose(solution[group].sum(), len(group), rtol=1e-12)
    np.testing.assert_array_equal(solution[dead], 0.0)
    scale = 1 + abs(objective(value.value))
    assert objective(solution) <= objective(value.value) + 1e-8 * scale
    np.testing.assert_allclose(solution, value.value, atol=1e-5)


# ---------------------------------------------------------------------------
# Jackknife blocks
# ---------------------------------------------------------------------------


def test_blocks_are_contiguous_and_cover_every_gene():
    labels = assign_jackknife_blocks(10, 3)
    assert len(labels) == 10
    # Contiguous runs: the label never decreases and never revisits a value.
    assert (np.diff(labels) >= 0).all()
    assert sorted(set(labels)) == [0, 1, 2]
    assert np.bincount(labels).tolist() == [4, 3, 3]
    assert assign_jackknife_blocks(3, 100).tolist() == [0, 1, 2]


@pytest.mark.parametrize("bad", [(0, 4), (5, 0)])
def test_rejects_degenerate_blocking(bad):
    with pytest.raises(ValueError):
        assign_jackknife_blocks(*bad)


def test_jackknife_se_is_the_textbook_delete_one_form():
    replicates = np.array([[1.0], [2.0], [4.0], [5.0]])
    deviations = replicates[:, 0] - replicates[:, 0].mean()
    expected = np.sqrt(3 / 4 * (deviations**2).sum())
    np.testing.assert_allclose(jackknife_se(replicates), [expected])
    # The reference form omits the 1/B; at four blocks that is 15% larger.
    reference = np.std(replicates[:, 0], ddof=1) * np.sqrt(3)
    assert reference / expected == pytest.approx(np.sqrt(4 / 3))


def test_jackknife_se_needs_at_least_two_blocks():
    assert np.isnan(jackknife_se(np.array([[1.0, 2.0]]))).all()


# ---------------------------------------------------------------------------
# The moment system
# ---------------------------------------------------------------------------


def joint_system_oracle(pair_trace, trace, quadform, sample_size):
    """The reference's joint (n_k + 1) construction, as an independent oracle.

    Builds and solves the augmented system the reference builds, rather than the
    reduced one under test, so agreement is a real check on the reduction.
    """
    n = len(trace)
    lhs = np.zeros((n + 1, n + 1))
    lhs[:n, :n] = (pair_trace + np.outer(trace, trace) / sample_size) * sample_size**2
    lhs[:n, -1] = lhs[-1, :n] = trace * sample_size
    lhs[-1, -1] = sample_size
    right = np.zeros(n + 1)
    right[:n] = quadform * sample_size**2
    right[-1] = sample_size
    solution = np.linalg.solve(lhs, right)
    return solution[:n], float(solution[-1])


def random_moment_inputs(n_kernels=4, seed=0):
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(n_kernels, n_kernels + 3))
    pair_trace = factor @ factor.T
    trace = np.abs(rng.normal(size=n_kernels)) + 0.5
    quadform = rng.normal(size=n_kernels) * 1e-3
    return pair_trace, trace, quadform


def test_moment_solve_matches_the_joint_system():
    pair_trace, trace, quadform = random_moment_inputs()
    theta, residual = moment_solve(pair_trace, trace, quadform, 1e5)
    expected_theta, expected_residual = joint_system_oracle(
        pair_trace, trace, quadform, 1e5
    )
    np.testing.assert_allclose(theta, expected_theta, rtol=1e-8)
    assert residual == pytest.approx(expected_residual, rel=1e-8)


def test_empty_kernels_are_dropped_not_singular():
    pair_trace, trace, quadform = random_moment_inputs(seed=2)
    pair_trace[2, :] = pair_trace[:, 2] = 0.0
    trace[2] = quadform[2] = 0.0
    theta, _ = moment_solve(pair_trace, trace, quadform, 1e5)
    assert theta[2] == 0.0
    live = [0, 1, 3]
    np.testing.assert_allclose(
        pair_trace[np.ix_(live, live)] @ theta[live],
        (quadform - trace / 1e5)[live],
        rtol=1e-9,
    )


def test_moment_solve_rejects_collinear_live_kernels():
    pair_trace = np.array([[1.0, 1.0], [1.0, 1.0]])
    trace = np.array([1.0, 1.0])
    quadform = np.array([2e-4, 2e-4])

    with pytest.raises(np.linalg.LinAlgError, match=r"2 live kernels, rank 1"):
        moment_solve(pair_trace, trace, quadform, 1e5)

    theta, _ = moment_solve(pair_trace, trace, quadform, 1e5, least_squares=True)
    np.testing.assert_allclose(pair_trace @ theta, quadform - trace / 1e5, rtol=1e-12)


def test_moment_rank_is_invariant_to_kernel_units():
    correlation = np.array([[1.0, 0.2], [0.2, 1.0]])
    units = np.array([1e-10, 1.0])
    pair_trace = correlation * np.outer(units, units)
    trace = np.array([1e-8, 1.0])
    expected = np.array([2e6, 3e-4])
    quadform = pair_trace @ expected + trace / 1e5

    theta, _ = moment_solve(pair_trace, trace, quadform, 1e5)

    np.testing.assert_allclose(theta, expected, rtol=1e-12)


def test_singular_least_squares_is_invariant_to_kernel_units():
    pair_trace = np.ones((2, 2))
    trace = np.array([1.0, 1.0])
    right = np.array([2e-4, 2e-4])
    quadform = right + trace / 1e5
    expected, _ = moment_solve(pair_trace, trace, quadform, 1e5, least_squares=True)

    units = np.array([1e-8, 1.0])
    scaled, _ = moment_solve(
        pair_trace * np.outer(units, units),
        trace * units,
        right * units + trace * units / 1e5,
        1e5,
        least_squares=True,
    )

    np.testing.assert_allclose(scaled * units, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Annotation-weighted block sums
# ---------------------------------------------------------------------------


def block_inputs(n_genes=9, n_kernels=3, n_annot=2, n_blocks=3, seed=0):
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(n_genes, n_kernels, n_kernels + 2))
    pair_trace = factor @ factor.swapaxes(1, 2)
    trace = np.abs(rng.normal(size=(n_genes, n_kernels)))
    quadform = rng.normal(size=(n_genes, n_kernels)) * 1e-3
    n_variants = rng.integers(2, 50, size=n_genes)
    gene_annot = np.eye(n_annot)[np.arange(n_genes) % n_annot]
    labels = assign_jackknife_blocks(n_genes, n_blocks)
    return pair_trace, trace, quadform, n_variants, gene_annot, labels


def test_annotation_weighted_sums_match_direct_contraction():
    """Binary annotations need one weighted pair sum; its square is identical."""
    pair_trace, trace, quadform, n_variants, gene_annot, labels = block_inputs()
    blocks = annotation_block_sums(
        pair_trace, trace, quadform, gene_annot, labels, n_blocks=3
    )
    totals = {key: value.sum(axis=0) for key, value in blocks.items()}
    np.testing.assert_allclose(totals["pt0"], pair_trace.sum(axis=0), rtol=1e-12)
    np.testing.assert_allclose(
        totals["pt1"], np.einsum("gij,ga->ija", pair_trace, gene_annot), rtol=1e-12
    )
    assert "pt2" not in totals
    assert "ga2" not in totals
    np.testing.assert_allclose(
        totals["tr1"], np.einsum("gi,ga->ia", trace, gene_annot), rtol=1e-12
    )
    np.testing.assert_allclose(
        totals["qf1"], np.einsum("gi,ga->ia", quadform, gene_annot), rtol=1e-12
    )


@pytest.mark.parametrize(
    ("gene_annot", "message"),
    [
        (np.array([[0.5, 0.0]]), "binary"),
        (np.array([[1.0, 1.0]]), "mutually exclusive"),
    ],
)
def test_block_sums_reject_invalid_gene_annotations(gene_annot, message):
    pair_trace, trace, quadform, n_variants, _, labels = block_inputs(n_genes=1)
    with pytest.raises(ValueError, match=message):
        annotation_block_sums(
            pair_trace, trace, quadform, gene_annot, labels, n_blocks=1
        )


# ---------------------------------------------------------------------------
# The random estimator's contract
# ---------------------------------------------------------------------------


def real_options(**overrides) -> KernelOptions:
    settings = dict(
        maf_col="AF",
        maf_bins=MAF_BINS,
        burden_cols=["constant", "am.pathogenicity"],
        burden_by_maf=True,
        overdispersion_cols=["constant"],
        overdispersion_by_maf=True,
    )
    settings.update(overrides)
    return KernelOptions(**settings)


@pytest.fixture
def fitted(test_dataset):
    model = Model(test_dataset, real_options())
    model.compute_suffstats()
    model.compute_gene_pvalue()
    return model


def test_excluding_everything_is_an_error(fitted):
    fitted.compute_jk_suffstats(exclude_genes=fitted.dataset.genes)
    with pytest.raises(ValueError, match="removes every gene"):
        estimate_random_hsq(fitted, exclude_genes=fitted.dataset.genes)


def test_hsq_is_the_heritable_variance_sum(fitted):
    fitted.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=4)
    result = estimate_random_hsq(fitted, exclude_genes=[], n_jk_blocks=4)
    heritable = np.array(fitted.layout.heritable_indices)
    assert result.hsq == pytest.approx(result.variance[heritable].sum())
    # Overdispersion is excluded, and it is not zero, so this is a real exclusion.
    overdispersion = np.array(fitted.layout.overdispersion_indices)
    assert abs(result.variance[overdispersion].sum()) > 0
    assert result.hsq != pytest.approx(result.variance.sum())


def test_streamed_jackknife_fit_matches_full_statistics(test_dataset):
    excluded = [test_dataset.genes[1]]
    full = Model(test_dataset, real_options())
    full.compute_suffstats()
    full.compute_jk_suffstats(exclude_genes=excluded, n_jk_blocks=4)
    expected = estimate_random_hsq(
        full,
        exclude_genes=excluded,
        n_jk_blocks=4,
        max_iter=10,
        tol=1e-6,
    )

    streamed = Model(test_dataset, real_options())
    streamed.compute_suffstats()
    compact = streamed.compute_jk_suffstats(exclude_genes=excluded, n_jk_blocks=4)
    observed = estimate_random_hsq(
        compact,
        exclude_genes=excluded,
        n_jk_blocks=4,
        max_iter=10,
        tol=1e-6,
    )

    assert observed.n_genes == expected.n_genes
    assert observed.n_variants == expected.n_variants
    np.testing.assert_allclose(observed.hsq, expected.hsq, rtol=1e-10)
    np.testing.assert_allclose(observed.hsq_jk, expected.hsq_jk, rtol=1e-10)
    np.testing.assert_allclose(observed.variance, expected.variance, rtol=1e-10)
    np.testing.assert_allclose(
        observed.gene_coef,
        expected.gene_coef,
        rtol=1e-10,
    )


def test_excluding_genes_matches_a_dataset_without_them(
    test_dataset, test_signif_genes
):
    """Exclusion must drop a gene's contribution, not merely down-weight it."""
    excluded = list(test_signif_genes[:2])
    full = Model(test_dataset, real_options())
    full.compute_suffstats()
    full.compute_jk_suffstats(exclude_genes=excluded, n_jk_blocks=1)
    with_exclusion = estimate_random_hsq(full, exclude_genes=excluded, n_jk_blocks=1)

    kept = test_dataset.variant_df().filter(~pl.col("gene").is_in(excluded))
    smaller = Model(
        Dataset(
            kept,
            gene_df=test_dataset.gene_df(),
            beta_col="BETA",
            se_col="SE",
            sample_size=test_dataset.sample_size,
            ldm=test_dataset.ldm_source,
        ),
        real_options(),
    )
    smaller.compute_suffstats()
    smaller.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=1)
    direct = estimate_random_hsq(smaller, exclude_genes=[], n_jk_blocks=1)

    assert with_exclusion.n_genes == direct.n_genes
    assert with_exclusion.hsq == pytest.approx(direct.hsq, rel=1e-9)


def test_excluding_a_whole_block_keeps_the_replicates_honest(test_dataset):
    """F2: an emptied canonical block invalidates the jackknife.

    Leaving out a block that retains no gene changes nothing, so its replicate
    would equal the point estimate and deflate the standard error. Dropping or
    renumbering it would destroy cross-fit alignment, so the fit must fail.
    """
    model = Model(test_dataset, real_options())
    model.compute_suffstats()
    genes = model.dataset.genes
    labels = assign_jackknife_blocks(len(genes), 4)
    doomed = [gene for gene, label in zip(genes, labels) if label == 3]
    assert doomed, "fixture should have a highest block to empty"

    model.compute_jk_suffstats(exclude_genes=doomed, n_jk_blocks=4)
    with pytest.raises(ValueError, match="every jackknife block.*empty blocks"):
        estimate_random_hsq(model, exclude_genes=doomed, n_jk_blocks=4)


def test_unknown_excluded_genes_are_rejected(fitted):
    """A stale significant-gene list would otherwise exclude nothing, silently."""
    with pytest.raises(KeyError, match="not in this model"):
        fitted.compute_jk_suffstats(exclude_genes=["NOT_A_GENE"], n_jk_blocks=2)


def test_unconverged_jackknife_retains_finite_standard_errors(fitted, monkeypatch):
    from kasm import fitting

    original = fitting.alternating_fit
    calls = 0

    def force_one_failure(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 2:
            result = [{**fit, "converged": False} for fit in result]
        return result

    monkeypatch.setattr(fitting, "alternating_fit", force_one_failure)
    fitted.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    result = estimate_random_hsq(
        fitted,
        exclude_genes=[],
        n_jk_blocks=2,
    )

    assert result.converged
    assert result.converged_jk.tolist() == [False, True]
    assert np.isfinite(result.hsq_se)
    assert np.isfinite(result.variance_se).all()
    assert np.isfinite(result.kernel_coef_se).all()


def test_single_constant_annotation_reduces_to_the_plain_moment_solve(fitted):
    """With nothing to alternate over, the gene-coefficient fit is a no-op."""
    fitted.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    result = estimate_random_hsq(fitted, exclude_genes=[], n_jk_blocks=2)
    assert result.n_iter == 1
    assert result.converged

    suffstats = fitted.suffstats
    theta, _ = moment_solve(
        suffstats.pair_trace().sum(axis=0),
        suffstats.trace.sum(axis=0),
        suffstats.quadform.sum(axis=0),
        fitted.dataset.sample_size,
    )
    np.testing.assert_allclose(result.kernel_coef, theta, rtol=1e-9)


def test_gene_covariates_are_fitted_and_reported(test_dataset, tmp_path):
    model = Model(
        test_dataset, real_options(gene_annot_cols=[f"oebin{i}" for i in range(1, 6)])
    )
    model.compute_suffstats()
    model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    result = estimate_random_hsq(model, exclude_genes=[], n_jk_blocks=2)

    assert result.converged, f"did not converge in {result.n_iter} iterations"
    assert list(result.gene_annot_cols) == [f"oebin{i}" for i in range(1, 6)]
    # Scale is fixed by convention so theta carries the magnitude. The fitted
    # annotation coefficients have arithmetic mean one.
    coefficients = result.gene_coef
    assert coefficients.mean() == pytest.approx(1.0)
    assert result.n_genes_per_annot["n_genes"].sum() == result.n_genes
    result.save(tmp_path / "fit.npz")
    loaded = RandomFit.load(tmp_path / "fit.npz")
    assert_frame_equal(loaded.n_genes_per_annot, result.n_genes_per_annot)
    assert loaded.excluded_genes == result.excluded_genes
    # The fit is not the constant one: gene covariates change the answer.
    plain = Model(test_dataset, real_options())
    plain.compute_suffstats()
    plain.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    assert result.hsq != pytest.approx(
        estimate_random_hsq(plain, exclude_genes=[], n_jk_blocks=2).hsq
    )


def test_gene_annotation_order_does_not_change_the_fit(test_dataset):
    """Reordering mutually exclusive annotations only reorders coefficients."""
    columns = [f"oebin{i}" for i in range(1, 6)]

    def fit(order):
        model = Model(test_dataset, real_options(gene_annot_cols=order))
        model.compute_suffstats()
        model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=1)
        return model, estimate_random_hsq(model, exclude_genes=[], n_jk_blocks=1)

    baseline_model, baseline = fit(columns)
    baseline_scale = baseline_model.gene_annot_matrix() @ baseline.gene_coef
    reordered_model, reordered = fit(columns[::-1])
    reordered_scale = reordered_model.gene_annot_matrix() @ reordered.gene_coef

    assert reordered.converged
    assert reordered.hsq == pytest.approx(baseline.hsq, rel=1e-10, abs=1e-12)
    np.testing.assert_allclose(reordered_scale, baseline_scale, rtol=1e-10)


def gene_scaled_totals_oracle(
    suffstats, gene_annot, gene_coef, heritable, overdispersion
):
    """Independent dense oracle for the gene-scaled aggregate.

    A plain Python loop over genes: heritable kernels are scaled by that gene's
    coefficient, overdispersion kernels are not. Written from the model rather
    than from the einsum contractions under test, so it can disagree with them.
    """
    n_kernels = suffstats.trace.shape[1]
    pair_trace = np.zeros((n_kernels, n_kernels))
    trace = np.zeros(n_kernels)
    quadform = np.zeros(n_kernels)
    dense = suffstats.pair_trace()

    for position in range(len(suffstats.genes)):
        scale = float(gene_annot[position] @ gene_coef)
        weight = np.ones(n_kernels)
        weight[heritable] = scale
        pair_trace += dense[position] * np.outer(weight, weight)
        trace += suffstats.trace[position] * weight
        quadform += suffstats.quadform[position] * weight
    return pair_trace, trace, quadform


def test_gene_scaled_aggregate_matches_a_dense_per_gene_loop(test_dataset):
    """The second half of the alternating fit, against an independent oracle.

    Covers the two things that must hold and are easy to get wrong: heritable
    kernels carry two powers of the gene coefficient in the pair trace and one in
    the trace, and overdispersion kernels carry none at all — which is what makes
    the identity and overdispersion kernels distinct system columns.
    """
    from kasm.fitting import grouped_scaled_totals

    annot_cols = [f"oebin{i}" for i in range(1, 6)]
    model = Model(test_dataset, real_options(gene_annot_cols=annot_cols))
    model.compute_suffstats()
    suffstats = model.suffstats
    heritable = np.array(model.layout.heritable_indices, dtype=np.intp)
    overdispersion = np.array(model.layout.overdispersion_indices, dtype=np.intp)
    gene_annot = model.gene_annot_matrix()

    totals = model.compute_jk_suffstats(
        exclude_genes=[],
        n_jk_blocks=3,
    ).totals
    rng = np.random.default_rng(0)
    for gene_coef in (
        np.ones(5),
        rng.normal(size=5),
        np.array([2.0, -1.0, 0.5, 0.0, 3.0]),
    ):
        groups = np.full(model.n_kernels, -1, dtype=np.intp)
        groups[heritable] = 0
        got = grouped_scaled_totals(totals, gene_coef[:, None], groups, overdispersion)
        expected = gene_scaled_totals_oracle(
            suffstats, gene_annot, gene_coef, heritable, overdispersion
        )
        for actual, wanted in zip(got, expected):
            np.testing.assert_allclose(actual, wanted, rtol=1e-9, atol=1e-20)


def test_gene_coefficient_system_is_symmetric(test_dataset):
    """M5: its left-hand side is a Gram matrix, so asymmetry is a placement bug.

    The blocks coupling the gene-annotation components to the overdispersion
    components are each other's transpose; getting one wrong changes the solved
    gene coefficients while leaving every other assertion intact.
    """
    from kasm.fitting import RandomEffectsInput, gene_coef_system

    annot_cols = [f"oebin{i}" for i in range(1, 6)]
    model = Model(test_dataset, real_options(gene_annot_cols=annot_cols))
    model.compute_suffstats()
    heritable = np.array(model.layout.heritable_indices, dtype=np.intp)
    overdispersion = np.array(model.layout.overdispersion_indices, dtype=np.intp)
    totals = model.compute_jk_suffstats(
        exclude_genes=[],
        n_jk_blocks=3,
    ).totals

    rng = np.random.default_rng(2)
    groups = np.full(model.n_kernels, -1, dtype=np.intp)
    groups[heritable] = 0
    item = RandomEffectsInput(
        totals,
        groups,
        overdispersion,
        model.dataset.sample_size,
        np.arange(model.n_kernels),
    )
    for theta in (np.ones(model.n_kernels), rng.normal(size=model.n_kernels)):
        pair_trace, _, _, constrained_groups = gene_coef_system(item, theta)
        n_annot = len(constrained_groups[0])
        np.testing.assert_allclose(pair_trace, pair_trace.T, rtol=1e-12, atol=1e-30)
        # And the coupling blocks are genuinely non-zero, so symmetry is a test.
        assert np.abs(pair_trace[:n_annot, n_annot:]).max() > 0


def test_overdispersion_is_never_gene_scaled(test_dataset):
    """Overdispersion remains unscaled so the fitted system is identifiable."""
    from kasm.fitting import grouped_scaled_totals

    annot_cols = [f"oebin{i}" for i in range(1, 6)]
    model = Model(test_dataset, real_options(gene_annot_cols=annot_cols))
    model.compute_suffstats()
    heritable = np.array(model.layout.heritable_indices, dtype=np.intp)
    overdispersion = np.array(model.layout.overdispersion_indices, dtype=np.intp)
    totals = model.compute_jk_suffstats(
        exclude_genes=[],
        n_jk_blocks=3,
    ).totals
    groups = np.full(model.n_kernels, -1, dtype=np.intp)
    groups[heritable] = 0
    first = grouped_scaled_totals(totals, np.ones((5, 1)), groups, overdispersion)
    second = grouped_scaled_totals(totals, np.full((5, 1), 3.0), groups, overdispersion)
    block = np.ix_(overdispersion, overdispersion)
    # Tripling every gene coefficient must leave the overdispersion block alone.
    np.testing.assert_allclose(first[0][block], second[0][block], rtol=1e-12)
    np.testing.assert_allclose(
        first[1][overdispersion], second[1][overdispersion], rtol=1e-12
    )
    # And must scale the heritable trace by exactly three.
    np.testing.assert_allclose(
        3.0 * first[1][heritable], second[1][heritable], rtol=1e-12
    )


# ---------------------------------------------------------------------------
# Fitted model portability
# ---------------------------------------------------------------------------


def test_random_fit_round_trips_arrays_and_diagnostics(fitted, tmp_path):
    fitted.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    result = estimate_random_hsq(fitted, exclude_genes=[], n_jk_blocks=2)
    assert result.gene_coef_jk.shape == (2, *result.gene_coef.shape)
    components = result.components
    path = tmp_path / "fit.npz"
    result.save(path)
    loaded = RandomFit.load(path)
    assert "maf_components" not in result.__dict__
    assert "maf_components" not in loaded.__dict__
    assert_frame_equal(loaded.components, components)
    assert_frame_equal(loaded.maf_components, result.maf_components)

    for name in (
        "kernel_coef",
        "gene_coef",
        "kernel_coef_jk",
        "gene_coef_jk",
        "variance_jk",
        "hsq_jk",
        "converged_jk",
        "n_iter_jk",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(result, name))
    assert loaded.residual == pytest.approx(result.residual)
    assert loaded.kernel_options == result.kernel_options
    assert loaded.hsq == pytest.approx(result.hsq)
    assert loaded.converged == result.converged


def test_score_variants_is_aligned_and_reuses_the_fit(fitted, test_dataset, tmp_path):
    fitted.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    result = estimate_random_hsq(fitted, exclude_genes=[], n_jk_blocks=2)
    scores = result.score_variants(test_dataset)

    assert scores.select("variant", "gene").equals(
        test_dataset.variant_df().select("variant", "gene")
    )
    assert scores.columns == ["variant", "gene", "effsq", "hsq"]
    assert np.isfinite(scores.select("effsq", "hsq").to_numpy()).all()
    # A saved fit scores identically, which is the point of saving it.
    path = tmp_path / "fit.npz"
    result.save(path)
    assert_frame_equal(RandomFit.load(path).score_variants(test_dataset), scores)


def test_score_variants_does_not_depend_on_declared_bin_order(fitted, test_dataset):
    """Regression: the fit's coefficients must be paired by label, not position.

    A re-derived configuration need not enumerate MAF bins in the order the fit
    used. A positional lookup would then multiply every column by the wrong
    coefficient.
    """
    fitted.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    result = estimate_random_hsq(fitted, exclude_genes=[], n_jk_blocks=2)
    scores = result.score_variants(test_dataset)

    reordered = dict(reversed(list(MAF_BINS.items())))
    assert list(reordered) != list(MAF_BINS)
    shuffled = RandomFit(
        kernel_coef=result.kernel_coef,
        gene_coef=result.gene_coef,
        kernel_groups=result.kernel_groups,
        residual=result.residual,
        sample_size=result.sample_size,
        kernel_labels=result.kernel_labels,
        gene_annot_cols=result.gene_annot_cols,
        heritable_labels=result.heritable_labels,
        kernel_options={
            **result.kernel_options,
            "maf_bins": {name: list(bounds) for name, bounds in reordered.items()},
        },
    )
    assert_frame_equal(shuffled.score_variants(test_dataset), scores)


def test_include_genes_generator_preserves_unknown_gene_validation(fitted):
    include = (gene for gene in [fitted.dataset.genes[0], "NOT_A_GENE"])
    with pytest.raises(KeyError, match="not in this model"):
        estimate_fixed_hsq(fitted, include_genes=include)


@pytest.mark.parametrize("method", ["saturated", "burden_full"])
def test_unknown_genes_and_methods_are_rejected(fitted, method):
    with pytest.raises(KeyError, match="not in this model"):
        estimate_fixed_hsq(fitted, include_genes=["NOT_A_GENE"])
    with pytest.raises(ValueError, match="unknown method"):
        estimate_fixed_hsq(
            fitted, include_genes=fitted.dataset.genes[:1], method=method
        )


def test_burden_1d_method_needs_projection_weights(test_dataset):
    model = Model(test_dataset, real_options())
    model.compute_suffstats()
    with pytest.raises(ValueError, match="needs the burden projection weights"):
        estimate_fixed_hsq(
            model, include_genes=test_dataset.genes[:1], method="burden_1d"
        )


def test_fixed_burden_1d_accepts_explicit_projection_weights(fitted):
    genes = fitted.dataset.genes[:4]
    expected = estimate_fixed_hsq(fitted, include_genes=genes, method="burden_1d")
    fresh = Model(fitted.dataset, fitted.options)
    fresh.compute_suffstats()

    observed = estimate_fixed_hsq(
        fresh,
        include_genes=genes,
        method="burden_1d",
        burden_1d_wgts=fitted.burden_1d_wgts,
    )

    assert observed.hsq == pytest.approx(expected.hsq)
    np.testing.assert_allclose(
        observed.per_gene["hsq"],
        expected.per_gene["hsq"],
    )


def test_explicit_burden_1d_wgts_must_cover_labels(fitted):
    with pytest.raises(ValueError, match="labels mismatch"):
        estimate_fixed_hsq(
            fitted,
            include_genes=fitted.dataset.genes[:1],
            method="burden_1d",
            burden_1d_wgts=dict(list(fitted.burden_1d_wgts.items())[:-1]),
        )


def test_saturated_fixed_effects_do_not_require_sufficient_statistics(
    test_dataset,
):
    model = Model(test_dataset, real_options())

    result = estimate_fixed_hsq(
        model,
        include_genes=test_dataset.genes[:2],
        method="indep",
    )

    assert result.n_genes == 2
    assert result.method == "indep"


def test_fixed_burden_1d_projects_each_partition_with_shared_weights(
    synthetic_dataset,
    monkeypatch,
):
    from scipy.linalg import pinvh

    from kasm.ldm import LDM

    model = Model(
        synthetic_dataset,
        KernelOptions(
            burden_cols=["constant", "score"],
            burden_by_maf=False,
            within_gene_group_col="exon",
        ),
    )
    model.compute_suffstats()
    burden_1d_wgts = dict(zip(model.layout.burden_labels, [0.75, -0.25]))
    gene = "GENE_A"
    rows = synthetic_dataset.gene_slice(gene)
    projected = model.burden_wgts(gene) @ np.array(list(burden_1d_wgts.values()))
    codes = model.burden_partition_codes(gene)
    design = np.column_stack([projected * (codes == code) for code in np.unique(codes)])
    residual = (
        synthetic_dataset.load_ldm(gene).scale(model.variant_scale(gene)).toarray()
    )
    covariance = design.T @ residual @ design
    inverse, rank = pinvh(covariance, return_rank=True, rtol=1e-6)
    beta_tilde = synthetic_dataset.beta[rows] / (
        synthetic_dataset.se[rows] ** 2 * synthetic_dataset.sample_size
    )
    score = design.T @ beta_tilde

    def reject_dense(_self):
        raise AssertionError("partitioned fixed effects must not densify LD")

    monkeypatch.setattr(LDM, "toarray", reject_dense)
    result = estimate_fixed_hsq(
        model,
        include_genes=[gene],
        method="burden_1d",
        burden_1d_wgts=burden_1d_wgts,
    )

    row = result.per_gene.filter(pl.col("gene") == gene).row(0, named=True)
    assert row["rank"] == rank
    assert row["quadform"] == pytest.approx(score @ inverse @ score)


def test_fixed_full_burden_jointly_fits_each_partition(synthetic_dataset):
    from scipy.linalg import pinvh

    model = Model(
        synthetic_dataset,
        KernelOptions(
            burden_cols=["constant", "score"],
            burden_by_maf=False,
            within_gene_group_col="exon",
        ),
    )
    model.compute_suffstats()
    gene = "GENE_A"

    result = estimate_fixed_hsq(model, include_genes=[gene], method="burden")

    rows = synthetic_dataset.gene_slice(gene)
    burden_wgts = model.burden_wgts(gene)
    codes = model.burden_partition_codes(gene)
    design = np.column_stack(
        [
            burden_wgts[:, burden] * (codes == code)
            for burden in range(burden_wgts.shape[1])
            for code in np.unique(codes)
        ]
    )
    residual = (
        synthetic_dataset.load_ldm(gene).scale(model.variant_scale(gene)).toarray()
    )
    covariance = design.T @ residual @ design
    inverse, rank = pinvh(covariance, return_rank=True, rtol=1e-6)
    beta_tilde = synthetic_dataset.beta[rows] / (
        synthetic_dataset.se[rows] ** 2 * synthetic_dataset.sample_size
    )
    score = design.T @ beta_tilde

    row = result.per_gene.row(0, named=True)
    assert row["rank"] == rank
    assert row["quadform"] == pytest.approx(score @ inverse @ score)


@pytest.mark.parametrize("method", ["burden_1d", "burden", "indep"])
def test_fixed_hsq_is_the_sum_of_debiased_per_gene_terms(fitted, method):
    genes = fitted.dataset.genes[:4]
    result = estimate_fixed_hsq(fitted, include_genes=genes, method=method)
    assert result.n_genes == len(genes)
    assert result.per_gene["gene"].to_list() == genes
    assert result.hsq == pytest.approx(result.per_gene["hsq"].sum())
    sample_size = fitted.dataset.sample_size
    expected = (sample_size * result.per_gene["quadform"] - result.per_gene["rank"]) / (
        sample_size - result.per_gene["rank"]
    )
    np.testing.assert_allclose(result.per_gene["hsq"], expected)
    assert result.hsq_se == pytest.approx(
        np.sqrt(2 * result.total_rank + 4 * sample_size * max(0.0, result.hsq))
        / sample_size
    )


@pytest.mark.parametrize("method", ["burden_1d", "burden", "indep"])
def test_fixed_maf_decomposition_closes_exactly_for_full_correlated_fit(
    fitted,
    method,
):
    genes = fitted.dataset.genes[:4]
    full = estimate_fixed_hsq(fitted, include_genes=genes, method=method)
    decomposed = full.maf_components.group_by("maf_bin", maintain_order=True).agg(
        pl.col("hsq").sum()
    )

    assert decomposed["maf_bin"].to_list() == list(MAF_BINS)
    assert decomposed["hsq"].sum() == pytest.approx(full.hsq, rel=1e-11, abs=1e-13)


def test_fixed_maf_decomposition_requires_configured_exhaustive_partition(
    test_dataset,
):
    genes = test_dataset.genes[:1]
    model = Model(test_dataset, KernelOptions(burden_cols=["constant"]))
    fit = fit_fixed_effects(model, include_genes=genes, method="indep")
    with pytest.raises(ValueError, match="requires maf_col and maf_bins"):
        _ = fit.maf_components

    model = Model(
        test_dataset,
        real_options(maf_bins={"incomplete": (0.0, 1e-4)}),
    )
    model.compute_suffstats()
    fit = fit_fixed_effects(model, include_genes=genes, method="indep")
    with pytest.raises(ValueError, match="exhaustive, disjoint partition"):
        _ = fit.maf_components


def test_burden_1d_method_uses_one_degree_of_freedom_per_gene(fitted):
    genes = fitted.dataset.genes[:5]
    burden_1d = estimate_fixed_hsq(fitted, include_genes=genes, method="burden_1d")
    assert burden_1d.total_rank <= len(genes)
    indep = estimate_fixed_hsq(fitted, include_genes=genes, method="indep")
    # The saturated fit spends as many degrees of freedom as the LD has rank.
    assert indep.total_rank > burden_1d.total_rank
    assert indep.hsq > burden_1d.hsq


def test_burden_1d_method_matches_the_projection_by_hand(fitted):
    genes = fitted.dataset.genes[:3]
    result = estimate_fixed_hsq(fitted, include_genes=genes, method="burden_1d")
    burden_1d_wgts = np.array(list(fitted.burden_1d_wgts.values()))
    for gene in genes:
        position = fitted.suffstats.genes.index(gene)
        projected = burden_1d_wgts @ fitted.suffstats.burden_beta[position]
        variance = burden_1d_wgts @ fitted.suffstats.burden_cov(gene) @ burden_1d_wgts
        observed = result.per_gene.filter(pl.col("gene") == gene)["quadform"][0]
        assert observed == pytest.approx(projected**2 / variance)


def test_default_burden_method_matches_full_design_by_hand(fitted):
    from scipy.linalg import pinvh

    genes = fitted.dataset.genes[:3]
    result = estimate_fixed_hsq(fitted, include_genes=genes)
    assert result.method == "burden"
    for gene in genes:
        position = fitted.suffstats.genes.index(gene)
        score = fitted.suffstats.burden_beta[position]
        covariance = fitted.suffstats.burden_cov(gene)
        inverse, rank = pinvh(covariance, return_rank=True, rtol=1e-6)
        row = result.per_gene.filter(pl.col("gene") == gene).row(0, named=True)
        assert row["rank"] == rank
        assert row["quadform"] == pytest.approx(score @ inverse @ score)


def test_indep_method_matches_a_dense_pseudo_inverse(fitted):
    """Independent oracle for the saturated quadratic form."""
    from scipy.linalg import pinvh

    gene = fitted.dataset.genes[0]
    result = estimate_fixed_hsq(fitted, include_genes=[gene], method="indep")

    dataset = fitted.dataset
    rows = dataset.gene_slice(gene)
    residual = dataset.load_ldm(gene).scale(fitted.variant_scale(gene)).toarray()
    inverse, rank = pinvh(residual, return_rank=True, rtol=1e-6)
    beta_tilde = dataset.beta[rows] / (dataset.se[rows] ** 2 * dataset.sample_size)

    row = result.per_gene.filter(pl.col("gene") == gene).row(0, named=True)
    assert row["rank"] == rank
    assert row["quadform"] == pytest.approx(beta_tilde @ inverse @ beta_tilde)


def test_fixed_estimator_ignores_gene_order(fitted):
    genes = list(fitted.dataset.genes[:4])
    forward = estimate_fixed_hsq(fitted, include_genes=genes)
    reverse = estimate_fixed_hsq(fitted, include_genes=genes[::-1])
    assert forward.method == "burden"
    assert forward.hsq == pytest.approx(reverse.hsq)
    assert forward.per_gene["gene"].to_list() == reverse.per_gene["gene"].to_list()
