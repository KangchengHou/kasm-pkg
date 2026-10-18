"""LD-aware summary-statistic simulation."""

from __future__ import annotations


import numpy as np
import pytest

import kasm
from kasm.simulation import mvn_sample


def random_fit(model, coefficients=None, init_gene_coef=None, groups=None):
    """A fit over *model*'s own kernels, optionally with several kernel groups.

    ``init_gene_coef`` is one mapping per group, and ``groups`` names the
    group of each kernel label; both default to a single shared group.
    """
    labels = tuple(kernel_id.label for kernel_id in model.layout.kernel_ids)
    kernel_coef = np.zeros(len(labels))
    for label, value in (coefficients or {}).items():
        kernel_coef[labels.index(label)] = value
    names = list(model.options.gene_annot_cols or ["constant"])
    per_kernel_group = init_gene_coef or [{}]
    if isinstance(per_kernel_group, dict):
        per_kernel_group = [per_kernel_group]
    gene_coef = np.ones((len(names), len(per_kernel_group)))
    for column, mapping in enumerate(per_kernel_group):
        for label, value in mapping.items():
            gene_coef[names.index(label), column] = value
    heritable = {labels[i] for i in model.layout.heritable_indices}
    kernel_groups = [
        (groups or {}).get(label, 0) if label in heritable else -1 for label in labels
    ]
    return kasm.RandomFit(
        kernel_coef=kernel_coef,
        gene_coef=gene_coef,
        kernel_groups=kernel_groups,
        residual=1.0,
        sample_size=model.dataset.sample_size,
        kernel_labels=labels,
        gene_annot_cols=tuple(model.options.gene_annot_cols or ()),
        heritable_labels=tuple(sorted(heritable)),
        kernel_options=model.options.to_dict(),
    )


def burden_model(dataset, **overrides):
    options = dict(
        maf_col="af",
        burden_cols=["constant"],
        burden_by_maf=False,
        variant_variance_from_maf=True,
    )
    return kasm.Model(dataset, kasm.KernelOptions(**(options | overrides)))


def test_negative_eigenvalues_are_clamped_rather_than_rejected():
    np.random.seed(5)
    draws = mvn_sample(np.diag([3.0, -2e-8]), 20000)

    np.testing.assert_allclose(np.cov(draws), np.diag([3.0, 0.0]), atol=0.06)


def test_noise_only_summary_statistics_are_reproducible_on_the_maf_scale(
    synthetic_dataset_no_ld,
):
    model = burden_model(synthetic_dataset_no_ld)
    causal = np.zeros(model.dataset.n_variants)

    np.random.seed(17)
    first, se = kasm.simulate_sumstats(model, causal)
    np.random.seed(17)
    second, _ = kasm.simulate_sumstats(model, causal)

    maf = model.dataset.variant_values("af")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(
        se,
        1.0 / np.sqrt(model.dataset.sample_size * 2.0 * maf * (1.0 - maf)),
        rtol=1e-15,
    )
    assert first.shape == (model.dataset.n_variants,)
    assert np.any(first != 0)


def test_null_summary_statistics_have_the_aligned_ld_covariance(
    synthetic_dataset,
):
    model = burden_model(synthetic_dataset)
    np.random.seed(29)
    draws, _ = kasm.simulate_sumstats(model, np.zeros((model.dataset.n_variants, 4000)))

    gene = "GENE_A"
    rows = model.dataset.gene_slice(gene)
    se = 1.0 / (np.sqrt(model.dataset.sample_size) * model.variant_scale(gene))

    np.testing.assert_allclose(
        np.cov(draws[rows] / se[:, None]),
        model.dataset.load_ldm(gene).toarray(),
        atol=0.06,
    )


def test_causal_burden_effects_follow_the_fitted_kernel_covariance(
    synthetic_dataset_no_ld,
):
    model = burden_model(synthetic_dataset_no_ld)
    variance = 0.25
    np.random.seed(41)
    beta = kasm.simulate_burden_effects(
        model, random_fit(model, {"B:constant": variance}), n_sim=4000
    )

    rows = model.dataset.gene_slice("GENE_A")
    np.testing.assert_allclose(
        np.cov(beta[rows]),
        np.full((rows.stop - rows.start,) * 2, variance),
        atol=0.015,
    )


def test_partitioned_burden_effects_are_independent_between_partitions(
    synthetic_dataset_no_ld,
):
    model = burden_model(synthetic_dataset_no_ld, within_gene_group_col="exon")
    variance = 0.2
    np.random.seed(39)
    beta = kasm.simulate_burden_effects(
        model, random_fit(model, {"B:constant": variance}), n_sim=4000
    )

    gene = "GENE_C"
    rows = model.dataset.gene_slice(gene)
    partition = model.burden_partition_codes(gene)
    np.testing.assert_allclose(
        np.cov(beta[rows], bias=True),
        variance * (partition[:, None] == partition[None, :]),
        rtol=0.1,
        atol=0.015,
    )


def test_marginal_effects_propagate_causal_effects_through_ld(
    synthetic_dataset,
):
    model = burden_model(synthetic_dataset)
    variance = 0.25
    np.random.seed(53)
    beta = kasm.simulate_burden_effects(
        model, random_fit(model, {"B:constant": variance}), n_sim=5000
    )
    marginal, _ = kasm.simulate_sumstats(model, beta)

    rows = model.dataset.gene_slice("GENE_A")
    observed = marginal[rows] @ beta[rows].T / beta.shape[1]
    scale = model.variant_scale("GENE_A")
    propagation = model.dataset.load_ldm("GENE_A").toarray() * (
        scale[None, :] / scale[:, None]
    )

    np.testing.assert_allclose(
        observed,
        propagation @ np.full((len(scale), len(scale)), variance),
        atol=0.025,
    )


def test_cross_burden_coefficient_enters_the_causal_covariance(
    synthetic_dataset_no_ld,
):
    model = burden_model(synthetic_dataset_no_ld, burden_cols=["constant", "score"])
    np.random.seed(71)
    beta = kasm.simulate_burden_effects(
        model,
        random_fit(
            model,
            {"B:constant": 0.3, "B:score": 0.2, "B:constant*score": 0.1},
        ),
        n_sim=5000,
    )

    rows = model.dataset.gene_slice("GENE_A")
    constant, score = model.burden_wgts("GENE_A").T
    expected = (
        0.3 * np.outer(constant, constant)
        + 0.2 * np.outer(score, score)
        + 0.1 * (np.outer(constant, score) + np.outer(score, constant))
    )
    np.testing.assert_allclose(np.cov(beta[rows]), expected, atol=0.03)


def test_gene_scale_rescales_each_genes_draw(
    synthetic_dataset_no_ld,
):
    model = burden_model(synthetic_dataset_no_ld)
    fit = random_fit(model, {"B:constant": 0.2})
    scale = np.ones(model.dataset.n_genes)
    scale[0] = 50.0

    np.random.seed(37)
    flat = kasm.simulate_burden_effects(model, fit, n_sim=200)
    np.random.seed(37)
    sparse = kasm.simulate_burden_effects(model, fit, n_sim=200, gene_scale=scale)

    first = model.dataset.gene_slice(model.dataset.genes[0])
    np.testing.assert_allclose(sparse[first], np.sqrt(50.0) * flat[first])
    np.testing.assert_array_equal(sparse[first.stop :], flat[first.stop :])


def test_init_gene_coef_scale_each_genes_causal_covariance(
    synthetic_dataset_no_ld,
):
    model = burden_model(
        synthetic_dataset_no_ld,
        gene_annot_cols=["constraint_low", "constraint_high"],
    )
    fit = random_fit(
        model,
        {"B:constant": 0.2},
        {"constraint_low": 0.5, "constraint_high": 2.0},
    )
    annot = model.gene_annot_matrix()
    low = model.dataset.genes[int(np.flatnonzero(annot[:, 0])[0])]
    high = model.dataset.genes[int(np.flatnonzero(annot[:, 1])[0])]

    np.random.seed(97)
    beta = kasm.simulate_burden_effects(model, fit, n_sim=4000)

    observed = {
        gene: np.var(beta[model.dataset.gene_slice(gene).start], ddof=1)
        for gene in (low, high)
    }
    assert observed[low] == pytest.approx(0.2 * 0.5, abs=0.01)
    assert observed[high] == pytest.approx(0.2 * 2.0, abs=0.025)


def test_grouped_init_gene_coef_scale_each_kernel_group_separately(
    synthetic_dataset_no_ld,
):
    """Missense and pLoF kernels may scale differently with gene constraint."""
    model = burden_model(
        synthetic_dataset_no_ld,
        burden_cols=["constant", "score"],
        gene_annot_cols=["constraint_low", "constraint_high"],
    )
    fit = random_fit(
        model,
        {"B:constant": 0.2, "B:score": 0.2},
        [
            {"constraint_low": 1.0, "constraint_high": 1.0},
            {"constraint_low": 0.0, "constraint_high": 3.0},
        ],
        groups={"B:score": 1, "B:constant*score": 1},
    )
    annot = model.gene_annot_matrix()
    low = model.dataset.genes[int(np.flatnonzero(annot[:, 0])[0])]
    high = model.dataset.genes[int(np.flatnonzero(annot[:, 1])[0])]

    np.random.seed(113)
    beta = kasm.simulate_burden_effects(model, fit, n_sim=6000)

    for gene, score_scale in ((low, 0.0), (high, 3.0)):
        rows = model.dataset.gene_slice(gene)
        score = model.burden_wgts(gene)[:, 1]
        np.testing.assert_allclose(
            beta[rows].var(axis=1, ddof=1),
            0.2 + 0.2 * score_scale * score**2,
            atol=0.03,
        )


def test_simulation_transports_per_allele_coefficients_from_se_scale(
    synthetic_dataset_no_ld,
):
    options = dict(maf_col="af", burden_cols=["constant"], burden_by_maf=False)
    fitted_on_se = kasm.Model(synthetic_dataset_no_ld, kasm.KernelOptions(**options))
    model = burden_model(synthetic_dataset_no_ld)
    assert not np.allclose(fitted_on_se.variant_scale(), model.variant_scale())

    np.random.seed(83)
    transported = kasm.simulate_burden_effects(
        model, random_fit(fitted_on_se, {"B:constant": 0.2})
    )
    np.random.seed(83)
    native = kasm.simulate_burden_effects(model, random_fit(model, {"B:constant": 0.2}))

    np.testing.assert_array_equal(transported, native)


def test_simulation_rejects_same_maf_label_with_different_bounds(
    synthetic_dataset_no_ld,
):
    source = kasm.Model(
        synthetic_dataset_no_ld,
        kasm.KernelOptions(
            maf_col="af",
            maf_bins={"rare": (0, 0.2)},
            burden_cols=["constant"],
            burden_by_maf=True,
        ),
    )
    target = burden_model(
        synthetic_dataset_no_ld,
        maf_bins={"rare": (0, 0.4)},
        burden_by_maf=True,
    )

    with pytest.raises(ValueError, match="kernel definitions"):
        kasm.simulate_burden_effects(target, random_fit(source))


def test_a_negative_fitted_variance_is_truncated_to_zero(
    synthetic_dataset_no_ld,
):
    model = burden_model(synthetic_dataset_no_ld)

    np.random.seed(89)
    beta = kasm.simulate_burden_effects(
        model, random_fit(model, {"B:constant": -0.1}), n_sim=4
    )

    np.testing.assert_array_equal(beta, np.zeros_like(beta))
