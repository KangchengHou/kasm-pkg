"""Extension statistics and the two enrichment estimators."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from extension_helpers import extension_designs

import numpy as np
import polars as pl
import pytest
from conftest import MAF_BINS

from kasm import (
    Dataset,
    JKSuffStats,
    KernelOptions,
    Model,
    fit_fixed_effects,
    fit_random_effects,
)
from kasm.enrichment import (
    EnrichResult,
    _annotation_heterozygosity,
    _nested_burden_designs,
    _random_enrich_result,
    _validate_burden_annotation,
    pool_enrichment,
)

ANNOTATION = "netsurfp.rsa_G1"


def real_options(**overrides) -> KernelOptions:
    settings = {
        "maf_col": "AF",
        "maf_bins": MAF_BINS,
        "burden_cols": ["constant", "am.pathogenicity"],
        "burden_by_maf": True,
        "overdispersion_cols": ["constant"],
        "overdispersion_by_maf": True,
    }
    settings.update(overrides)
    return KernelOptions(**settings)


@pytest.fixture
def fitted(test_dataset):
    model = Model(test_dataset, real_options())
    model.compute_suffstats(extra_enrich_cols=[ANNOTATION])
    return model


@pytest.fixture
def significant(fitted):
    table = fitted.compute_gene_pvalue()
    genes = table.filter(pl.col("pvalue") < 0.05 / len(table))["gene"].to_list()
    assert genes, "fixture should contain significant genes"
    return genes


# ---------------------------------------------------------------------------
# The extended design
# ---------------------------------------------------------------------------


def test_interaction_design_is_the_annotated_subset(fitted):
    """An interaction column is the base column zeroed off the annotation."""
    extended = fitted._with_burden_column(ANNOTATION)
    burden_wgts = extended.burden_wgts()
    annotation = extended.dataset.variant_values(ANNOTATION)
    n_base = fitted.layout.n_burden
    for position, maf in enumerate(extended._maf_masks.values()):
        np.testing.assert_array_equal(
            burden_wgts[:, n_base + position], annotation * maf
        )


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_constant_annotations_are_accepted(fitted, value):
    frame = fitted.dataset.variant_df().clone()
    frame = frame.with_columns(pl.lit(value).alias("degenerate"))
    model = Model(
        Dataset(
            frame,
            gene_df=fitted.dataset.gene_df(),
            beta_col="BETA",
            se_col="SE",
            sample_size=fitted.dataset.sample_size,
            ldm=fitted.dataset.ldm_source,
        ),
        real_options(),
    )

    _validate_burden_annotation(model, "degenerate")


def test_mapping_enrichment_requires_a_single_model(fitted, significant):
    parent = fitted.compute_jk_suffstats(exclude_genes=significant, n_jk_blocks=2)
    with pytest.raises(ValueError, match="extra_enrich_col requires a single model"):
        fit_random_effects({"cell": parent}, extra_enrich_col=ANNOTATION)


# ---------------------------------------------------------------------------
# The proportion
# ---------------------------------------------------------------------------


def test_proportion_is_the_heterozygosity_share(fitted):
    """Burden traces weight variants by d^2, so the null share is the d^2 share."""
    annot_het, total_het = _annotation_heterozygosity(fitted, ANNOTATION)
    proportion = annot_het / total_het
    annotation = fitted.dataset.variant_values(ANNOTATION)
    scale_squared = fitted.variant_scale() ** 2
    assert proportion == pytest.approx(
        (annotation * scale_squared).sum() / scale_squared.sum()
    )
    assert 0 < proportion < 1
    # It is NOT the variant-count share, which is what makes it a choice.
    assert proportion != pytest.approx(annotation.mean())


# ---------------------------------------------------------------------------
# Random enrichment
# ---------------------------------------------------------------------------


def test_random_enrichment_is_the_ratio_of_ratios(fitted, significant):
    from kasm.fitting import jackknife_se

    fitted.compute_jk_suffstats(exclude_genes=significant, n_jk_blocks=4)
    _, result = fit_random_effects(
        fitted,
        extra_enrich_col=ANNOTATION,
        exclude_genes=significant,
        n_jk_blocks=4,
    )
    assert result.annotation == ANNOTATION
    assert result.enrichment() == pytest.approx(
        (result.annot_hsq / result.annot_het) / (result.total_hsq / result.total_het)
    )
    assert result.enrichment_jk().shape == (4,)
    assert np.isfinite(result.enrichment_se())
    np.testing.assert_allclose(
        result.enrichment_se(), jackknife_se(result.enrichment_jk()[:, None])[0]
    )
    # A delta-method combination of the two marginal errors would differ.
    delta = (
        np.sqrt(
            (result.annot_hsq_se / result.total_hsq) ** 2
            + (result.annot_hsq * result.total_hsq_se / result.total_hsq**2) ** 2
        )
        * result.total_het
        / result.annot_het
    )
    assert result.enrichment_se() != pytest.approx(delta, rel=1e-6)


def test_random_enrichment_accepts_external_genomic_blocks(fitted, significant):
    blocks = np.arange(fitted.dataset.n_genes) % 4
    model = Model(fitted.dataset, fitted.options)
    model.compute_suffstats(extra_enrich_cols=[ANNOTATION])
    base = model.compute_jk_suffstats(
        exclude_genes=significant,
        n_jk_blocks=4,
        gene_blocks=blocks,
    )

    _, _, selected, attribution = next(
        extension_designs(base, extra_enrich_cols=[ANNOTATION])
    )
    fit = fit_random_effects(
        selected,
        exclude_genes=significant,
        n_jk_blocks=4,
    )

    result = _random_enrich_result(attribution, fit)
    np.testing.assert_array_equal(fit.gene_blocks, blocks)
    np.testing.assert_array_equal(fit.jk["block"], np.arange(4))
    assert result.annot_hsq_jk.shape == (4,)
    assert result.total_hsq_jk.shape == (4,)
    np.testing.assert_allclose(
        result.enrichment_jk(),
        (result.annot_hsq_jk / result.annot_het_jk)
        / (result.total_hsq_jk / result.total_het_jk),
    )
    np.testing.assert_allclose(
        result.annot_het_jk,
        attribution["annot_het"].sum() - attribution["annot_het"],
    )
    np.testing.assert_allclose(
        result.total_het_jk,
        attribution["total_het"].sum() - attribution["total_het"],
    )


def test_random_enrichment_honors_fixed_gene_coefficient(fitted, significant):
    fitted.compute_jk_suffstats(exclude_genes=significant, n_jk_blocks=2)
    fit, result = fit_random_effects(
        fitted,
        extra_enrich_col=ANNOTATION,
        exclude_genes=significant,
        fix_gene_coef=True,
        init_gene_coef={"constant": 2.0},
        n_jk_blocks=2,
    )

    np.testing.assert_array_equal(fit.gene_coef, [[2.0]])
    np.testing.assert_array_equal(fit.gene_coef_jk[:, :, 0], [[2.0], [2.0]])
    assert result.enrichment() == pytest.approx(
        (result.annot_hsq / result.annot_het) / (result.total_hsq / result.total_het)
    )
    np.testing.assert_allclose(
        result.enrichment_jk(),
        (result.annot_hsq_jk / result.annot_het_jk)
        / (result.total_hsq_jk / result.total_het_jk),
    )


def test_compact_random_extensions_support_within_gene_groups(
    synthetic,
    tmp_path: Path,
):
    dataset = Dataset(
        synthetic.variant_df,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=1000.0,
        ldm=synthetic.ldm_path,
    )
    options = KernelOptions(burden_cols=["constant"], within_gene_group_col="exon")
    excluded = [dataset.genes[1]]
    model = Model(dataset, options).compute_suffstats(extra_enrich_cols=["buried"])
    compact = model.compute_jk_suffstats(
        exclude_genes=excluded,
        n_jk_blocks=2,
    )
    direct = (
        (Model(dataset, options))
        ._with_burden_column("buried")
        .compute_suffstats()
        .compute_jk_suffstats(exclude_genes=excluded, n_jk_blocks=2)
    )
    _, _, selected, _ = next(extension_designs(compact, extra_enrich_cols=["buried"]))

    for key in direct.blocks:
        np.testing.assert_allclose(selected.blocks[key], direct.blocks[key])
    path = tmp_path / "partitioned-jackknife.npz"
    selected.save(path)
    loaded = JKSuffStats.load(path)
    for key in selected.blocks:
        np.testing.assert_array_equal(loaded.blocks[key], selected.blocks[key])


def _dense_diagonal_hsq(extended, theta, gene_coef, genes, enrichment_col):
    """Independent dense oracle for fitted covariance-diagonal attribution."""
    gene_annot = extended.gene_annot_matrix()
    gene_positions = {gene: i for i, gene in enumerate(extended.dataset.genes)}
    annotated = total = 0.0
    for gene in genes:
        burden_wgts = extended.burden_wgts(gene)
        covariance = np.zeros((len(burden_wgts), len(burden_wgts)))
        for index, column in zip(extended.layout.burden_indices, burden_wgts.T):
            covariance += theta[index] * np.outer(column, column)
        for index, (left, right) in zip(
            extended.layout.cross_indices, extended.layout.cross_pairs
        ):
            covariance += theta[index] * (
                np.outer(burden_wgts[:, left], burden_wgts[:, right])
                + np.outer(burden_wgts[:, right], burden_wgts[:, left])
            )
        if extended.layout.independent_index is not None:
            covariance += theta[extended.layout.independent_index] * np.eye(
                len(burden_wgts)
            )
        gene_scale = float(gene_annot[gene_positions[gene]] @ gene_coef)
        diagonal = gene_scale * extended.variant_scale(gene) ** 2 * np.diag(covariance)
        annotation = extended.dataset.variant_values(enrichment_col, gene)
        annotated += float(annotation @ diagonal)
        total += float(diagonal.sum())
    return annotated, total


def test_full_diagonal_attribution_dense_oracle(synthetic):
    """All burden, cross-burden, and independent diagonals are attributed."""
    dataset = Dataset(
        synthetic.variant_df,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=1000.0,
        ldm=synthetic.ldm_path,
    )
    model = Model(
        dataset,
        KernelOptions(burden_cols=["constant", "score"], independent_col="constant"),
    )
    extended = model._with_burden_column("buried")
    theta = np.linspace(
        0.03, 0.03 * extended.layout.n_kernels, extended.layout.n_kernels
    )
    # Ensure this oracle exercises nonzero cross coefficients.
    assert np.all(theta[list(extended.layout.cross_indices)] != 0)
    genes = extended.dataset.genes[:-1]
    model.compute_suffstats(extra_enrich_cols=["buried"])
    jk_stats = model.compute_jk_suffstats(
        exclude_genes=[model.genes[-1]],
        n_jk_blocks=1,
    )
    _, _, _, attribution = next(
        extension_designs(jk_stats, extra_enrich_cols=["buried"])
    )
    observed_annotated = float(
        np.einsum(
            "k,ka,a->",
            theta,
            attribution["annot_hsq_weights"].sum(axis=0),
            np.array([1.7]),
        )
    )
    observed_total = float(
        np.einsum(
            "k,ka,a->",
            theta,
            attribution["total_hsq_weights"].sum(axis=0),
            np.array([1.7]),
        )
    )

    expected_annotated, expected_total = _dense_diagonal_hsq(
        extended, theta, np.array([1.7]), genes, "buried"
    )

    assert observed_annotated == pytest.approx(expected_annotated)
    assert observed_total == pytest.approx(expected_total)
    d2 = np.concatenate([model.variant_scale(gene) ** 2 for gene in genes])
    annot = np.concatenate(
        [model.dataset.variant_values("buried", gene) for gene in genes]
    )
    annot_het, total_het = _annotation_heterozygosity(model, "buried", genes)
    assert annot_het / total_het == pytest.approx((annot * d2).sum() / d2.sum())


def test_batched_jk_attribution_matches_repeated_point_attribution(synthetic):
    dataset = Dataset(
        synthetic.variant_df,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=1000.0,
        ldm=synthetic.ldm_path,
    )
    extended = (
        Model(
            dataset,
            KernelOptions(
                burden_cols=["constant", "score"], independent_col="constant"
            ),
        )
    )._with_burden_column("buried")
    genes = list(extended.dataset.genes)
    blocks = np.arange(len(genes)) % 2
    live_blocks = np.array([0, 1])
    theta = np.linspace(
        0.01, 0.01 * extended.layout.n_kernels, extended.layout.n_kernels
    )
    theta_jk = np.vstack((theta * 0.7, theta * 1.3))
    gene_coef = np.array([1.4])
    gene_coef_jk = np.array([[0.8], [1.9]])

    from types import SimpleNamespace

    base = Model(
        dataset,
        KernelOptions(burden_cols=["constant", "score"], independent_col="constant"),
    )
    base.compute_suffstats(extra_enrich_cols=["buried"])
    jk = base.compute_jk_suffstats(
        exclude_genes=[],
        n_jk_blocks=2,
        gene_blocks=blocks,
    )
    _, _, _, attribution = next(extension_designs(jk, extra_enrich_cols=["buried"]))
    result = _random_enrich_result(
        attribution,
        SimpleNamespace(
            kernel_coef=theta,
            gene_coef=gene_coef[:, None],
            kernel_coef_jk=theta_jk,
            gene_coef_jk=gene_coef_jk[:, :, None],
            n_jk_blocks=len(live_blocks),
            n_genes=len(genes),
            n_variants=dataset.n_variants,
        ),
    )
    observed = (
        result.annot_hsq,
        result.total_hsq,
        result.annot_hsq_jk,
        result.total_hsq_jk,
    )
    expected_point = _dense_diagonal_hsq(extended, theta, gene_coef, genes, "buried")
    expected_jk = [
        _dense_diagonal_hsq(
            extended,
            theta_jk[position],
            gene_coef_jk[position],
            [gene for gene, gene_block in zip(genes, blocks) if gene_block != block],
            "buried",
        )
        for position, block in enumerate(live_blocks)
    ]
    assert observed[0] == pytest.approx(expected_point[0])
    assert observed[1] == pytest.approx(expected_point[1])
    np.testing.assert_allclose(observed[2], [value[0] for value in expected_jk])
    np.testing.assert_allclose(observed[3], [value[1] for value in expected_jk])


def test_unconverged_random_enrichment_replicate_retains_standard_errors(
    fitted, significant, monkeypatch
):
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
    fitted.compute_jk_suffstats(exclude_genes=significant, n_jk_blocks=2)
    fit, result = fit_random_effects(
        fitted,
        extra_enrich_col=ANNOTATION,
        exclude_genes=significant,
        n_jk_blocks=2,
    )

    assert not fit.converged_jk[0]
    assert np.isfinite(result.enrichment_se())
    assert np.isfinite(result.annot_hsq_se)
    assert np.isfinite(result.total_hsq_se)


# ---------------------------------------------------------------------------
# Fixed enrichment
# ---------------------------------------------------------------------------


def test_fixed_enrichment_does_not_compute_random_suffstats(
    fitted, significant, monkeypatch
):
    def fail(*args, **kwargs):
        raise AssertionError("fixed enrichment must not compute random suffstats")

    monkeypatch.setattr(Model, "compute_suffstats", fail)
    result = fit_fixed_effects(
        fitted, extra_enrich_col=ANNOTATION, include_genes=significant
    )[1]
    assert result.total_hsq > 0


def test_fixed_enrichment_loads_each_selected_gene_ldm_once(
    fitted, significant, monkeypatch
):
    original = fitted.dataset.load_ldm
    loaded = []

    def record(gene):
        loaded.append(gene)
        return original(gene)

    def fail(*args, **kwargs):
        raise AssertionError("fixed enrichment must not materialize a variant frame")

    monkeypatch.setattr(fitted.dataset, "load_ldm", record)
    monkeypatch.setattr(fitted.dataset, "variant_df", fail)
    fit_fixed_effects(fitted, extra_enrich_col=ANNOTATION, include_genes=significant)

    assert loaded == [gene for gene in fitted.dataset.genes if gene in set(significant)]


def test_fixed_enrichment_is_the_ratio_of_ratios(fitted, significant):
    result = fit_fixed_effects(
        fitted, extra_enrich_col=ANNOTATION, include_genes=significant
    )[1]
    assert result.enrichment() == pytest.approx(
        (result.annot_hsq / result.annot_het) / (result.total_hsq / result.total_het)
    )
    assert result.enrichment_jk() is None
    assert result.total_hsq > 0
    scale = result.total_het / result.annot_het
    expected_variance = scale**2 * (
        result.annot_hsq_variance / result.total_hsq**2
        + result.annot_hsq**2 * result.total_hsq_variance / result.total_hsq**4
        - 2 * result.annot_hsq * result.annot_total_hsq_covariance / result.total_hsq**3
    )
    assert np.isfinite(result.enrichment_se())
    assert result.enrichment_se() == pytest.approx(np.sqrt(expected_variance))


def test_nested_burden_designs_generate_base_annotation_interactions(synthetic):
    dataset = Dataset(
        synthetic.variant_df,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=1000.0,
        ldm=synthetic.ldm_path,
    )
    model = Model(
        dataset,
        KernelOptions(burden_cols=["constant", "score"]),
    )
    gene = dataset.genes[0]
    burden_wgts = model.burden_wgts(gene)
    total, annotated = _nested_burden_designs(model, gene, "buried")

    np.testing.assert_array_equal(total[:, : burden_wgts.shape[1]], burden_wgts)
    np.testing.assert_array_equal(
        annotated,
        burden_wgts * dataset.variant_values("buried", gene)[:, None],
    )
    np.testing.assert_array_equal(total[:, burden_wgts.shape[1] :], annotated)


def test_fixed_enrichment_expands_within_gene_groups(synthetic):
    dataset = Dataset(
        synthetic.variant_df,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=1000.0,
        ldm=synthetic.ldm_path,
    )
    model = Model(
        dataset,
        KernelOptions(
            burden_cols=["constant", "score"],
            within_gene_group_col="exon",
        ),
    )
    gene = dataset.genes[0]
    burden_wgts = model.burden_wgts(gene)
    partition = model.burden_partition_codes(gene)
    expanded = np.column_stack(
        [
            burden_wgts[:, burden] * (partition == group)
            for burden in range(burden_wgts.shape[1])
            for group in range(partition.max() + 1)
        ]
    )
    annotation = dataset.variant_values("buried", gene)[:, None]

    total, annotated = _nested_burden_designs(model, gene, "buried")

    np.testing.assert_array_equal(total[:, : expanded.shape[1]], expanded)
    np.testing.assert_array_equal(annotated, expanded * annotation)
    np.testing.assert_array_equal(total[:, expanded.shape[1] :], annotated)
    result = fit_fixed_effects(
        model,
        extra_enrich_col="buried",
        include_genes=[gene],
    )[1]
    assert np.isfinite(result.total_hsq)
    assert np.isfinite(result.annot_hsq)


def test_fixed_full_and_subset_burden_forms_match_dense_oracle(synthetic):
    from scipy.linalg import pinvh

    from kasm import fitting

    dataset = Dataset(
        synthetic.variant_df,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=1000.0,
        ldm=synthetic.ldm_path,
    )
    model = Model(dataset, KernelOptions(burden_cols=["constant", "score"]))
    gene = dataset.genes[0]
    observed = fit_fixed_effects(
        model,
        extra_enrich_col="buried",
        include_genes=[gene],
    )[1]
    total, annotated = _nested_burden_designs(model, gene, "buried")

    expected = []
    for design in (total, annotated):
        covariance = (
            design.T
            @ (dataset.load_ldm(gene).scale(model.variant_scale(gene)).toarray())
            @ design
        )
        inverse, rank = pinvh(covariance, return_rank=True, rtol=1e-6)
        rows = dataset.gene_slice(gene)
        beta_tilde = dataset.beta[rows] / (dataset.se[rows] ** 2 * dataset.sample_size)
        score = design.T @ beta_tilde
        expected.append(
            fitting.fixed_hsq_from_quadforms(
                np.array([score @ inverse @ score]),
                np.array([rank]),
                dataset.sample_size,
            )
        )
    assert observed.total_hsq == pytest.approx(expected[0][0])
    assert observed.total_hsq_se == pytest.approx(expected[0][1])
    assert observed.annot_hsq == pytest.approx(expected[1][0])
    assert observed.annot_hsq_se == pytest.approx(expected[1][1])


def test_fixed_enrichment_is_zero_when_selected_genes_have_no_annotation(
    fitted, significant
):
    annotation = "absent_in_significant"
    frame = fitted.dataset.variant_df().with_columns(
        (~pl.col("gene").is_in(significant)).cast(pl.Float64).alias(annotation)
    )
    model = Model(
        Dataset(
            frame,
            gene_df=fitted.dataset.gene_df(),
            beta_col=fitted.dataset.beta_col,
            se_col=fitted.dataset.se_col,
            sample_size=fitted.dataset.sample_size,
            ldm=fitted.dataset.ldm_source,
        ),
        fitted.options,
    )

    result = fit_fixed_effects(
        model, extra_enrich_col=annotation, include_genes=significant
    )[1]

    assert result.annot_hsq == 0.0
    assert result.annot_hsq_se == 0.0
    assert np.isnan(result.enrichment())


def test_pool_enrichment_pools_fixed_four_moments_before_ratios():
    def result(annot_hsq, total_hsq, annot_het, total_het, variance):
        return EnrichResult(
            annot_hsq=annot_hsq,
            total_hsq=total_hsq,
            annot_het=annot_het,
            total_het=total_het,
            annotation="buried",
            n_genes=1,
            n_variants=2,
            annot_hsq_variance=variance,
            total_hsq_variance=variance + 1,
            annot_total_hsq_covariance=variance / 2,
        )

    weights = np.array([1.0, 3.0])
    pooled = pool_enrichment(
        [result(2, 10, 1, 2, 1), result(9, 20, 3, 4, 3)],
        weights=weights,
        independent=True,
    )

    np.testing.assert_array_equal(weights, [1.0, 3.0])
    assert pooled.annot_hsq == pytest.approx(7.25)
    assert pooled.total_hsq == pytest.approx(17.5)
    assert pooled.annot_het == pytest.approx(2.5)
    assert pooled.total_het == pytest.approx(3.5)
    assert pooled.enrichment() == pytest.approx((7.25 / 2.5) / (17.5 / 3.5))
    assert pooled.annot_hsq_variance == pytest.approx(0.25**2 * 1 + 0.75**2 * 3)


def test_pool_enrichment_pools_aligned_random_delete_moments():
    def result(annot_hsq, total_hsq, annot_het, total_het, annot_hsq_jk):
        n_blocks = len(annot_hsq_jk)
        return EnrichResult(
            annot_hsq=annot_hsq,
            total_hsq=total_hsq,
            annot_het=annot_het,
            total_het=total_het,
            annotation="buried",
            n_genes=3,
            n_variants=6,
            annot_hsq_jk=np.asarray(annot_hsq_jk),
            total_hsq_jk=np.arange(8.0, 8.0 + n_blocks),
            annot_het_jk=np.arange(1.0, 1.0 + n_blocks),
            total_het_jk=np.full(n_blocks, 4.0),
        )

    pooled = pool_enrichment(
        [result(4, 10, 2, 5, [3, 5]), result(8, 12, 3, 6, [9, 7])],
        weights=[1, 1],
    )

    np.testing.assert_allclose(pooled.annot_hsq_jk, [6, 6])
    np.testing.assert_allclose(pooled.total_hsq_jk, [8, 9])
    np.testing.assert_allclose(pooled.annot_het_jk, [1, 2])
    np.testing.assert_allclose(pooled.total_het_jk, [4, 4])
    np.testing.assert_allclose(
        pooled.enrichment_jk(),
        ([6, 6] / np.array([1, 2])) / (np.array([8, 9]) / 4),
    )

    with pytest.raises(ValueError, match="must be aligned"):
        pool_enrichment(
            [
                result(4, 10, 2, 5, [3, 5]),
                result(8, 12, 3, 6, [9, 7, 6]),
            ]
        )

    fixed = EnrichResult(
        annot_hsq=1.0,
        total_hsq=2.0,
        annot_het=1.0,
        total_het=2.0,
        annotation="buried",
        n_genes=1,
        n_variants=2,
    )
    with pytest.raises(ValueError, match="must be aligned"):
        pool_enrichment(
            [fixed, result(4, 10, 2, 5, [3, 5]), result(8, 12, 3, 6, [9, 7, 6])]
        )


def test_fixed_enrichment_retains_an_empty_annotation(fitted, significant):
    frame = fitted.dataset.variant_df().clone()
    frame = frame.with_columns(pl.lit(0.0).alias("never"))
    model = Model(
        Dataset(
            frame,
            gene_df=fitted.dataset.gene_df(),
            beta_col="BETA",
            se_col="SE",
            sample_size=fitted.dataset.sample_size,
            ldm=fitted.dataset.ldm_source,
        ),
        real_options(),
    )
    model.compute_suffstats()
    fit, info = fit_fixed_effects(
        model, extra_enrich_col="never", include_genes=significant
    )
    assert info.annot_hsq == info.annot_het == 0
    assert np.isnan(info.enrichment()) and np.isnan(info.enrichment_se())
    assert np.isfinite(fit.hsq)


def test_enrichment_needs_a_burden_column(test_dataset):
    model = Model(test_dataset, KernelOptions(independent_col="constant", maf_col="AF"))
    model.compute_suffstats()
    with pytest.raises(ValueError, match="at least one burden column"):
        _validate_burden_annotation(model, ANNOTATION)


@pytest.mark.parametrize("n_blocks", [1, 4])
def test_enrichment_uses_one_augmented_fit_and_initializes_deletes(
    fitted, significant, monkeypatch, n_blocks
):
    from kasm import fitting

    solver = fitting.alternating_fit
    calls = []

    def record_solver(*args, **kwargs):
        calls.append(kwargs)
        return solver(*args, **kwargs)

    monkeypatch.setattr(fitting, "alternating_fit", record_solver)
    fitted.compute_jk_suffstats(exclude_genes=significant, n_jk_blocks=n_blocks)
    fit, info = fit_random_effects(
        fitted,
        exclude_genes=significant,
        extra_enrich_col=ANNOTATION,
        n_jk_blocks=n_blocks,
    )
    assert fit.kernel_labels == tuple(
        kernel.label
        for kernel in fitted._with_burden_column(ANNOTATION).layout.kernel_ids
    )
    assert len(calls) == (1 if n_blocks == 1 else n_blocks + 1)
    for call in calls[1:]:
        np.testing.assert_array_equal(call["init_gene_coef"][0], fit.gene_coef)
    if n_blocks == 1:
        np.testing.assert_array_equal(fit.kernel_coef_jk[0], fit.kernel_coef)
        assert np.isnan(info.enrichment_se())


@pytest.mark.parametrize("n_blocks", [1, 2])
def test_enrichment_persists_independently_of_random_fit(
    fitted, significant, tmp_path, monkeypatch, n_blocks
):
    from dataclasses import fields

    import kasm.api as api

    fitted.compute_jk_suffstats(exclude_genes=significant, n_jk_blocks=n_blocks)
    _, random = fit_random_effects(
        fitted,
        exclude_genes=significant,
        extra_enrich_col=ANNOTATION,
        n_jk_blocks=n_blocks,
    )
    fixed = fit_fixed_effects(
        fitted,
        include_genes=significant,
        extra_enrich_col=ANNOTATION,
    )[1]
    results = [
        random,
        fixed,
        pool_enrichment([random, random]),
        pool_enrichment([fixed, fixed], independent=True),
    ]

    def no_refit(*args, **kwargs):
        raise AssertionError("persistence and pooling must not refit")

    monkeypatch.setattr(api, "_fit_random_effects", no_refit)
    for index, result in enumerate(results):
        path = tmp_path / f"enrichment-{index}.npz"
        result.save(path)
        loaded = EnrichResult.load(path)
        for field in fields(result):
            np.testing.assert_equal(
                getattr(loaded, field.name), getattr(result, field.name)
            )
        for name in ("annot_hsq_se", "total_hsq_se"):
            np.testing.assert_equal(getattr(loaded, name), getattr(result, name))
        for name in ("enrichment", "enrichment_jk", "enrichment_se"):
            np.testing.assert_equal(getattr(loaded, name)(), getattr(result, name)())
        pooled = pool_enrichment([loaded, result], independent=True)
        expected = pool_enrichment([result, result], independent=True)
        for field in fields(expected):
            np.testing.assert_equal(
                getattr(pooled, field.name), getattr(expected, field.name)
            )


def test_count_normalization_pools_point_and_delete_denominators(tmp_path):
    results = [
        EnrichResult(
            annot_hsq=a,
            total_hsq=t,
            annot_het=2,
            total_het=5,
            annotation="selected",
            n_genes=3,
            n_variants=10,
            annot_hsq_jk=aj,
            total_hsq_jk=tj,
            annot_het_jk=[1, 1.5, 1.8],
            total_het_jk=[3, 4, 4.5],
            annot_count=c,
            total_count=n,
            annot_count_jk=cj,
            total_count_jk=nj,
        )
        for a, t, aj, tj, c, n, cj, nj in [
            (4, 10, [2, 3, 3.5], [7, 8, 9], 3, 20, [1, 2, 2], [12, 14, 16]),
            (8, 20, [5, 6, 7], [15, 16, 18], 8, 30, [4, 5, 6], [20, 22, 24]),
        ]
    ]
    pooled = pool_enrichment(results, weights=[1, 3])
    expected_deletes = np.array(
        [4.25 / 13 / (3.25 / 18), 5.25 / 14 / (4.25 / 20), 6.125 / 15.75 / (5 / 22)]
    )
    assert pooled.enrichment(normalization="count") == pytest.approx(
        (7 / 17.5) / (6.75 / 27.5)
    )
    np.testing.assert_allclose(
        pooled.enrichment_jk(normalization="count"), expected_deletes
    )
    expected_se = np.sqrt(
        2 / 3 * np.sum((expected_deletes - expected_deletes.mean()) ** 2)
    )
    assert pooled.enrichment_se(normalization="count") == pytest.approx(expected_se)
    path = tmp_path / "enrichment.npz"
    pooled.save(path)
    loaded = EnrichResult.load(path)
    np.testing.assert_allclose(
        loaded.enrichment_jk(normalization="count"), expected_deletes
    )


def test_count_normalization_fixed_covariance_and_optional_counts():
    info = EnrichResult(
        annot_hsq=4,
        total_hsq=10,
        annot_het=2,
        total_het=6,
        annotation="selected",
        n_genes=1,
        n_variants=20,
        annot_hsq_variance=2,
        total_hsq_variance=5,
        annot_total_hsq_covariance=1,
        annot_count=5,
        total_count=20,
    )
    pooled = pool_enrichment([info, info], weights=[1, 3], independent=True)
    assert pooled.enrichment(normalization="count") == pytest.approx(1.6)
    assert pooled.enrichment_se(normalization="count") == pytest.approx(
        np.sqrt(16 * (2 / 100 + 16 * 5 / 10000 - 8 / 1000) * (0.25**2 + 0.75**2))
    )
    without_counts = replace(info, annot_count=None, total_count=None)
    assert np.isfinite(without_counts.enrichment_se(normalization="het"))
    with pytest.raises(ValueError, match="denominator moments"):
        without_counts.enrichment(normalization="count")
    with pytest.raises(ValueError, match="normalization"):
        info.enrichment(normalization="variants")
    mixed = pool_enrichment([info, without_counts], independent=True)
    assert np.isfinite(mixed.enrichment_se(normalization="het"))
    with pytest.raises(ValueError, match="denominator moments"):
        mixed.enrichment(normalization="count")
