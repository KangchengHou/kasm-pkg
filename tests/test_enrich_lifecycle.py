"""Prepared enrichment selection, raw signed moments, and fixed nested semantics."""

import numpy as np
import polars as pl
import pytest
from scipy import sparse

import kasm
from kasm.ldm import LDM


@pytest.fixture
def model(monkeypatch):
    rng = np.random.default_rng(72)
    count, size = 12, 8
    n = count * size
    frame = pl.DataFrame(
        {
            "variant": [f"1:{100 + i}:A:G" for i in range(n)],
            "gene": np.repeat([f"G{i:02}" for i in range(count)], size),
            "beta": rng.normal(0, 0.04, n),
            "se": np.full(n, 0.01),
            "constant": np.ones(n),
            "base": rng.uniform(0.2, 1.2, n),
            "continuous": rng.uniform(0.1, 2, n),
            "signed": np.tile([-3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0], count),
            "binary": np.tile([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0], count),
            "af": np.tile([0.001, 0.02, 0.004, 0.03, 0.006, 0.07, 0.008, 0.1], count),
        }
    )
    dataset = kasm.Dataset(frame, beta_col="beta", se_col="se", sample_size=10000)
    corr = 0.3 ** np.abs(np.subtract.outer(np.arange(size), np.arange(size)))
    monkeypatch.setattr(
        dataset, "load_ldm", lambda gene: LDM(sparse.csr_matrix(corr), np.zeros(size))
    )
    return kasm.Model(
        dataset, kasm.KernelOptions(burden_cols=["base"], independent_col="constant")
    )


@pytest.mark.parametrize("by_maf", [True, False])
def test_prepared_selection_no_ld_or_reaggregation(
    model, monkeypatch, tmp_path, by_maf
):
    from dataclasses import replace

    from kasm import fitting, suffstats

    model = kasm.Model(
        model.dataset,
        replace(
            model.options,
            maf_col="af",
            maf_bins={"low": (0, 0.01), "high": (0.01, 1)},
            burden_by_maf=by_maf,
        ),
    )
    model.compute_suffstats(extra_enrich_cols=["binary", "continuous", "signed"])
    per_gene = model.suffstats
    parent = model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=3)
    assert model.jk_suffstats is parent and model.suffstats is per_gene
    path = tmp_path / "jk.npz"
    parent.save(path)
    loaded = kasm.JKSuffStats.load(path)

    def unexpected(*args, **kwargs):
        pytest.fail("prepared fitting reloaded LD or aggregated blocks")

    monkeypatch.setattr(model.dataset, "load_ldm", unexpected)
    monkeypatch.setattr(suffstats, "compute_jk_suffstats", unexpected)
    monkeypatch.setattr(fitting, "annotation_block_sums", unexpected)
    monkeypatch.setattr(fitting, "annotation_weighted_sums", unexpected)
    for column in ["binary", "continuous", "signed"]:
        fit, info = kasm.fit_random_effects(model, extra_enrich_col=column)
        compact, compact_info = kasm.fit_random_effects(loaded, extra_enrich_col=column)
        np.testing.assert_array_equal(fit.kernel_coef_jk, compact.kernel_coef_jk)
        np.testing.assert_array_equal(info.annot_hsq_jk, compact_info.annot_hsq_jk)
        with pytest.raises(TypeError, match="MAF components are unavailable"):
            fit.maf_components
        assert isinstance(info, kasm.EnrichResult)
    assert kasm.fit_random_effects(model).maf_components.height == 2
    assert model.suffstats is per_gene and model.jk_suffstats is parent


def test_model_cache_lifecycle_and_explicit_geometry(model):
    assert model.suffstats is None
    assert model.jk_suffstats is None
    model.compute_suffstats(extra_enrich_cols=["continuous"])
    with pytest.raises(TypeError, match="exclude_genes"):
        model.compute_jk_suffstats()
    blocks = np.arange(len(model.genes)) % 3
    stats = model.compute_jk_suffstats(
        exclude_genes=[model.genes[0]], n_jk_blocks=3, gene_blocks=blocks
    )
    kasm.fit_random_effects(
        model, exclude_genes=[model.genes[0]], n_jk_blocks=3, gene_blocks=blocks
    )
    for options in [
        {"exclude_genes": []},
        {"n_jk_blocks": 2},
        {"gene_blocks": blocks[::-1]},
    ]:
        with pytest.raises(ValueError, match="JK cache"):
            kasm.fit_random_effects(model, **options)
    model.suffstats = model.suffstats
    assert model.jk_suffstats is stats
    model.jk_suffstats = None
    assert model.jk_suffstats is None
    model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=3)
    model.compute_suffstats(extra_enrich_cols=["continuous"])
    assert model.jk_suffstats is None and model.suffstats is not stats


def test_unprepared_model_auto_preparation_matches_explicit_with_extension(model):
    model.compute_suffstats(extra_enrich_cols=["continuous"])
    blocks = np.arange(len(model.genes)) % 3
    explicit_stats = model.compute_jk_suffstats(
        exclude_genes=[model.genes[0]], n_jk_blocks=3, gene_blocks=blocks
    )
    explicit_fit, explicit_info = kasm.fit_random_effects(
        explicit_stats, extra_enrich_col="continuous"
    )

    model.jk_suffstats = None
    auto_fit, auto_info = kasm.fit_random_effects(
        model,
        exclude_genes=[model.genes[0]],
        n_jk_blocks=3,
        gene_blocks=blocks,
        extra_enrich_col="continuous",
    )
    assert model.jk_suffstats is not None
    np.testing.assert_allclose(auto_fit.kernel_coef, explicit_fit.kernel_coef)
    np.testing.assert_allclose(auto_fit.kernel_coef_jk, explicit_fit.kernel_coef_jk)
    np.testing.assert_allclose(auto_info.annot_hsq_jk, explicit_info.annot_hsq_jk)


@pytest.mark.parametrize("column", ["binary", "continuous", "signed"])
@pytest.mark.parametrize("blocks", [1, 3])
def test_raw_annotation_attribution_dense_oracle(model, column, blocks):
    model.compute_suffstats(extra_enrich_cols=[column])
    parent = model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=blocks)
    fit, info = kasm.fit_random_effects(model, extra_enrich_col=column)
    augmented = model.with_burden_extension(column)
    for delete in [None, *range(blocks)]:
        coeff = fit.kernel_coef if delete is None else fit.kernel_coef_jk[delete]
        scale = (
            fit.gene_coef[0, 0] if delete is None else fit.gene_coef_jk[delete, 0, 0]
        )
        moments = np.zeros(4)
        for g, gene in enumerate(model.genes):
            if delete is not None and blocks > 1 and parent.gene_blocks[g] == delete:
                continue
            weights = augmented.burden_wgts(gene)
            diagonal = np.zeros(len(weights))
            for k, kernel in enumerate(augmented.layout.kernel_ids):
                if kernel.kind == "B":
                    left = (
                        augmented.layout.burden_indices.index(k)
                        if kernel.col2 is None
                        else None
                    )
                    if left is not None:
                        values = weights[:, left] ** 2
                    else:
                        pair = augmented.layout.cross_pairs[
                            augmented.layout.cross_indices.index(k)
                        ]
                        values = 2 * weights[:, pair[0]] * weights[:, pair[1]]
                elif kernel.kind == "I":
                    values = np.ones(len(weights))
                else:
                    continue
                diagonal += coeff[k] * scale * values
            d2 = model.variant_scale(gene) ** 2
            a = model.dataset.variant_values(column, gene)
            moments += [
                np.sum(a * d2 * diagonal),
                np.sum(d2 * diagonal),
                np.sum(a * d2),
                np.sum(d2),
            ]
        names = ["annot_hsq", "total_hsq", "annot_het", "total_het"]
        actual = [
            getattr(info, name)
            if delete is None
            else getattr(info, name + "_jk")[delete]
            for name in names
        ]
        np.testing.assert_allclose(actual, moments, rtol=1e-11, atol=1e-12)
    if column == "signed":
        assert info.annot_het == 0
        assert np.isnan(info.enrichment()) and np.isnan(info.enrichment_se())
        assert np.isfinite(info.annot_hsq_jk).all()
    if blocks == 1:
        assert np.isnan(info.annot_hsq_se) and np.isnan(info.enrichment_se())
        assert info.annot_hsq_jk[0] == info.annot_hsq


@pytest.mark.parametrize("column", ["binary", "continuous", "signed"])
def test_fixed_selection_retains_nested_projection_and_noise_correction(model, column):
    # Base has no constant column, so [X,a] and [X,X*a] are distinct designs.
    selected = model.genes[:2]
    fit, info = kasm.fit_fixed_effects(
        model, include_genes=selected, extra_enrich_col=column
    )
    expected = []
    annotated = []
    for gene in selected:
        x = model.burden_wgts(gene)
        a = model.dataset.variant_values(column, gene)[:, None]
        ld = model.dataset.load_ldm(gene).scale(model.variant_scale(gene)).toarray()
        rows = model.dataset.gene_slice(gene)
        beta = model.dataset.beta[rows] / (
            model.dataset.se[rows] ** 2 * model.dataset.sample_size
        )
        for design, destination in [
            (np.column_stack([x, x * a]), expected),
            (x * a, annotated),
        ]:
            gram = design.T @ ld @ design
            rank = np.linalg.matrix_rank(gram)
            score = design.T @ beta
            q = score @ np.linalg.pinv(gram) @ score
            destination.append(
                (model.dataset.sample_size * q - rank)
                / (model.dataset.sample_size - rank)
            )
    assert fit.hsq == pytest.approx(sum(expected))
    assert info.annot_hsq == pytest.approx(sum(annotated))
    ordinary = kasm.fit_fixed_effects(
        kasm.Model(model.dataset, kasm.KernelOptions(burden_cols=["base", column])),
        include_genes=selected,
    )
    assert not np.isclose(fit.hsq, ordinary.hsq)


def test_undefined_delete_ratio_keeps_signed_raw_moments():
    info = kasm.EnrichResult(
        annot_hsq=-2.0,
        total_hsq=3.0,
        annot_het=-4.0,
        total_het=5.0,
        annotation="signed",
        n_genes=3,
        n_variants=6,
        annot_hsq_jk=[-1.0, -2.0],
        total_hsq_jk=[2.0, 1.0],
        annot_het_jk=[0.0, -3.0],
        total_het_jk=[4.0, 3.0],
    )
    assert info.enrichment() == pytest.approx(5 / 6)
    assert np.isnan(info.enrichment_jk()[0]) and np.isnan(info.enrichment_se())
    assert np.isfinite(info.annot_hsq_se)
    np.testing.assert_array_equal(info.annot_hsq_jk, [-1.0, -2.0])
    from dataclasses import replace

    with pytest.raises(ValueError, match="raw moments must be finite"):
        replace(info, annot_hsq=np.inf)
    with pytest.raises(ValueError, match="raw moments must be finite"):
        replace(info, total_hsq_jk=[np.nan, 1.0])


def test_prepared_explicit_block_count_is_not_silently_capped(model):
    model.compute_suffstats()
    model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=len(model.genes))
    with pytest.raises(ValueError, match="n_jk_blocks"):
        kasm.fit_random_effects(model, n_jk_blocks=100)


@pytest.mark.parametrize("operation", ["prepare", "random", "fixed", "prepared_random"])
def test_multigroup_enrichment_rejected_before_computation(
    model, monkeypatch, operation
):
    grouped = kasm.Model(
        model.dataset,
        kasm.KernelOptions(
            kernel_groups={
                "first": {"burden_cols": ["base"]},
                "second": {"burden_cols": ["continuous"]},
            }
        ),
    )
    if operation == "prepared_random":
        grouped.compute_suffstats()
        stats = grouped.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)

    def unexpected(*args, **kwargs):
        pytest.fail("multi-group enrichment reached numerical computation")

    monkeypatch.setattr(model.dataset, "load_ldm", unexpected)
    with pytest.raises(ValueError, match="enrichment requires a single kernel group"):
        if operation == "prepare":
            grouped.compute_suffstats(extra_enrich_cols=["binary"])
        elif operation == "random":
            kasm.fit_random_effects(
                grouped, extra_enrich_col="binary", exclude_genes=[]
            )
        elif operation == "fixed":
            kasm.fit_fixed_effects(
                grouped, extra_enrich_col="binary", include_genes=grouped.genes
            )
        else:
            kasm.fit_random_effects(stats, extra_enrich_col="binary")


def test_single_explicit_group_enrichment_needs_no_independent_column(model):
    grouped = kasm.Model(
        model.dataset,
        kasm.KernelOptions(kernel_groups={"target": {"burden_cols": ["base"]}}),
    )
    grouped.compute_suffstats(extra_enrich_cols=["binary"])
    grouped.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    fit, info = kasm.fit_random_effects(grouped, extra_enrich_col="binary")
    assert tuple(fit.kernel_group_names) == ("target",)
    assert np.isfinite(info.annot_hsq)
