"""Tests for post-hoc decomposition of fixed-effect fits."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import kasm
from kasm.fixed_effects import (
    _allocation_covariance,
    _quadratic_form_covariance,
)


def _one_gene_dataset(dataset, gene="GENE_C"):
    frame = (
        dataset.variant_df()
        .filter(pl.col("gene") == gene)
        .with_columns(
            pl.Series("binary", np.arange(dataset.variants_per_gene[gene]) % 2),
            pl.Series(
                "quartile",
                [f"Q{i + 1}" for i in range(dataset.variants_per_gene[gene])],
            ),
        )
    )
    return kasm.Dataset(
        frame,
        beta_col=dataset.beta_col,
        se_col=dataset.se_col,
        sample_size=dataset.sample_size,
        ldm=dataset.ldm_source,
    )


def _options():
    return kasm.KernelOptions(
        maf_col="af",
        maf_bins={"rare": (0, 5e-4), "common": (5e-4, 1.0)},
        burden_cols=["constant", "score"],
        burden_by_maf=True,
    )


def _weights(options):
    labels = options.layout.burden_labels
    return dict(zip(labels, np.linspace(1.0, 0.5, len(labels))))


@pytest.mark.parametrize("method", ["burden", "burden_1d", "indep"])
def test_fixed_decomposition_uses_common_additive_whole_gene_denominator(
    synthetic_dataset_no_ld,
    method,
):
    dataset = _one_gene_dataset(synthetic_dataset_no_ld)
    options = _options()
    fit = kasm.fit_fixed_effects(
        dataset,
        kernel_options=options,
        include_genes=dataset.genes,
        method=method,
        burden_1d_wgts=_weights(options) if method == "burden_1d" else None,
    )

    result = fit.decompose(["binary", "quartile"])

    for (source,), rows in result.group_by("source", maintain_order=True):
        assert rows["total_hsq"].n_unique() == 1
        assert rows["hsq"].sum() == pytest.approx(rows["total_hsq"][0])
        assert rows["total_hsq"][0] == pytest.approx(fit.hsq)
        assert rows["total_hsq_se"].n_unique() == 1
        assert rows["total_hsq_se"][0] == pytest.approx(fit.hsq_se)
        assert rows["hsq_total_covariance"].sum() == pytest.approx(
            rows["total_hsq_se"][0] ** 2
        )
        assert rows["heterozygosity_proportion"].sum() == pytest.approx(1.0)
        np.testing.assert_allclose(
            rows["enrichment"],
            (rows["hsq"] / rows["total_hsq"]) / rows["heterozygosity_proportion"],
        )
        expected_share_variance = (
            rows["hsq_se"] ** 2 / rows["total_hsq"] ** 2
            + rows["hsq"] ** 2 * rows["total_hsq_se"] ** 2 / rows["total_hsq"] ** 4
            - 2 * rows["hsq"] * rows["hsq_total_covariance"] / rows["total_hsq"] ** 3
        )
        np.testing.assert_allclose(rows["hsq_share_se"] ** 2, expected_share_variance)
        np.testing.assert_allclose(
            rows["enrichment_se"],
            rows["hsq_share_se"] / rows["heterozygosity_proportion"],
        )

    assert result.columns == [
        "source",
        "group",
        "method",
        "n_genes",
        "n_variants",
        "hsq",
        "hsq_se",
        "total_hsq",
        "total_hsq_se",
        "hsq_total_covariance",
        "heterozygosity_proportion",
        "hsq_share",
        "hsq_share_se",
        "enrichment",
        "enrichment_se",
    ]
    assert result.filter(pl.col("source") == "partition:binary")["group"].to_list() == [
        "0.0",
        "1.0",
    ]
    assert result.filter(pl.col("source") == "partition:quartile")[
        "group"
    ].to_list() == [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    ]


def test_fixed_decomposition_validates_inputs(synthetic_dataset):
    dataset = _one_gene_dataset(synthetic_dataset)
    options = _options()
    with pytest.raises(ValueError, match="kernel_options is required"):
        kasm.fit_fixed_effects(dataset, include_genes=dataset.genes, method="indep")

    fit = kasm.fit_fixed_effects(
        dataset,
        kernel_options=options,
        include_genes=dataset.genes,
        method="indep",
    )
    with pytest.raises(KeyError, match="never"):
        fit.decompose("never")
    with pytest.raises(ValueError, match="at least two groups"):
        fit.decompose("constant")


def test_fixed_decomposition_excludes_genes_whose_rank_reaches_sample_size(
    synthetic_dataset_no_ld,
):
    frame = synthetic_dataset_no_ld.variant_df().filter(
        pl.col("gene").is_in(["GENE_A", "GENE_C"])
    )
    dataset = kasm.Dataset(
        frame,
        beta_col=synthetic_dataset_no_ld.beta_col,
        se_col=synthetic_dataset_no_ld.se_col,
        sample_size=4,
    )
    fit = kasm.fit_fixed_effects(
        dataset,
        kernel_options=kasm.KernelOptions(burden_cols=["constant"]),
        include_genes=dataset.genes,
        method="indep",
    )

    result = fit.decompose("exon")

    assert fit.per_gene.select("gene", "rank").to_dicts() == [
        {"gene": "GENE_A", "rank": 3.0},
        {"gene": "GENE_C", "rank": 4.0},
    ]
    assert set(result["n_genes"]) == {1}
    assert result["n_variants"].sum() == 3
    assert result["hsq"].sum() == pytest.approx(fit.hsq)
    assert result["total_hsq"].unique().item() == pytest.approx(fit.hsq)
    assert result["heterozygosity_proportion"].sum() == pytest.approx(1.0)
    assert np.isfinite(result.select(result.columns[3:]).to_numpy()).all()


def test_quadratic_covariance_null_has_no_signal_inflation():
    residual = np.array([[1.0, 0.2], [0.2, 0.8]])
    operator = (np.eye(2), np.linalg.inv(residual))
    sample_size = 1_000
    beta_tilde = np.linalg.cholesky(residual) @ np.array(
        [np.sqrt(2 / sample_size), 0.0]
    )

    variance = _quadratic_form_covariance(
        beta_tilde, residual, operator, operator, sample_size, 2, 2
    )

    assert variance == pytest.approx(
        (sample_size / (sample_size - 2)) ** 2 * 4 / sample_size**2
    )


@pytest.mark.parametrize("method", ["burden", "indep"])
def test_allocation_covariance_matches_dense_oracle(method):
    y = np.array([0.12, -0.07, 0.2, 0.04])
    residual = np.array(
        [
            [1.2, 0.1, 0.0, 0.05],
            [0.1, 0.9, 0.08, 0.0],
            [0.0, 0.08, 1.1, -0.03],
            [0.05, 0.0, -0.03, 0.8],
        ]
    )
    if method == "burden":
        design = np.array([[1.0], [0.7], [-0.3], [0.4]])
        inverse = np.linalg.pinv(design.T @ residual @ design)
        operator = design @ inverse @ design.T
        rank = 1
    else:
        operator = np.linalg.pinv(residual)
        design, inverse = np.eye(4), operator
        rank = 4
    masks = [
        np.array([True, True, False, False]),
        np.array([False, False, True, True]),
    ]
    sample_size = 500

    actual = _allocation_covariance(
        y, residual, (design, inverse), masks, sample_size, rank
    )
    selectors = [np.diag(mask.astype(float)) for mask in masks]
    matrices = [
        (selector @ operator + operator @ selector) / 2 for selector in selectors
    ]
    trace_terms = np.array(
        [
            [np.trace(left @ residual @ right @ residual) for right in matrices]
            for left in matrices
        ]
    )
    signal_terms = np.array(
        [
            [(left @ y) @ residual @ (right @ y) for right in matrices]
            for left in matrices
        ]
    )
    total_hsq = max(
        0.0,
        (sample_size * (y @ operator @ y) - rank) / (sample_size - rank),
    )
    expected = 2 * trace_terms / sample_size**2
    expected += 4 * total_hsq / sample_size * signal_terms / signal_terms.sum()

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual, actual.T)
    assert np.linalg.eigvalsh(actual).min() >= -1e-14
    total_variance = 2 * rank / sample_size**2 + 4 * total_hsq / sample_size
    assert actual.sum() == pytest.approx(total_variance)
