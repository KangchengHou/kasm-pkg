"""Kernel enumeration, configuration validation, and Model design matrices."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from conftest import (
    SYNTHETIC_MAF_BINS,
)

from kasm import Dataset, KernelId, KernelOptions, Model

TWO_BINS = {"rare": (0, 1e-4), "common": (1e-4, 1.0)}


def extra_kernels(_frame):
    return {}


def synthetic_options(**overrides) -> KernelOptions:
    settings = dict(
        maf_col="af",
        maf_bins=SYNTHETIC_MAF_BINS,
        gene_annot_cols=["constraint_low", "constraint_high"],
        burden_cols=["constant", "score"],
        burden_by_maf=True,
        overdispersion_cols=["constant"],
        overdispersion_by_maf=True,
    )
    settings.update(overrides)
    return KernelOptions(**settings)


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def test_kernel_order_is_burden_then_cross_then_diagonal_then_overdispersion():
    layout = KernelOptions(
        maf_col="af",
        maf_bins=TWO_BINS,
        burden_cols=["constant", "score"],
        burden_by_maf=True,
        independent_col="constant",
        overdispersion_cols=["constant"],
        overdispersion_by_maf=True,
    ).layout

    # 4 burden (2 cols x 2 bins), 6 crosses, 1 identity, 2 overdispersion.
    # Burden and overdispersion kernels keep the caller's column order crossed
    # with the caller's bin order. Cross labels are canonicalized, so their two
    # components are sorted by (column, bin) name and `common` precedes `rare`;
    # the pair they came from is checked against the burden columns below.
    assert [kid.label for kid in layout.kernel_ids] == [
        "B:constant[rare]",
        "B:constant[common]",
        "B:score[rare]",
        "B:score[common]",
        "B:constant[common]*constant[rare]",
        "B:constant[rare]*score[rare]",
        "B:constant[rare]*score[common]",
        "B:constant[common]*score[rare]",
        "B:constant[common]*score[common]",
        "B:score[common]*score[rare]",
        "I:constant",
        "O:constant[rare]",
        "O:constant[common]",
    ]
    assert layout.n_kernels == 13
    assert layout.n_burden == 4
    assert layout.burden_indices == (0, 1, 2, 3)
    assert layout.cross_indices == (4, 5, 6, 7, 8, 9)
    assert layout.independent_index == 10
    assert layout.independent_indices == (10,)
    assert layout.n_independent == 1
    assert layout.overdispersion_indices == (11, 12)
    assert layout.n_overdispersion == 2
    assert layout.n_diagonal == layout.n_independent + layout.n_overdispersion
    # Overdispersion is the tail, so a hsq fit can truncate it.
    assert max(layout.burden_indices + (layout.independent_index,)) < min(
        layout.overdispersion_indices
    )

    for cross_index, (left, right) in zip(layout.cross_indices, layout.cross_pairs):
        cross = layout.kernel_ids[cross_index]
        assert left < right
        components = {
            (cross.col, cross.maf_label),
            (cross.col2, cross.maf_label2),
        }
        expected = set()
        for burden_position in (left, right):
            kid = layout.kernel_ids[layout.burden_indices[burden_position]]
            expected.add((kid.col, kid.maf_label))
        assert components == expected
    n_burden = layout.n_burden
    assert len(layout.cross_pairs) == n_burden * (n_burden - 1) // 2

    assert layout.diagonal_indices == (
        layout.independent_index,
        *layout.overdispersion_indices,
    )

    heritable = set(layout.heritable_indices)
    assert heritable == set(layout.burden_indices) | set(layout.cross_indices) | {
        layout.independent_index
    }
    assert heritable.isdisjoint(layout.overdispersion_indices)


def test_named_kernel_groups_control_between_group_covariance():
    options = KernelOptions(
        maf_col="af",
        maf_bins=TWO_BINS,
        kernel_groups={
            "missense": {
                "burden_cols": ["m1", "m2"],
                "independent_col": "is_missense",
                "overdispersion_cols": ["is_missense"],
            },
            "plof": {
                "burden_cols": ["l"],
                "independent_col": "is_plof",
                "overdispersion_cols": ["is_plof"],
            },
        },
        covariance_between_kernel_groups={"missense_x_plof": ("missense", "plof")},
        overdispersion_shared=True,
        overdispersion_by_maf=True,
    )
    layout = options.layout

    assert layout.kernel_group_names == ("missense", "plof", "missense_x_plof")
    grouped = {
        name: {
            kernel.label
            for kernel, group in zip(
                layout.kernel_ids, layout.kernel_groups, strict=True
            )
            if group == index
        }
        for index, name in enumerate(layout.kernel_group_names)
    }
    assert "I:is_missense" in grouped["missense"]
    assert "I:is_plof" in grouped["plof"]
    assert all(
        layout.kernel_ids[index].kind == "B"
        for index, group in enumerate(layout.kernel_groups)
        if group == 2
    )
    assert len(grouped["missense_x_plof"]) == 8
    assert all(
        any(column in label for column in ("m1", "m2")) and "l" in label
        for label in grouped["missense_x_plof"]
    )
    rebuilt = KernelOptions(
        **{
            key: value
            for key, value in options.to_dict().items()
            if key != "kernel_labels"
        }
    )
    assert rebuilt.layout == layout


@pytest.mark.parametrize("independent", [False, True])
def test_grouped_overdispersion_has_explicit_owners_without_independent(independent):
    options = KernelOptions(
        kernel_groups={
            "missense": {
                "burden_cols": ["missense"],
                "independent_col": "missense" if independent else None,
                "overdispersion_cols": ["missense"],
            },
            "plof": {"burden_cols": ["plof"], "overdispersion_cols": ["plof"]},
        },
        overdispersion_shared=True,
    )
    assert [
        options.layout.kernel_ids[i].label
        for i in options.layout.overdispersion_indices
    ] == ["O:missense", "O:plof"]
    assert [
        options.layout.kernel_groups[i] for i in options.layout.overdispersion_indices
    ] == [0, 1]


def test_configured_group_requires_a_burden_column():
    with pytest.raises(ValueError, match="requires at least one burden column"):
        KernelOptions(
            kernel_groups={
                "missense": {
                    "burden_cols": [],
                    "independent_col": "is_missense",
                },
                "plof": {"burden_cols": ["plof"]},
            },
            covariance_between_kernel_groups={"cross": ("missense", "plof")},
        )


def test_cross_kernel_identity_is_order_free():
    assert KernelId("B", "b", "rare", "a", "common") == KernelId(
        "B", "a", "common", "b", "rare"
    )
    assert KernelId("B", "a", "rare").is_cross is False
    assert KernelId("B", "a", "rare", "b", "rare").is_cross is True
    assert KernelId("I", "constant").is_diagonal
    assert KernelId("O", "c").is_diagonal
    assert not KernelId("B", "a").is_diagonal


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("maf", [0.0, 1.0])
def test_maf_scaled_models_reject_zero_genotype_variance(synthetic, maf):
    frame = synthetic.variant_df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(maf)
        .otherwise(pl.col("af"))
        .alias("af")
    )
    dataset = Dataset(
        frame,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=50_000,
    )

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        Model(
            dataset,
            KernelOptions(
                maf_col="af",
                independent_col="constant",
                variant_variance_from_maf=True,
            ),
        )


@pytest.mark.parametrize("maf", [1e-12, 1.0 - 1e-12])
def test_maf_scaled_models_accept_positive_genotype_variance(synthetic, maf):
    frame = synthetic.variant_df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(maf)
        .otherwise(pl.col("af"))
        .alias("af")
    )
    dataset = Dataset(
        frame,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=50_000,
    )

    Model(
        dataset,
        KernelOptions(
            maf_col="af",
            independent_col="constant",
            variant_variance_from_maf=True,
        ),
    )


@pytest.mark.parametrize(
    "maf_bins,message",
    [
        ({}, "must not be empty"),
        ({"bad": (0.5, 0.5)}, "low >= high"),
        ({"bad": (0.5, 0.1)}, "low >= high"),
        ({"a": (0, 1e-3), "b": (1e-4, 1.0)}, "overlap"),
        ({"a": (0, 1e-3, 2)}, r"\(low, high\) pair"),
    ],
)
def test_rejects_unusable_maf_bins(maf_bins, message):
    with pytest.raises(ValueError, match=message):
        KernelOptions(maf_col="af", maf_bins=maf_bins, burden_cols=["constant"])


def test_grouped_extra_covariance_requires_independent_col():
    with pytest.raises(ValueError, match="requires independent_col"):
        KernelOptions(
            kernel_groups={
                "a": {
                    "burden_cols": ["is_a"],
                    "extra_covar_kernel_fun": extra_kernels,
                }
            }
        )


def test_within_gene_group_requires_a_complete_variant_column(synthetic):
    dataset = Dataset(
        synthetic.variant_df,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=50_000,
    )
    with pytest.raises(KeyError, match="variant_df has no column 'missing_partition'"):
        Model(
            dataset,
            KernelOptions(
                burden_cols=["constant"],
                within_gene_group_col="missing_partition",
            ),
        )

    frame = synthetic.variant_df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(None)
        .otherwise(pl.lit("one"))
        .alias("incomplete_partition")
    )
    incomplete = Dataset(
        frame,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=50_000,
    )
    with pytest.raises(ValueError, match="must not contain missing"):
        Model(
            incomplete,
            KernelOptions(
                burden_cols=["constant"],
                within_gene_group_col="incomplete_partition",
            ),
        )


# ---------------------------------------------------------------------------
# Model binding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_b", [pl.col("is_a"), pl.lit(0.0)])
def test_explicit_overdispersion_does_not_require_independent_partition(
    synthetic, is_b
):
    frame = synthetic.variant_df.with_columns(
        (pl.col("score") >= 0.5).cast(pl.Float64).alias("is_a")
    ).with_columns(is_b.alias("is_b"))
    dataset = Dataset(
        frame,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=100.0,
    )
    Model(
        dataset,
        KernelOptions(
            kernel_groups={
                "a": {
                    "burden_cols": ["is_a"],
                    "independent_col": "is_a",
                    "overdispersion_cols": ["is_a"],
                },
                "b": {
                    "burden_cols": ["is_b"],
                    "independent_col": "is_b",
                    "overdispersion_cols": ["is_b"],
                },
            },
            overdispersion_shared=True,
        ),
    )


def test_model_rejects_rank_deficient_gene_annotations(synthetic, synthetic_dataset):
    gene_df = synthetic.gene_df.with_columns(pl.lit(0.0).alias("empty_annotation"))
    from kasm import Dataset

    dataset = Dataset(
        synthetic.variant_df,
        gene_df=gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=100.0,
    )
    with pytest.raises(ValueError, match="not full rank"):
        Model(
            dataset,
            synthetic_options(gene_annot_cols=["constraint_low", "empty_annotation"]),
        )


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        (pl.col("constraint_low") * 0.5, "binary"),
        (pl.lit(1.0), "mutually exclusive"),
    ],
)
def test_model_rejects_non_partition_gene_annotations(synthetic, expression, message):
    gene_df = synthetic.gene_df.with_columns(expression.alias("invalid_annotation"))
    dataset = Dataset(
        synthetic.variant_df,
        gene_df=gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=100.0,
    )
    with pytest.raises(ValueError, match=message):
        Model(
            dataset,
            synthetic_options(gene_annot_cols=["constraint_low", "invalid_annotation"]),
        )


def test_model_rejects_maf_outside_the_unit_interval(synthetic):
    from kasm import Dataset

    variant_df = synthetic.variant_df.with_columns((-pl.col("af")).alias("af"))
    dataset = Dataset(
        variant_df,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=100.0,
    )
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        Model(dataset, synthetic_options())


def test_model_rejects_bins_that_match_nothing(synthetic_dataset):
    with pytest.raises(ValueError, match="no variant falls in any MAF bin"):
        Model(synthetic_dataset, synthetic_options(maf_bins={"impossible": (0.9, 1.0)}))


# ---------------------------------------------------------------------------
# Kernel weights
# ---------------------------------------------------------------------------


def test_burden_wgts_mask_each_column_to_its_maf_bin(synthetic_dataset):
    model = Model(synthetic_dataset, synthetic_options())
    gene = synthetic_dataset.genes[0]
    burden_wgts = model.burden_wgts(gene)
    assert burden_wgts.shape == (synthetic_dataset.variants_per_gene[gene], 4)

    af = synthetic_dataset.variant_values("af", gene)
    score = synthetic_dataset.variant_values("score", gene)
    for position, label in enumerate(model.layout.burden_labels):
        column, bin_name = label.rstrip("]").split("[")
        low, high = SYNTHETIC_MAF_BINS[bin_name]
        mask = (low <= af) & (af < high)
        source = np.ones_like(af) if column == "constant" else score
        np.testing.assert_array_equal(
            burden_wgts[:, position], np.where(mask, source, 0.0)
        )


def test_diagonal_matrix_gives_the_identity_kernel_unit_weight(synthetic_dataset):
    model = Model(synthetic_dataset, synthetic_options(independent_col="constant"))
    gene = synthetic_dataset.genes[0]
    diagonal = model.diag_kernel_weights(gene)
    assert diagonal.shape == (
        synthetic_dataset.variants_per_gene[gene],
        model.layout.n_diagonal,
    )
    np.testing.assert_array_equal(diagonal[:, 0], 1.0)
    # The MAF-stratified overdispersion columns sum to the identity column
    # wherever a variant is binned at all, so the diagonal *design* is
    # rank-deficient. The fitted system is not: heritable kernels, the identity
    # among them, are scaled by per-gene coefficients built from
    # gene_annot_cols, while overdispersion kernels are not — so the two are
    # distinct system columns whenever that gene coefficient varies. The
    # reference's production subthreshold model uses exactly this combination
    # (kasm/analyses/position/03-subthreshold.py). With no gene covariates they
    # do coincide; that case is caught at fit time.
    af = synthetic_dataset.variant_values("af", gene)
    binned = np.zeros_like(af, dtype=bool)
    for low, high in SYNTHETIC_MAF_BINS.values():
        binned |= (low <= af) & (af < high)
    np.testing.assert_array_equal(diagonal[:, 1:].sum(axis=1), binned.astype(float))


def test_annotated_independent_kernel_has_a_heritable_diagonal(
    synthetic, synthetic_dataset
):
    frame = synthetic.variant_df.with_columns(
        (pl.int_range(pl.len()) % 2 == 0).cast(pl.Float64).alias("class_a"),
        (pl.int_range(pl.len()) % 2 == 1).cast(pl.Float64).alias("class_b"),
    )
    dataset = Dataset(
        frame,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=50_000,
    )
    model = Model(
        dataset,
        KernelOptions(burden_cols=["constant"], independent_col="class_a"),
    )

    assert [
        model.layout.kernel_ids[index].label
        for index in model.layout.independent_indices
    ] == ["I:class_a"]
    np.testing.assert_array_equal(
        model.diag_kernel_weights()[:, 0],
        frame["class_a"].to_numpy(),
    )
    assert set(model.layout.independent_indices) <= set(model.layout.heritable_indices)

    with pytest.raises(ValueError, match="must be nonnegative"):
        Model(
            Dataset(
                frame.with_columns((-pl.col("class_a")).alias("class_a")),
                gene_df=synthetic.gene_df,
                beta_col="beta",
                se_col="se",
                sample_size=50_000,
            ),
            KernelOptions(independent_col="class_a"),
        )

    with pytest.raises(TypeError, match="must be a string"):
        KernelOptions(independent_col=["class_a"])


def test_variant_scale_conventions_differ(synthetic_dataset):
    from_se = Model(synthetic_dataset, synthetic_options()).variant_scale()
    from_maf = Model(
        synthetic_dataset, synthetic_options(variant_variance_from_maf=True)
    ).variant_scale()

    se = synthetic_dataset.se
    np.testing.assert_allclose(
        from_se, 1.0 / (se * np.sqrt(synthetic_dataset.sample_size))
    )
    af = synthetic_dataset.variant_values("af")
    np.testing.assert_allclose(from_maf, np.sqrt(2 * af * (1 - af)))
    # The scales parameterize different observation geometries.
    assert not np.allclose(from_se, from_maf)


def _edge_dataset():
    """Three variants: below a bin edge, exactly on it, and above."""
    from kasm import Dataset

    return Dataset(
        pl.DataFrame(
            {
                "variant": ["1:10:A:G", "1:20:A:G", "1:30:A:G"],
                "gene": ["G", "G", "G"],
                "af": [0.5e-4, 1e-4, 2e-4],
                "constant": [1.0, 1.0, 1.0],
                "beta": [0.0, 0.0, 0.0],
                "se": [1.0, 1.0, 1.0],
            },
        ),
        beta_col="beta",
        se_col="se",
        sample_size=100.0,
    )


def test_maf_bins_are_half_open_at_the_lower_bound():
    """`low <= maf < high`, so a variant on an edge belongs to the upper bin.

    Real allele frequencies are ratios of integers, so exact bin edges do occur;
    a closed upper bound would put such a variant in two bins at once.
    """
    model = Model(
        _edge_dataset(),
        KernelOptions(
            maf_col="af",
            maf_bins={"low": (0, 1e-4), "high": (1e-4, 1.0)},
            burden_cols=["constant"],
            burden_by_maf=True,
        ),
    )
    burden_wgts = model.burden_wgts()
    assert model.layout.burden_labels == ("constant[low]", "constant[high]")
    np.testing.assert_array_equal(burden_wgts[:, 0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(burden_wgts[:, 1], [0.0, 1.0, 1.0])
    # Exactly one bin per variant: no double counting, none dropped.
    np.testing.assert_array_equal((burden_wgts != 0).sum(axis=1), 1)
    assert model.n_unbinned == 0


def test_unbinned_variants_are_counted_and_excluded():
    model = Model(
        _edge_dataset(),
        KernelOptions(
            maf_col="af",
            maf_bins={"only": (1e-4, 1.5e-4)},
            burden_cols=["constant"],
            burden_by_maf=True,
        ),
    )
    # Bin covers 1e-4 only: the 0.5e-4 and 2e-4 variants fall outside.
    assert model.n_unbinned == 2
    np.testing.assert_array_equal(model.burden_wgts()[:, 0], [0.0, 1.0, 0.0])


def test_rejects_non_finite_annotation_values(synthetic):
    from kasm import Dataset

    variant_df = synthetic.variant_df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(np.nan)
        .otherwise(pl.col("score"))
        .alias("score")
    )
    with pytest.raises(ValueError, match="finite"):
        Dataset(
            variant_df,
            gene_df=synthetic.gene_df,
            beta_col="beta",
            se_col="se",
            sample_size=100.0,
        )


def test_rejects_non_finite_gene_annotations(synthetic):
    from kasm import Dataset

    value_col = next(col for col in synthetic.gene_df.columns if col != "gene")
    gene_df = synthetic.gene_df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(np.inf)
        .otherwise(pl.col(value_col))
        .alias(value_col)
    )
    with pytest.raises(ValueError, match="finite"):
        Dataset(
            synthetic.variant_df,
            gene_df=gene_df,
            beta_col="beta",
            se_col="se",
            sample_size=100.0,
        )


def test_rejects_a_repeated_burden_column():
    with pytest.raises(ValueError, match=r"duplicate kernels: \['B:constant'\]"):
        KernelOptions(burden_cols=["constant", "constant"])


def test_gene_annot_matrix_is_writable_in_both_branches(synthetic_dataset):
    """Joint fits normalize gene coefficients in place."""
    for gene_annot_cols in (None, ["constraint_low", "constraint_high"]):
        matrix = Model(
            synthetic_dataset, synthetic_options(gene_annot_cols=gene_annot_cols)
        ).gene_annot_matrix()
        assert matrix.flags.writeable, gene_annot_cols
        matrix /= matrix.mean(axis=0)
