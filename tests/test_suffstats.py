"""Base sufficient statistics, against a dense oracle and their own invariants."""

from __future__ import annotations

from dataclasses import replace

from extension_helpers import extension_designs

import numpy as np
import polars as pl
import pytest
from conftest import SYNTHETIC_MAF_BINS, MAF_BINS

from kasm import Dataset, KernelOptions, Model, fitting
from kasm.suffstats import (
    JKSuffStats,
    SuffStats,
    compute_suffstats,
    pack_symmetric,
    packed_size,
    unpack_symmetric,
)

# ---------------------------------------------------------------------------
# The oracle: build every kernel densely and take the traces literally
# ---------------------------------------------------------------------------


def dense_kernels(layout, burden_wgts, diagonal, partition=None):
    """Every kernel of a gene as an explicit dense matrix, in kernel order.

    Written from the kernel definitions rather than from the reducer: burden
    weights form rank-one outer products, cross is the symmetrized pair, and
    diagonal is a diagonal matrix.
    """
    m = burden_wgts.shape[0] if burden_wgts.size else diagonal.shape[0]
    kernels = [None] * layout.n_kernels
    if partition is None:
        masks = [np.ones(m)]
    else:
        masks = [partition == value for value in np.unique(partition)]
    for position, kernel_index in enumerate(layout.burden_indices):
        column_wgts = burden_wgts[:, position]
        kernels[kernel_index] = sum(
            np.outer(column_wgts * mask, column_wgts * mask) for mask in masks
        )
    for (left, right), kernel_index in zip(layout.cross_pairs, layout.cross_indices):
        left_wgts, right_wgts = burden_wgts[:, left], burden_wgts[:, right]
        kernels[kernel_index] = sum(
            np.outer(left_wgts * mask, right_wgts * mask)
            + np.outer(right_wgts * mask, left_wgts * mask)
            for mask in masks
        )
    for position, kernel_index in enumerate(layout.diagonal_indices):
        kernels[kernel_index] = np.diag(diagonal[:, position])
    assert all(kernel is not None for kernel in kernels)
    assert all(kernel.shape == (m, m) for kernel in kernels)
    return kernels


def dense_oracle(model, gene):
    """Statistics for one gene by literal dense linear algebra."""
    dataset = model.dataset
    rows = dataset.gene_slice(gene)
    scale = model.variant_scale(gene)
    correlation = dataset.load_ldm(gene).toarray()
    residual = np.diag(scale) @ correlation @ np.diag(scale)

    beta = dataset.beta[rows]
    se = dataset.se[rows]
    beta_tilde = beta / (se**2 * dataset.sample_size)

    burden_wgts = model.burden_wgts(gene)
    diagonal = model.diag_kernel_weights(gene)
    partition = (
        model.burden_partition_codes(gene)
        if model.options.within_gene_group_col is not None
        else None
    )
    kernels = dense_kernels(model.layout, burden_wgts, diagonal, partition)

    n = len(kernels)
    trace = np.array([np.trace(kernel @ residual) for kernel in kernels])
    quadform = np.array([beta_tilde @ kernel @ beta_tilde for kernel in kernels])
    pair_trace = np.zeros((n, n))
    for i in range(n):
        left = kernels[i] @ residual
        for j in range(i, n):
            value = np.trace(left @ kernels[j] @ residual)
            pair_trace[i, j] = pair_trace[j, i] = value

    return {
        "trace": trace,
        "quadform": quadform,
        "pair_trace": pair_trace,
        "burden_cov": burden_wgts.T @ residual @ burden_wgts,
        "burden_beta": burden_wgts.T @ beta_tilde,
    }


# A configuration small enough that the oracle's dense pair traces are quick,
# but exercising every kernel kind and the MAF masking.
def small_options(**overrides) -> KernelOptions:
    settings = dict(
        maf_col="af",
        maf_bins=SYNTHETIC_MAF_BINS,
        burden_cols=["constant", "score"],
        burden_by_maf=False,
        independent_col="constant",
        overdispersion_cols=["constant"],
        overdispersion_by_maf=True,
    )
    settings.update(overrides)
    return KernelOptions(**settings)


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(6, 6), (4, 5, 5)])
def test_packed_symmetric_round_trip(shape):
    rng = np.random.default_rng(0)
    matrices = rng.normal(size=shape)
    matrices += matrices.swapaxes(-1, -2)
    n = shape[-1]
    packed = (
        pack_symmetric(matrices)
        if matrices.ndim == 2
        else np.stack([pack_symmetric(matrix) for matrix in matrices])
    )

    assert packed.shape[-1] == packed_size(n) == n * (n + 1) // 2
    np.testing.assert_allclose(unpack_symmetric(packed, n), matrices)


# ---------------------------------------------------------------------------
# Truth: agreement with the dense oracle
# ---------------------------------------------------------------------------


def test_matches_dense_oracle_on_every_synthetic_gene(synthetic_dataset):
    model = Model(synthetic_dataset, small_options())
    suffstats = compute_suffstats(model)
    for position, gene in enumerate(synthetic_dataset.genes):
        expected = dense_oracle(model, gene)
        np.testing.assert_allclose(
            suffstats.trace[position], expected["trace"], rtol=1e-10, atol=1e-18
        )
        np.testing.assert_allclose(
            suffstats.quadform[position], expected["quadform"], rtol=1e-10, atol=1e-18
        )
        np.testing.assert_allclose(
            suffstats.pair_trace(gene), expected["pair_trace"], rtol=1e-9, atol=1e-24
        )
        np.testing.assert_allclose(
            suffstats.burden_cov(gene), expected["burden_cov"], rtol=1e-10, atol=1e-18
        )
        np.testing.assert_allclose(
            suffstats.burden_beta[position],
            expected["burden_beta"],
            rtol=1e-10,
            atol=1e-18,
        )


def test_matches_dense_oracle_with_maf_stratified_burden(synthetic_dataset):
    """The masked-burden path, including structurally empty MAF strata."""
    model = Model(synthetic_dataset, small_options(burden_by_maf=True))
    suffstats = compute_suffstats(model)
    for position, gene in enumerate(synthetic_dataset.genes):
        expected = dense_oracle(model, gene)
        np.testing.assert_allclose(
            suffstats.pair_trace(gene), expected["pair_trace"], rtol=1e-9, atol=1e-24
        )
        np.testing.assert_allclose(
            suffstats.trace[position], expected["trace"], rtol=1e-10, atol=1e-18
        )


def _partitioned_model(synthetic_dataset, *, one_partition=False):
    frame = synthetic_dataset.variant_df().clone()
    if one_partition:
        frame = frame.with_columns(pl.lit("only").alias("exon"))
    else:
        frame = frame.with_columns(
            ((pl.col("gene").cum_count().over("gene") - 1) % 2).alias("exon")
        )
    dataset = Dataset(
        frame,
        gene_df=synthetic_dataset.gene_df(),
        beta_col=synthetic_dataset.beta_col,
        se_col=synthetic_dataset.se_col,
        sample_size=synthetic_dataset.sample_size,
        ldm=synthetic_dataset.ldm_source,
    )
    return Model(
        dataset,
        small_options(within_gene_group_col="exon"),
    )


def test_partitioned_burden_matches_dense_logical_kernel_oracle(
    synthetic_dataset,
):
    model = _partitioned_model(synthetic_dataset)
    suffstats = compute_suffstats(model)

    for position, gene in enumerate(model.dataset.genes):
        expected = dense_oracle(model, gene)
        np.testing.assert_allclose(
            suffstats.trace[position], expected["trace"], rtol=1e-10
        )
        np.testing.assert_allclose(
            suffstats.quadform[position], expected["quadform"], rtol=1e-10
        )
        np.testing.assert_allclose(
            suffstats.pair_trace(gene),
            expected["pair_trace"],
            rtol=1e-9,
            atol=1e-24,
        )


def test_partitioned_extension_matches_dense_oracle_and_roundtrips(
    synthetic_dataset,
    tmp_path,
):
    model = _partitioned_model(synthetic_dataset)
    model.compute_suffstats(extra_enrich_cols=["buried"])
    assembled = model.suffstats.with_burden_extension("buried")
    extended = model._with_burden_column("buried")

    for position, gene in enumerate(model.dataset.genes):
        expected = dense_oracle(extended, gene)
        np.testing.assert_allclose(
            assembled.pair_trace(gene),
            expected["pair_trace"],
            rtol=1e-9,
            atol=1e-24,
        )
        np.testing.assert_allclose(
            assembled.trace[position], expected["trace"], rtol=1e-10
        )
        np.testing.assert_allclose(
            assembled.quadform[position], expected["quadform"], rtol=1e-10
        )

    path = tmp_path / "partitioned-extension.npz"
    model.suffstats.save(path)
    reloaded = Model(model.dataset, model.options)
    reloaded.suffstats = SuffStats.load(path)
    roundtrip = reloaded.suffstats.with_burden_extension("buried")
    np.testing.assert_array_equal(
        roundtrip.pair_trace_upper, assembled.pair_trace_upper
    )
    np.testing.assert_array_equal(roundtrip.trace, assembled.trace)
    np.testing.assert_array_equal(roundtrip.quadform, assembled.quadform)


def test_one_burden_partition_reduces_to_the_unpartitioned_model(
    synthetic_dataset,
):
    ordinary = Model(synthetic_dataset, small_options())
    partitioned = _partitioned_model(synthetic_dataset, one_partition=True)

    ordinary_stats = compute_suffstats(ordinary)
    partitioned_stats = compute_suffstats(partitioned)

    np.testing.assert_allclose(
        partitioned_stats.pair_trace(), ordinary_stats.pair_trace()
    )
    np.testing.assert_allclose(partitioned_stats.trace, ordinary_stats.trace)
    np.testing.assert_allclose(partitioned_stats.quadform, ordinary_stats.quadform)


def test_burden_partition_label_spelling_does_not_change_the_model(
    synthetic_dataset,
):
    numeric = _partitioned_model(synthetic_dataset)
    frame = numeric.dataset.variant_df().clone()
    frame = frame.with_columns(
        pl.col("exon").replace_strict({0: "first", 1: "second"}).alias("exon")
    )
    renamed = Model(
        Dataset(
            frame,
            gene_df=numeric.dataset.gene_df(),
            beta_col=numeric.dataset.beta_col,
            se_col=numeric.dataset.se_col,
            sample_size=numeric.dataset.sample_size,
            ldm=numeric.dataset.ldm_source,
        ),
        numeric.options,
    )

    numeric_stats = compute_suffstats(numeric)
    renamed_stats = compute_suffstats(renamed)

    np.testing.assert_array_equal(
        numeric_stats.pair_trace_upper,
        renamed_stats.pair_trace_upper,
    )


def test_matches_dense_oracle_without_ld(synthetic_dataset_no_ld):
    model = Model(synthetic_dataset_no_ld, small_options())
    suffstats = compute_suffstats(model)
    for position, gene in enumerate(synthetic_dataset_no_ld.genes):
        expected = dense_oracle(model, gene)
        np.testing.assert_allclose(
            suffstats.pair_trace(gene), expected["pair_trace"], rtol=1e-10, atol=1e-24
        )
        np.testing.assert_allclose(
            suffstats.quadform[position], expected["quadform"], rtol=1e-10, atol=1e-18
        )


@pytest.mark.parametrize(
    "options",
    [
        small_options(independent_col=None),
        small_options(overdispersion_cols=None),
        small_options(burden_cols=["constant"]),
        small_options(independent_col=None, overdispersion_cols=None),
        small_options(maf_bins=None, maf_col=None, overdispersion_by_maf=False),
    ],
    ids=["no-indep", "no-ovd", "one-burden", "burden-only", "no-maf-bins"],
)
def test_matches_dense_oracle_across_configurations(synthetic_dataset, options):
    """Every kernel kind may be absent; the index bookkeeping must still hold."""
    model = Model(synthetic_dataset, options)
    suffstats = compute_suffstats(model)
    gene = synthetic_dataset.genes[0]
    expected = dense_oracle(model, gene)
    np.testing.assert_allclose(
        suffstats.pair_trace(gene), expected["pair_trace"], rtol=1e-9, atol=1e-24
    )
    np.testing.assert_allclose(
        suffstats.trace[0], expected["trace"], rtol=1e-10, atol=1e-18
    )
    np.testing.assert_allclose(
        suffstats.quadform[0], expected["quadform"], rtol=1e-10, atol=1e-18
    )


def test_matches_dense_oracle_on_a_real_gene(test_dataset):
    """One real gene, real LD, against the same literal computation."""
    options = KernelOptions(
        maf_col="AF",
        maf_bins=MAF_BINS,
        burden_cols=["constant", "am.pathogenicity"],
        burden_by_maf=False,
        independent_col="constant",
        overdispersion_cols=["constant"],
        overdispersion_by_maf=True,
    )
    model = Model(test_dataset, options)
    smallest = min(
        test_dataset.genes, key=lambda gene: test_dataset.variants_per_gene[gene]
    )
    suffstats = compute_suffstats(model)
    position = test_dataset.genes.index(smallest)
    expected = dense_oracle(model, smallest)

    np.testing.assert_allclose(
        suffstats.trace[position], expected["trace"], rtol=1e-8, atol=0
    )
    np.testing.assert_allclose(
        suffstats.pair_trace(smallest), expected["pair_trace"], rtol=1e-7, atol=0
    )
    np.testing.assert_allclose(
        suffstats.quadform[position], expected["quadform"], rtol=1e-8, atol=0
    )
    np.testing.assert_allclose(
        suffstats.burden_cov(smallest), expected["burden_cov"], rtol=1e-8, atol=0
    )


def test_identity_ld_single_bin_matches_a_hand_calculation(synthetic):
    """Degenerate agreement, written out by hand.

    With identity LD, ``d = 1/(se sqrt(N))`` gives ``R_s = diag(d^2)``, so for a
    single unstratified burden column ``u``:
      trace   = sum_a u_a^2 d_a^2
      quadform = (sum_a u_a b_a)^2
      pair    = (sum_a u_a^2 d_a^2)^2
    and for the identity kernel, trace = sum_a d_a^2.
    """
    dataset = Dataset(
        synthetic.variant_df,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=50_000,
        ldm=None,
    )
    model = Model(
        dataset,
        KernelOptions(burden_cols=["score"], independent_col="constant"),
    )
    suffstats = compute_suffstats(model)

    gene = dataset.genes[0]
    position = dataset.genes.index(gene)
    rows = dataset.gene_slice(gene)
    u = dataset.variant_values("score", gene)
    d_squared = 1.0 / (dataset.se[rows] ** 2 * dataset.sample_size)
    b = dataset.beta[rows] / (dataset.se[rows] ** 2 * dataset.sample_size)

    burden_position = model.layout.burden_indices[0]
    identity_position = model.layout.independent_index
    assert suffstats.trace[position, burden_position] == pytest.approx(
        np.sum(u**2 * d_squared)
    )
    assert suffstats.quadform[position, burden_position] == pytest.approx(
        np.sum(u * b) ** 2
    )
    assert suffstats.pair_trace(gene)[
        burden_position, burden_position
    ] == pytest.approx(np.sum(u**2 * d_squared) ** 2)
    assert suffstats.trace[position, identity_position] == pytest.approx(
        np.sum(d_squared)
    )
    assert suffstats.pair_trace(gene)[
        identity_position, identity_position
    ] == pytest.approx(np.sum(d_squared**2))


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_statistics_do_not_depend_on_input_row_order(synthetic, synthetic_dataset):
    """Per-gene statistics are invariant to row order within each gene block."""
    variant_df = pl.concat(
        [
            synthetic.variant_df.filter(pl.col("gene") == gene).reverse()
            for gene in synthetic_dataset.genes
        ]
    )
    shuffled = Dataset(
        variant_df,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=50_000,
        ldm=str(synthetic.ldm_path),
    )
    reference = compute_suffstats(Model(synthetic_dataset, small_options()))
    other = compute_suffstats(Model(shuffled, small_options()))
    np.testing.assert_allclose(
        other.pair_trace_upper, reference.pair_trace_upper, rtol=1e-12
    )
    np.testing.assert_allclose(other.trace, reference.trace, rtol=1e-12)
    np.testing.assert_allclose(other.quadform, reference.quadform, rtol=1e-12)


# ---------------------------------------------------------------------------
# Save and load
# ---------------------------------------------------------------------------


def test_save_load_round_trip_is_exact(synthetic_dataset, tmp_path):
    model = Model(synthetic_dataset, small_options())
    suffstats = compute_suffstats(model)
    path = tmp_path / "base.npz"
    suffstats.save(path)
    loaded = SuffStats.load(path)

    for name in (
        "pair_trace_upper",
        "trace",
        "quadform",
        "burden_cov_upper",
        "burden_beta",
        "n_variants",
        "total_het",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(suffstats, name))
    assert loaded.genes == suffstats.genes
    assert loaded.kernel_labels == suffstats.kernel_labels
    assert loaded.burden_labels == suffstats.burden_labels


def test_extension_assembly_scatter_supports_noncontiguous_mapping_and_no_genes():
    base = SuffStats(
        pair_trace_upper=np.array([[1.0, 2.0, 3.0]]),
        trace=np.array([[4.0, 5.0]]),
        quadform=np.array([[6.0, 7.0]]),
        burden_cov_upper=np.array([[8.0]]),
        burden_beta=np.array([[9.0]]),
        n_variants=np.array([2]),
        total_het=np.zeros(1),
        genes=("G",),
        kernel_labels=("I:a", "I:b"),
        burden_labels=("u",),
    )
    from kasm.suffstats import ExtSuffStats, ExtraEnrichCols
    from kasm.kernels import KernelLayout, KernelId

    extension = ExtSuffStats(
        pair_trace=np.array([[[10.0, 11.0, 12.0]]]),
        trace=np.array([[13.0]]),
        quadform=np.array([[14.0]]),
        layout=KernelLayout(
            kernel_ids=(KernelId("I", "b"), KernelId("B", "x"), KernelId("I", "a")),
            burden_labels=("x", "u"),
            burden_indices=(1,),
            cross_indices=(),
            cross_pairs=(),
            independent_indices=(0, 2),
            overdispersion_indices=(),
            kernel_group_names=("all",),
            kernel_groups=(0, 0, 0),
        ),
        base_to_extended=np.array([2, 0]),
        payload=ExtraEnrichCols(
            total_hsq_weights=np.zeros((1, 3)),
            annot_hsq_weights=np.zeros((1, 3)),
            annot_het=np.zeros(1),
            column="x",
            burden_cov=np.array([[[15.0, 16.0]]]),
            burden_beta=np.array([[17.0]]),
        ),
    )
    assembled = replace(base, burden_extensions={"x": extension}).with_burden_extension(
        "x"
    )
    np.testing.assert_array_equal(
        assembled.pair_trace(),
        np.array([[[3.0, 10.0, 2.0], [10.0, 11.0, 12.0], [2.0, 12.0, 1.0]]]),
    )
    np.testing.assert_array_equal(assembled.trace, [[5.0, 13.0, 4.0]])
    np.testing.assert_array_equal(assembled.burden_cov(), [[[15.0, 16.0], [16.0, 8.0]]])

    empty_base = replace(
        base,
        **{
            name: getattr(base, name)[:0]
            for name in (
                "pair_trace_upper",
                "trace",
                "quadform",
                "burden_cov_upper",
                "burden_beta",
                "n_variants",
                "total_het",
            )
        },
        genes=(),
    )
    empty_extension = replace(
        extension,
        trace=extension.trace[:0],
        quadform=extension.quadform[:0],
        pair_trace=extension.pair_trace[:0],
        payload=replace(
            extension.payload,
            total_hsq_weights=extension.payload.total_hsq_weights[:0],
            annot_hsq_weights=extension.payload.annot_hsq_weights[:0],
            annot_het=extension.payload.annot_het[:0],
            burden_cov=extension.payload.burden_cov[:0],
            burden_beta=extension.payload.burden_beta[:0],
        ),
    )
    assert replace(
        empty_base, burden_extensions={"x": empty_extension}
    ).with_burden_extension("x").pair_trace_upper.shape == (0, 6)


def test_streamed_jackknife_statistics_match_full_reduction(
    synthetic_dataset, tmp_path
):
    model = Model(synthetic_dataset, small_options())
    full = compute_suffstats(model)
    excluded = [synthetic_dataset.genes[1]]
    model.compute_suffstats()
    reduced = model.compute_jk_suffstats(exclude_genes=excluded, n_jk_blocks=3)
    keep = np.array([gene not in excluded for gene in full.genes])
    block_of = fitting.assign_jackknife_blocks(len(full.genes), 3)
    expected = fitting.annotation_block_sums(
        full.pair_trace(rows=keep),
        full.trace[keep],
        full.quadform[keep],
        model.gene_annot_matrix()[keep],
        block_of[keep],
        n_blocks=int(block_of.max()) + 1,
    )

    # Support is owned by conversion and counted once, independently of moments.
    for block in range(3):
        rows = keep & (block_of == block)
        assert reduced.blocks["ng"][block] == rows.sum()
        assert reduced.blocks["nv"][block] == full.n_variants[rows].sum()
        np.testing.assert_array_equal(
            reduced.blocks["ga1"][block], model.gene_annot_matrix()[rows].sum(axis=0)
        )
    assert reduced.totals["nv"] == full.n_variants[keep].sum()

    assert isinstance(reduced, JKSuffStats)
    assert not hasattr(reduced, "pair_trace")
    for name in expected:
        np.testing.assert_allclose(reduced.blocks[name], expected[name], rtol=1e-12)
    for name in (
        "burden_cov",
        "burden_cov_upper",
        "burden_beta",
        "n_variants",
        "n_burden",
    ):
        assert not hasattr(reduced, name)

    path = tmp_path / "jackknife.npz"
    reduced.save(path)
    with np.load(path, allow_pickle=False) as stored:
        assert "format_version" not in stored
        assert not {"burden_cov_upper", "burden_beta", "n_variants"} & set(stored.files)
    loaded = JKSuffStats.load(
        path,
        exclude_genes=excluded,
        n_jk_blocks=3,
    )
    for name in reduced.blocks:
        np.testing.assert_array_equal(loaded.blocks[name], reduced.blocks[name])
    with pytest.raises(ValueError, match="exclusions do not match"):
        JKSuffStats.load(path, exclude_genes=[])
    with pytest.raises(ValueError, match="n_jk_blocks does not match"):
        JKSuffStats.load(path, n_jk_blocks=2)


def test_jackknife_retains_requested_empty_block_indices(synthetic_dataset):
    n_blocks = synthetic_dataset.n_genes + 2
    reduced = (
        Model(synthetic_dataset, small_options())
        .compute_suffstats()
        .compute_jk_suffstats(exclude_genes=[], n_jk_blocks=n_blocks)
    )

    assert reduced.n_jk_blocks == n_blocks
    assert all(values.shape[0] == n_blocks for values in reduced.blocks.values())
    np.testing.assert_array_equal(
        reduced.live_blocks, np.arange(synthetic_dataset.n_genes)
    )


def test_jackknife_validates_exclusions_and_external_blocks(synthetic_dataset):
    model = Model(synthetic_dataset, small_options())
    with pytest.raises(KeyError, match="excluded units are not in this model"):
        model.compute_suffstats()
        model.compute_jk_suffstats(exclude_genes=["unknown"], n_jk_blocks=3)
    with pytest.raises(ValueError, match="one aligned label per model gene"):
        model.compute_jk_suffstats(
            exclude_genes=[],
            n_jk_blocks=3,
            gene_blocks=np.zeros(synthetic_dataset.n_genes - 1),
        )
    with pytest.raises(ValueError, match=r"labels must be in \[0, [0-9]+\)"):
        model.compute_jk_suffstats(
            exclude_genes=[],
            n_jk_blocks=3,
            gene_blocks=np.full(synthetic_dataset.n_genes, 3),
        )

    blocks = np.arange(synthetic_dataset.n_genes, dtype=np.int16) % 3
    reduced = model.compute_jk_suffstats(
        exclude_genes=[], n_jk_blocks=3, gene_blocks=blocks
    )
    assert reduced.gene_blocks.dtype == np.intp
    np.testing.assert_array_equal(reduced.gene_blocks, blocks)


@pytest.mark.parametrize("partition", [None, "exon"])
@pytest.mark.parametrize("burden", [None, ["constant"]])
def test_retained_extensions_and_each_block_match_dense_attribution(
    synthetic, monkeypatch, tmp_path, partition, burden
):
    """Signed cross terms, weighted diagonals and zero annotation support survive."""
    frame = synthetic.variant_df.with_columns(
        (pl.col("score") - 0.5).alias("signed"),
        (0.2 + pl.col("score")).alias("diagonal"),
    )
    genes = synthetic.gene_df.with_columns(
        (pl.col("gene") == "GENE_A").cast(pl.Float64).alias("a"),
        (pl.col("gene") == "GENE_C").cast(pl.Float64).alias("c"),
    )
    dataset = Dataset(
        frame,
        gene_df=genes,
        beta_col="beta",
        se_col="se",
        sample_size=1000,
        ldm=synthetic.ldm_path,
    )
    model = Model(
        dataset,
        KernelOptions(
            burden_cols=burden,
            independent_col="diagonal",
            overdispersion_cols=["constant"],
            gene_annot_cols=["a", "c"],
            within_gene_group_col=partition,
        ),
    ).compute_suffstats(extra_enrich_cols=["signed", "buried"])
    original = model.suffstats
    expected = {}
    for annotation in original.burden_extensions:
        augmented = model._with_burden_column(annotation)
        expected[annotation] = [dense_oracle(augmented, gene) for gene in model.genes]
        weights = []
        for gene in model.genes:
            kernels = dense_kernels(
                augmented.layout,
                augmented.burden_wgts(gene),
                augmented.diag_kernel_weights(gene),
                augmented.burden_partition_codes(gene),
            )
            diagonal = np.stack([np.diag(kernel) for kernel in kernels])
            diagonal[list(augmented.layout.overdispersion_indices)] = 0
            d2 = augmented.variant_scale(gene) ** 2
            weights.append(
                (
                    diagonal @ d2,
                    diagonal @ (d2 * dataset.variant_values(annotation, gene)),
                )
            )
        extension = original.burden_extensions[annotation]
        assert isinstance(extension, __import__("kasm").ExtSuffStats)
        for position, key in enumerate(("total_hsq_weights", "annot_hsq_weights")):
            np.testing.assert_allclose(
                getattr(extension.payload, key),
                np.stack([w[position] for w in weights]),
                rtol=1e-12,
                atol=1e-16,
            )
        assert extension.payload.total_hsq_weights.ndim == 2
        assert not hasattr(extension.payload, "total_het")
        np.testing.assert_allclose(
            original.total_het,
            [np.sum(model.variant_scale(gene) ** 2) for gene in model.genes],
            rtol=1e-12,
            atol=1e-16,
        )

    path = tmp_path / "retained.npz"
    original.save(path)
    with np.load(path, allow_pickle=False) as stored:
        assert "total_het" in stored
        assert not any(
            key.startswith("burden_extension_") and key.endswith("total_het")
            for key in stored.files
        )
    model.suffstats = SuffStats.load(path)
    monkeypatch.setattr(
        dataset, "load_ldm", lambda *_: pytest.fail("conversion loaded LD")
    )
    monkeypatch.setattr(
        model,
        "_with_burden_column",
        lambda *_: pytest.fail("conversion constructed a Model"),
    )
    blocks = np.array([2, 0, 2, 1, 0])
    excluded = [model.genes[1]]
    jk_stats = model.compute_jk_suffstats(
        exclude_genes=excluded,
        n_jk_blocks=4,
        gene_blocks=blocks,
    )
    for annotation, per_gene in expected.items():
        _, _, selected, attribution = next(
            extension_designs(jk_stats, extra_enrich_cols=[annotation])
        )
        assert selected.total_het is jk_stats.total_het
        assert attribution["total_het"] is jk_stats.total_het
        assert jk_stats.burden_extensions
        assembled = model.suffstats.with_burden_extension(annotation)
        for position, oracle in enumerate(per_gene):
            np.testing.assert_allclose(
                assembled.pair_trace(rows=position),
                oracle["pair_trace"],
                rtol=1e-12,
                atol=1e-16,
            )
        for block in range(4):
            positions = [
                i
                for i, gene in enumerate(model.genes)
                if blocks[i] == block and gene not in excluded
            ]
            np.testing.assert_allclose(
                selected.total_het[block],
                sum(original.total_het[position] for position in positions),
                rtol=1e-12,
                atol=1e-16,
            )
            for key, moment in (
                ("pt", "pair_trace"),
                ("tr", "trace"),
                ("qf", "quadform"),
            ):
                expected_plain = np.zeros_like(selected.blocks[key + "0"][block])
                expected_annot = np.zeros_like(selected.blocks[key + "1"][block])
                for position in positions:
                    expected_plain += per_gene[position][moment]
                    expected_annot += (
                        per_gene[position][moment][..., None]
                        * model.gene_annot_matrix()[position]
                    )
                np.testing.assert_allclose(
                    selected.blocks[key + "0"][block],
                    expected_plain,
                    rtol=1e-12,
                    atol=1e-16,
                )
                np.testing.assert_allclose(
                    selected.blocks[key + "1"][block],
                    expected_annot,
                    rtol=1e-12,
                    atol=1e-16,
                )
            for key in ("total_hsq_weights", "annot_hsq_weights"):
                expected_weights = np.zeros_like(attribution[key][block])
                for position in positions:
                    expected_weights += (
                        getattr(original.burden_extensions[annotation].payload, key)[
                            position, :, None
                        ]
                        * model.gene_annot_matrix()[position]
                    )
                np.testing.assert_allclose(
                    attribution[key][block],
                    expected_weights,
                    rtol=1e-12,
                    atol=1e-16,
                )
        assert selected.blocks["ng"].sum() == len(model.genes) - 1
        np.testing.assert_array_equal(selected.gene_blocks, blocks)
        path = tmp_path / f"{annotation}.npz"
        selected.save(path)
        with np.load(path, allow_pickle=False) as stored:
            assert not any(
                key.startswith("burden_extension_") and key.endswith("total_het")
                for key in stored.files
            )
        loaded = JKSuffStats.load(path)
        for key in selected.blocks:
            np.testing.assert_array_equal(loaded.blocks[key], selected.blocks[key])


def test_jackknife_requires_explicit_gene_preparation(synthetic_dataset, monkeypatch):
    model = Model(synthetic_dataset, small_options())
    original = synthetic_dataset.load_ldm
    calls = []

    def load(gene):
        calls.append(gene)
        return original(gene)

    monkeypatch.setattr(synthetic_dataset, "load_ldm", load)
    with pytest.raises(ValueError, match="no sufficient statistics"):
        model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    assert not calls
    model.compute_suffstats(extra_enrich_cols=["buried"])
    jk_stats = model.compute_jk_suffstats(
        exclude_genes=[],
        n_jk_blocks=2,
    )
    assert calls == model.genes
    assert isinstance(model.suffstats, SuffStats)
    again = model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    assert calls == model.genes
    for key in jk_stats.blocks:
        np.testing.assert_array_equal(jk_stats.blocks[key], again.blocks[key])


@pytest.mark.parametrize("pair_shape", [(10,), (3, 7)])
def test_shape_agnostic_block_sums_match_independent_sums(pair_shape):
    rng = np.random.default_rng(37)
    pair = rng.normal(size=(7, *pair_shape))
    trace = rng.normal(size=(7, 3))
    quad = rng.normal(size=(7, 3))
    annotations = np.array([[1, 0], [0, 1], [0, 0], [0, 1], [1, 0], [0, 0], [1, 0]])
    labels = np.array([2, 0, 2, 0, 2, 2, 0])
    keep = np.array([True, False, True, True, False, True, True])
    blocks = fitting.annotation_block_sums(
        pair[keep],
        trace[keep],
        quad[keep],
        annotations[keep],
        labels[keep],
        n_blocks=4,
    )
    assert set(blocks) == {"pt0", "pt1", "tr0", "tr1", "qf0", "qf1"}
    assert blocks["pt0"].shape == (4, *pair_shape)
    for block in range(4):
        selected = keep & (labels == block)
        for prefix, values in (("pt", pair), ("tr", trace), ("qf", quad)):
            np.testing.assert_allclose(
                blocks[prefix + "0"][block],
                values[selected].sum(axis=0),
                rtol=1e-12,
                atol=1e-16,
            )
            for annotation in range(2):
                rows = selected & (annotations[:, annotation] == 1)
                np.testing.assert_allclose(
                    blocks[prefix + "1"][block, ..., annotation],
                    values[rows].sum(axis=0),
                    rtol=1e-12,
                    atol=1e-16,
                )


def test_conversion_unpacks_only_aggregated_base_blocks(synthetic_dataset, monkeypatch):
    import kasm.suffstats as module

    model = Model(synthetic_dataset, small_options()).compute_suffstats(
        extra_enrich_cols=["buried", "score32"]
    )
    original = module.unpack_symmetric
    shapes = []

    def unpack_blocks(packed, size):
        shapes.append(packed.shape)
        assert packed.shape[0] == 3
        return original(packed, size)

    monkeypatch.setattr(module, "unpack_symmetric", unpack_blocks)
    monkeypatch.setattr(
        SuffStats,
        "pair_trace",
        lambda *_args, **_kwargs: pytest.fail("expanded per-gene base matrices"),
    )
    monkeypatch.setattr(
        module, "_extend_packed", lambda *_: pytest.fail("assembled per-gene extension")
    )
    jk_stats = model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=3)
    assert shapes == [
        (3, packed_size(model.n_kernels)),
        (3, 1, packed_size(model.n_kernels)),
    ]
    for annotation in ("buried", "score32"):
        _, _, selected, attribution = next(
            extension_designs(jk_stats, extra_enrich_cols=[annotation])
        )
        assert selected.blocks["pt0"].shape[0] == 3
    assert len(shapes) == 2
