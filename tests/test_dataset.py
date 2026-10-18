"""Dataset construction, caller ordering, and per-gene LD alignment."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sps
from conftest import write_ldm

import kasm
from kasm.dataset import Dataset, parse_variant_ids
from kasm.ldm import LDM, LDMLoader

# ---------------------------------------------------------------------------
# Variant IDs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant_id",
    [
        "1:1000:A",  # too few fields
        "1:1000:A:G:extra",  # too many fields
        "1:abc:A:G",  # non-numeric position
        "1:1.9:A:G",  # fractional position
        "1:1e3:A:G",  # non-canonical scientific notation
        "1:001:A:G",  # non-canonical leading zeros
        "1:0:A:G",  # positions are one-based
        "1:-4:A:G",  # negative position
        "1:nan:A:G",  # non-finite position
        "1:inf:A:G",  # non-finite position
        "99:1000:A:G",  # unknown chromosome
        "chr1:1000:A:G",  # chromosome prefix is not canonical
        "x:1000:A:G",  # chromosome case is canonical
        "M:1000:A:G",  # mitochondrial chromosome is MT
        "1:1000::G",  # empty reference allele
        "1:1000:A:",  # empty alternate allele
        None,  # null ID
    ],
)
def test_variant_id_entry_points_reject_the_same_malformed_ids(variant_id):
    with pytest.raises(ValueError):
        parse_variant_ids(["1:1000:A:G", variant_id])
    with pytest.raises(ValueError, match="not canonical"):
        _dataset_from_rows([("1:1000:A:G", "G"), (variant_id, "G")])


# ---------------------------------------------------------------------------
# Caller ordering
# ---------------------------------------------------------------------------


def test_public_variant_sorting_is_canonical_stable_and_collision_safe():
    frame = pl.DataFrame(
        {
            "variant": ["X:2:A:G", "2:10:A:T", "2:3:C:A", "MT:1:G:A", "2:3:C:A"],
            "gene": ["GX", "G2", "G2", "GMT", "G3"],
            "row": [0, 1, 2, 3, 4],
            "__kasm_sort_pos": [10, 11, 12, 13, 14],
        }
    )

    ordered = kasm.sort_variant_df(frame)
    assert ordered["row"].to_list() == [2, 4, 1, 0, 3]
    assert ordered["__kasm_sort_pos"].to_list() == [12, 14, 11, 10, 13]
    gene_ordered = kasm.sort_variant_df(frame, gene_col="gene")
    assert gene_ordered["gene"].to_list() == ["G2", "G2", "G3", "GX", "GMT"]
    tied = pl.DataFrame({"variant": ["1:1:A:G", "1:1:A:G"], "gene": ["Z", "A"]})
    assert kasm.sort_variant_df(tied, gene_col="gene")["gene"].to_list() == ["A", "Z"]
    with pytest.raises(ValueError, match="canonical"):
        kasm.sort_variant_df(frame.with_columns(pl.lit("chr2:3:C:A").alias("variant")))


def _dataset_from_rows(rows, **kwargs):
    variant_df = pl.DataFrame(
        {
            "variant": [row[0] for row in rows],
            "gene": [row[1] for row in rows],
            "beta": np.zeros(len(rows)),
            "se": np.ones(len(rows)),
        },
        schema={
            "variant": pl.String,
            "gene": pl.String,
            "beta": pl.Float64,
            "se": pl.Float64,
        },
    )
    return Dataset(
        variant_df, beta_col="beta", se_col="se", sample_size=100.0, **kwargs
    )


def test_dataset_retains_caller_gene_and_variant_order():
    dataset = _dataset_from_rows(
        [
            ("11:10:A:G", "LATE"),
            ("2:70:A:G", "MIDDLE"),
            ("2:50:A:G", "MIDDLE"),
            ("X:5:A:G", "LAST"),
            ("1:99:A:G", "FIRST"),
        ]
    )
    assert dataset.genes == ["LATE", "MIDDLE", "LAST", "FIRST"]
    assert dataset.variants_per_gene == {
        "LATE": 1,
        "MIDDLE": 2,
        "LAST": 1,
        "FIRST": 1,
    }
    assert dataset.variant_df("MIDDLE")["variant"].to_list() == [
        "2:70:A:G",
        "2:50:A:G",
    ]

    single_gene = _dataset_from_rows(
        [
            ("1:10:A:T", "G"),
            ("1:10:A:C", "G"),
            ("1:2:G:A", "G"),
            ("1:10:AA:C", "G"),
        ]
    )
    assert single_gene.variant_df()["variant"].to_list() == [
        "1:10:A:T",
        "1:10:A:C",
        "1:2:G:A",
        "1:10:AA:C",
    ]


def test_dataset_rejects_noncontiguous_gene_rows(test_variant_df, test_dataset):
    rng = np.random.default_rng(0)
    shuffled = test_variant_df[rng.permutation(len(test_variant_df))]
    with pytest.raises(ValueError, match="rows for each gene must be contiguous"):
        Dataset(
            shuffled,
            beta_col="BETA",
            se_col="SE",
            sample_size=test_dataset.sample_size,
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_variants_assigned_to_multiple_genes():
    with pytest.raises(ValueError, match="duplicate variant IDs"):
        _dataset_from_rows([("1:1:A:G", "G1"), ("1:1:A:G", "G2")])


def test_rejects_empty_variant_df():
    with pytest.raises(ValueError, match="empty"):
        _dataset_from_rows([])


@pytest.mark.parametrize(
    "column,values,message",
    [
        ("se", [0.0], "positive and finite"),
        ("se", [np.nan], "positive and finite"),
        ("beta", [np.inf], "finite"),
    ],
)
def test_rejects_unusable_summary_statistics(column, values, message):
    variant_df = pl.DataFrame(
        {"variant": ["1:1:A:G"], "gene": ["G"], "beta": [0.0], "se": [1.0]}
    )
    variant_df = variant_df.with_columns(pl.Series(column, values))
    with pytest.raises(ValueError, match=message):
        Dataset(variant_df, beta_col="beta", se_col="se", sample_size=100.0)


def test_rejects_missing_summary_statistic_column():
    with pytest.raises(KeyError, match="absent_col"):
        Dataset(
            pl.DataFrame(
                {"variant": ["1:1:A:G"], "gene": ["G"], "beta": [0.0], "se": [1.0]}
            ),
            beta_col="beta",
            se_col="absent_col",
            sample_size=100.0,
        )


@pytest.mark.parametrize("sample_size", [0.0, -1.0, np.nan])
def test_rejects_unusable_sample_size(sample_size):
    with pytest.raises(ValueError, match="sample_size"):
        Dataset(
            pl.DataFrame(
                {"variant": ["1:1:A:G"], "gene": ["G"], "beta": [0.0], "se": [1.0]}
            ),
            beta_col="beta",
            se_col="se",
            sample_size=sample_size,
        )


def test_gene_df_may_cover_extra_genes_and_is_aligned_by_name():
    gene_df = pl.DataFrame(
        {"gene": ["LAST", "FIRST", "MIDDLE", "UNUSED"], "annot": [3.0, 1.0, 2.0, 99.0]}
    )
    dataset = _dataset_from_rows(
        [("1:1:A:G", "FIRST"), ("2:1:A:G", "MIDDLE"), ("3:1:A:G", "LAST")],
        gene_df=gene_df,
    )
    assert dataset.gene_df()["gene"].to_list() == ["FIRST", "MIDDLE", "LAST"]
    np.testing.assert_array_equal(dataset.gene_values("annot"), [1.0, 2.0, 3.0])


def test_gene_df_defines_inner_gene_overlap_without_reordering_variants(caplog):
    gene_df = pl.DataFrame(
        {"gene": ["MIDDLE", "LATE", "UNUSED"], "annot": [2.0, 1.0, 99.0]}
    )
    with caplog.at_level(logging.INFO, logger="kasm"):
        dataset = _dataset_from_rows(
            [
                ("3:1:A:G", "LATE"),
                ("2:1:A:G", "MISSING"),
                ("1:10:A:G", "MIDDLE"),
                ("1:2:A:G", "MIDDLE"),
            ],
            gene_df=gene_df,
        )

    assert dataset.variant_df().select("variant", "gene").rows() == [
        ("3:1:A:G", "LATE"),
        ("1:10:A:G", "MIDDLE"),
        ("1:2:A:G", "MIDDLE"),
    ]
    assert dataset.genes == ["LATE", "MIDDLE"]
    assert dataset.variants_per_gene == {"LATE": 1, "MIDDLE": 2}
    assert dataset.gene_df()["gene"].to_list() == ["LATE", "MIDDLE"]
    np.testing.assert_array_equal(dataset.gene_values("annot"), [1.0, 2.0])
    assert "event=gene_df_alignment" in caplog.text
    assert "input_genes=3 retained_genes=2 dropped_genes=1" in caplog.text
    assert "input_variants=4 retained_variants=3 dropped_variants=1" in caplog.text
    assert "extra_gene_df_genes=1" in caplog.text


def test_rejects_gene_df_without_overlapping_genes():
    gene_df = pl.DataFrame({"gene": ["OTHER"], "annot": [1.0]})
    with pytest.raises(ValueError, match="no overlapping genes"):
        _dataset_from_rows([("1:1:A:G", "FIRST")], gene_df=gene_df)


def test_rejects_non_numeric_gene_df_column():
    gene_df = pl.DataFrame({"gene": ["FIRST"], "label": ["x"]})
    with pytest.raises(ValueError, match="numeric"):
        _dataset_from_rows([("1:1:A:G", "FIRST")], gene_df=gene_df)


# ---------------------------------------------------------------------------
# Column access
# ---------------------------------------------------------------------------


def test_returned_values_are_read_only_whatever_the_input_dtype(
    synthetic_dataset, test_dataset
):
    """Input numeric dtypes normalize once; reducers always see float64."""
    assert synthetic_dataset.variant_df()["score32"].dtype == pl.Float64
    assert synthetic_dataset.variant_df()["carriers"].dtype == pl.Float64
    dtypes = {test_dataset.variant_df()[col].dtype for col in ("BETA", "AF", "N")}
    assert dtypes == {pl.Float64}

    for values in (
        synthetic_dataset.variant_values("beta"),
        synthetic_dataset.variant_values("score32"),
        synthetic_dataset.variant_values("carriers"),
        synthetic_dataset.variant_values(["score32", "carriers"]),
        synthetic_dataset.variant_values("score", synthetic_dataset.genes[0]),
        synthetic_dataset.gene_values("constraint_low"),
        synthetic_dataset.gene_values(["constraint_low", "constraint_high"]),
        synthetic_dataset.beta,
        synthetic_dataset.se,
        test_dataset.variant_values("BETA"),
        test_dataset.variant_values("N"),
        test_dataset.beta,
    ):
        assert not values.flags.writeable
        assert values.dtype == np.float64
        with pytest.raises(ValueError, match="read-only"):
            values[0] = 1.0


def test_column_arrays_are_zero_copy_views_of_polars_buffers(synthetic_dataset):
    """Dataset tables own the only numeric buffers."""
    for values in (
        synthetic_dataset.beta,
        synthetic_dataset.variant_values("score"),
        synthetic_dataset.gene_values("constraint_low"),
    ):
        assert values.base is not None
        assert values.flags.owndata is False
    assert np.shares_memory(
        synthetic_dataset.variant_values("score"),
        synthetic_dataset.variant_df()["score"].to_numpy(allow_copy=False),
    )
    assert np.shares_memory(
        synthetic_dataset.gene_values("constraint_low"),
        synthetic_dataset.gene_df()["constraint_low"].to_numpy(allow_copy=False),
    )


def test_dataset_owns_normalized_input_buffers():
    beta = np.array([0.1, 0.2])
    score = np.array([1, 2], dtype=np.int32)
    dataset = Dataset(
        pl.DataFrame(
            {
                "variant": ["1:1:A:G", "1:2:C:T"],
                "gene": ["G", "G"],
                "beta": beta,
                "se": [0.01, 0.02],
                "score": score,
            }
        ),
        beta_col="beta",
        se_col="se",
        sample_size=100.0,
    )
    beta[:] = 9.0
    score[:] = 9
    np.testing.assert_array_equal(dataset.beta, [0.1, 0.2])
    np.testing.assert_array_equal(dataset.variant_values("score"), [1.0, 2.0])


def test_polars_frames_are_independent_snapshots(synthetic_dataset):
    expected_variant = synthetic_dataset.variant_values("score").copy()
    expected_gene = synthetic_dataset.gene_values("constraint_low").copy()
    variants = synthetic_dataset.variant_df()
    genes = synthetic_dataset.gene_df()
    variants[0, "score"] = 99.0
    genes[0, "constraint_low"] = 99.0

    np.testing.assert_array_equal(
        synthetic_dataset.variant_values("score"), expected_variant
    )
    np.testing.assert_array_equal(
        synthetic_dataset.gene_values("constraint_low"), expected_gene
    )
    assert synthetic_dataset.variant_df()[0, "score"] != 99.0
    assert synthetic_dataset.gene_df()[0, "constraint_low"] != 99.0
    with pytest.raises(AttributeError):
        synthetic_dataset.sample_size = 1
    with pytest.raises(AttributeError):
        synthetic_dataset.ldm_source = None


def test_boolean_annotations_are_numeric_dataset_columns(synthetic):
    variants = synthetic.variant_df.with_columns((pl.col("score") > 0.5).alias("flag"))
    genes = synthetic.gene_df.with_columns(
        (pl.col("constraint_low") > 0).alias("gene_flag")
    )
    dataset = Dataset(
        variants,
        gene_df=genes,
        beta_col="beta",
        se_col="se",
        sample_size=50_000,
    )

    np.testing.assert_array_equal(
        dataset.variant_values("flag"),
        (dataset.variant_values("score") > 0.5).astype(float),
    )
    np.testing.assert_array_equal(
        dataset.gene_values("gene_flag"),
        dataset.gene_values("constraint_low"),
    )


# ---------------------------------------------------------------------------
# LD: the stored-panel convention
# ---------------------------------------------------------------------------


def test_panel_ld_equals_correlation_of_the_genotypes_it_came_from(
    synthetic, synthetic_dataset
):
    """The current-format panel reproduces its source genotype correlations."""
    for gene in ("GENE_A", "GENE_C"):
        variant_ids = synthetic_dataset.variant_df(gene)["variant"].to_list()
        got = synthetic_dataset.load_ldm(gene).toarray()
        np.testing.assert_allclose(
            got, synthetic.genotype_correlation(variant_ids), rtol=0, atol=1e-12
        )


def test_uncovered_variants_get_an_identity_row(synthetic_dataset):
    """A panel-monomorphic variant carries no LD, so it must not be dropped."""
    variant_ids = synthetic_dataset.variant_df("GENE_D")["variant"].to_list()
    ldm = synthetic_dataset.load_ldm("GENE_D")
    dense = ldm.toarray()

    np.testing.assert_allclose(dense, np.eye(len(variant_ids)), atol=1e-12)
    assert ldm.n_covered == len(variant_ids) - 1


def test_off_panel_variant_between_correlated_variants(synthetic, synthetic_dataset):
    """The scatter-index test: LD must land either side of an absent variant.

    GENE_E's middle variant is in no panel while its neighbours are strongly
    correlated, so writing the covered block at the wrong offsets shows up.
    """
    variant_ids = synthetic_dataset.variant_df("GENE_E")["variant"].to_list()
    assert variant_ids == ["3:1000:A:G", "3:1500:C:T", "3:2000:G:C"]

    ldm = synthetic_dataset.load_ldm("GENE_E")
    dense = ldm.toarray()
    assert ldm.n_covered == 2

    # The absent variant is uncorrelated with everything; its neighbours are not.
    np.testing.assert_allclose(dense[1], [0.0, 1.0, 0.0], atol=1e-12)
    flanking = synthetic.genotype_correlation(["3:1000:A:G", "3:2000:G:C"])
    assert abs(flanking[0, 1]) > 0.1
    np.testing.assert_allclose(dense[0, 2], flanking[0, 1], atol=1e-12)
    np.testing.assert_allclose(dense[0, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(dense[2, 2], 1.0, atol=1e-12)


def test_extra_panel_variants_are_excluded(synthetic, synthetic_dataset, tmp_path):
    """GENE_C's panel holds a variant that variant_df does not.

    Its presence must not perturb the LD among the requested variants, which is
    checked by rewriting the panel without it.
    """
    assert "2:6500:C:A" in synthetic.panel_variants["GENE_C"]
    query = synthetic_dataset.variant_df("GENE_C")["variant"].to_list()
    with_extra = synthetic_dataset.load_ldm("GENE_C").toarray()
    assert with_extra.shape == (len(query),) * 2

    columns = [synthetic.variant_order.index(vid) for vid in query]
    path = write_ldm(
        tmp_path / "panel.ldm",
        {"GENE_C": (query, synthetic.genotypes[:, columns])},
    )
    without_extra = LDMLoader(path).load_gene_ldm("GENE_C", query).toarray()
    np.testing.assert_allclose(with_extra, without_extra, atol=1e-12)


def test_panel_row_order_does_not_matter(synthetic, tmp_path):
    """GENE_C's panel is shuffled; a sorted rewrite must give the same LD."""
    panel = synthetic.panel_variants["GENE_C"]
    assert panel != sorted(panel), "fixture should ship a shuffled panel"

    sorted_panel = sorted(panel)
    columns = [synthetic.variant_order.index(vid) for vid in sorted_panel]
    path = write_ldm(
        tmp_path / "panel.ldm",
        {"GENE_C": (sorted_panel, synthetic.genotypes[:, columns])},
    )

    query = [
        vid for vid in synthetic.variant_order if vid in set(panel) - {"2:6500:C:A"}
    ]
    shuffled = LDMLoader(synthetic.ldm_path).load_gene_ldm("GENE_C", query)
    resorted = LDMLoader(path).load_gene_ldm("GENE_C", query)
    np.testing.assert_allclose(shuffled.toarray(), resorted.toarray(), atol=1e-12)


def test_query_order_permutes_the_result(synthetic):
    """Subsetting and reordering follow the requested variant order."""
    query = ["1:1000:A:G", "1:2000:C:T", "1:3000:G:A"]
    loader = LDMLoader(synthetic.ldm_path)
    forward = loader.load_gene_ldm("GENE_A", query).toarray()
    reverse = loader.load_gene_ldm("GENE_A", query[::-1]).toarray()
    np.testing.assert_allclose(forward, reverse[::-1, ::-1], atol=1e-12)

    single = loader.load_gene_ldm("GENE_A", ["1:2000:C:T"]).toarray()
    np.testing.assert_allclose(single, [[1.0]], atol=1e-12)


def test_missing_gene_is_an_error(synthetic, synthetic_dataset):
    with pytest.raises(KeyError, match="not in the LD panel"):
        LDMLoader(synthetic.ldm_path).load_gene_ldm("NOT_A_GENE", ["1:1000:A:G"])
    with pytest.raises(KeyError, match="not in this dataset"):
        synthetic_dataset.load_ldm("NOT_A_GENE")


def test_dataset_gene_absent_from_panel_warns_and_uses_identity(
    synthetic_ldm_path, caplog
):
    dataset = _dataset_from_rows(
        [("1:1000:A:G", "NOT_IN_PANEL"), ("1:2000:C:T", "NOT_IN_PANEL")],
        ldm=synthetic_ldm_path,
    )

    with caplog.at_level(logging.WARNING, logger="kasm"):
        np.testing.assert_array_equal(
            dataset.load_ldm("NOT_IN_PANEL").toarray(), np.eye(2)
        )
    assert "event=ldm_gene_missing" in caplog.text
    assert "gene=NOT_IN_PANEL" in caplog.text


def test_duplicate_requested_ids_are_an_error(synthetic):
    with pytest.raises(ValueError, match="duplicate variant IDs requested"):
        LDMLoader(synthetic.ldm_path).load_gene_ldm(
            "GENE_A", ["1:1000:A:G", "1:1000:A:G"]
        )


def test_empty_panel_falls_back_to_identity(synthetic, tmp_path):
    path = write_ldm(tmp_path / "panel.ldm", {"GENE": ([], np.empty((2, 0)))})
    ldm = LDMLoader(path).load_gene_ldm("GENE", ["1:1000:A:G", "1:2000:C:T"])
    np.testing.assert_array_equal(ldm.toarray(), np.eye(2))
    assert ldm.n_covered == 0


def test_closing_a_dataset_releases_the_panel(synthetic, synthetic_ldm_path):
    with Dataset(
        synthetic.variant_df,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=100.0,
        ldm=str(synthetic_ldm_path),
    ) as dataset:
        dataset.load_ldm("GENE_A")
        loader = dataset._ldm_loader
        assert loader._zfile.fp is not None
    # The archive handle itself is released, not merely forgotten.
    assert loader._zfile.fp is None
    assert dataset._ldm_loader is None
    # Still usable afterwards; the loader is simply reopened.
    assert dataset.load_ldm("GENE_A").shape == (3, 3)


def test_no_ldm_source_gives_exact_identity(synthetic_dataset_no_ld):
    for gene in synthetic_dataset_no_ld.genes:
        m = synthetic_dataset_no_ld.variants_per_gene[gene]
        np.testing.assert_array_equal(
            synthetic_dataset_no_ld.load_ldm(gene).toarray(), np.eye(m)
        )


def test_ldm_scale_matches_dense_diagonal_multiplication():
    ldm = LDM(
        sps.csr_matrix([[2.0, 0.5, 0.0], [0.5, 3.0, 1.0], [0.0, 1.0, 4.0]]),
        np.array([0.1, 0.2, 0.3]),
    )
    d = np.array([2.0, 3.0, 5.0])
    np.testing.assert_allclose(
        ldm.scale(d).toarray(), np.diag(d) @ ldm.toarray() @ np.diag(d)
    )


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------


def test_real_panel_ld_is_valid_for_every_gene(test_dataset):
    for gene in test_dataset.genes:
        ldm = test_dataset.load_ldm(gene)
        dense = ldm.toarray()
        assert dense.shape == (test_dataset.variants_per_gene[gene],) * 2
        np.testing.assert_allclose(dense, dense.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(dense), 1.0, atol=1e-10)
        # Cholesky of A + 1e-8 I succeeds exactly when min eigenvalue > -1e-8.
        np.linalg.cholesky(dense + 1e-8 * np.eye(len(dense)))
