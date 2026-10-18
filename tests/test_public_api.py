"""Simulation-facing public API composition over the numerical estimators."""

import numpy as np
import polars as pl
import pytest
from conftest import MAF_BINS

import kasm


def _simple_model(
    dataset,
    *,
    burden_cols=("constant",),
    extra_enrich_cols=(),
    overdispersion=False,
    by_maf=False,
    independent=False,
    maf_bins=False,
    gene_annot_cols=None,
):
    return kasm.Model(
        dataset,
        kasm.KernelOptions(
            maf_col="AF",
            maf_bins=MAF_BINS if by_maf or maf_bins else None,
            burden_cols=burden_cols,
            burden_by_maf=by_maf,
            independent_col="constant" if independent else None,
            overdispersion_cols=["constant"] if overdispersion else None,
            gene_annot_cols=gene_annot_cols,
        ),
    ).compute_suffstats(extra_enrich_cols=extra_enrich_cols)


def test_ordinary_fixed_gene_coef_are_exact_in_point_and_jackknife(
    test_dataset,
):
    annotations = [f"oebin{i}" for i in range(1, 6)]
    model = _simple_model(test_dataset, gene_annot_cols=annotations)
    supplied = dict(zip(annotations, [2.0, 4.0, 6.0, 8.0, 10.0], strict=True))
    model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)

    fit = kasm.fit_random_effects(
        model,
        exclude_genes=[],
        fix_gene_coef=True,
        init_gene_coef=supplied,
        n_jk_blocks=2,
    )

    expected = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    np.testing.assert_array_equal(fit.gene_coef[:, 0], expected)
    np.testing.assert_array_equal(
        fit.gene_coef_jk[:, :, 0], np.broadcast_to(expected, (2, 5))
    )
    assert fit.n_iter == 0
    assert fit.n_iter_jk.tolist() == [0, 0]

    # Independent conditional moment solve on each retained gene set.
    stats = model.suffstats
    scale = model.gene_annot_matrix() @ expected
    blocks = kasm.fitting.assign_jackknife_blocks(len(stats.genes), 2)
    for block in (None, 0, 1):
        keep = np.ones(len(scale), dtype=bool) if block is None else blocks != block
        left = np.einsum("g,gij->ij", scale[keep] ** 2, stats.pair_trace()[keep])
        trace = np.einsum("g,gk->k", scale[keep], stats.trace[keep])
        right = np.einsum("g,gk->k", scale[keep], stats.quadform[keep])
        theta = np.linalg.solve(left, right - trace / model.dataset.sample_size)
        np.testing.assert_allclose(
            fit.kernel_coef if block is None else fit.kernel_coef_jk[block],
            theta,
            rtol=1e-10,
        )


@pytest.mark.parametrize("max_iter", [1, 200])
def test_single_and_one_group_have_identical_point_and_delete_fits(
    test_dataset, max_iter
):
    model = _simple_model(
        test_dataset,
        gene_annot_cols=[f"oebin{i}" for i in range(1, 6)],
        overdispersion=True,
    )
    model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    ordinary = kasm.fit_random_effects(
        model, exclude_genes=[], n_jk_blocks=2, max_iter=max_iter
    )
    grouped = kasm.fit_random_effects(
        model,
        exclude_genes=[],
        kernel_groups={"all": ["constant"]},
        n_jk_blocks=2,
        max_iter=max_iter,
    )
    for field in (
        "kernel_coef",
        "gene_coef",
        "variance",
        "hsq",
        "kernel_coef_jk",
        "gene_coef_jk",
        "variance_jk",
        "hsq_jk",
        "kernel_coef_se",
        "variance_se",
        "hsq_se",
        "converged",
        "n_iter",
        "converged_jk",
        "n_iter_jk",
        "residual",
    ):
        np.testing.assert_array_equal(getattr(ordinary, field), getattr(grouped, field))


def test_nonconverged_point_initializes_every_delete_fit(test_dataset, monkeypatch):
    model = _simple_model(
        test_dataset,
        gene_annot_cols=[f"oebin{i}" for i in range(1, 6)],
        overdispersion=True,
    )
    original = kasm.fitting.alternating_fit
    starts = []

    def record(inputs, **kwargs):
        starts.append(kwargs["init_gene_coef"])
        return original(inputs, **kwargs)

    monkeypatch.setattr(kasm.fitting, "alternating_fit", record)
    model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    fit = kasm.fit_random_effects(model, exclude_genes=[], n_jk_blocks=2, max_iter=1)
    assert not fit.converged
    assert starts[0] is None
    for start in starts[1:]:
        np.testing.assert_array_equal(start[0], fit.gene_coef)
    assert len(starts) == 3
    assert np.isfinite(fit.hsq_se)
    assert fit.hsq_se == pytest.approx(abs(fit.hsq_jk[0] - fit.hsq_jk[1]) / 2)


@pytest.mark.parametrize(
    ("gene_coef", "message"),
    [
        ({"oebin1": 1.0}, "keys must match"),
        (
            {**{f"oebin{i}": 1.0 for i in range(1, 6)}, "extra": 3.0},
            "keys must match",
        ),
        (
            {**{f"oebin{i}": 1.0 for i in range(1, 6)}, "oebin1": np.inf},
            "must be finite",
        ),
        (
            {**{f"oebin{i}": 1.0 for i in range(1, 6)}, "oebin1": -1.0},
            "must be nonnegative",
        ),
    ],
)
def test_ordinary_fixed_gene_coef_validate_contract(test_dataset, gene_coef, message):
    model = _simple_model(
        test_dataset, gene_annot_cols=[f"oebin{i}" for i in range(1, 6)]
    )
    model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=1)
    with pytest.raises(ValueError, match=message):
        kasm.fit_random_effects(
            model,
            exclude_genes=[],
            fix_gene_coef=True,
            init_gene_coef=gene_coef,
            n_jk_blocks=1,
        )

    with pytest.raises(ValueError, match="requires init_gene_coef"):
        kasm.fit_random_effects(
            model, exclude_genes=[], fix_gene_coef=True, n_jk_blocks=1
        )


def test_empty_fixed_gene_set_still_uses_the_standard_mixed_api(test_dataset):
    model = _simple_model(test_dataset, overdispersion=True)
    model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    random_fit = kasm.fit_random_effects(model, exclude_genes=[], n_jk_blocks=2)
    fixed_fit = kasm.fit_fixed_effects(model, include_genes=[], method="burden")
    mixed_fit = kasm.combine_mixed_effects(
        random_fit=random_fit,
        fixed_fit=fixed_fit,
    )

    assert fixed_fit.n_genes == 0
    assert fixed_fit.hsq == 0.0
    assert mixed_fit[0] == random_fit.hsq
    assert mixed_fit[1] == pytest.approx(random_fit.hsq_se)


def test_public_fit_maf_components_close_to_unchanged_totals(test_dataset):
    model = _simple_model(
        test_dataset,
        burden_cols=("constant", "am.pathogenicity"),
        by_maf=True,
        independent=True,
    )
    model.compute_gene_pvalue(method="burden_1d")
    genes = model.dataset.genes[:2]
    model.compute_jk_suffstats(exclude_genes=genes, n_jk_blocks=2)
    random = kasm.fit_random_effects(model, exclude_genes=genes, n_jk_blocks=2)
    fixed = kasm.fit_fixed_effects(model, include_genes=genes, method="burden")

    assert random.maf_components["maf_bin"].to_list() == list(MAF_BINS)
    assert random.maf_components["hsq"].sum() == pytest.approx(
        random.hsq, rel=1e-9, abs=1e-11
    )
    random_maf_jk = np.asarray(random.maf_components["hsq_jk"].to_list())
    assert random_maf_jk.shape == (len(MAF_BINS), random.n_jk_blocks)
    np.testing.assert_allclose(
        random_maf_jk.sum(axis=0), random.hsq_jk, rtol=1e-9, atol=1e-11
    )
    np.testing.assert_allclose(
        random.maf_components["hsq_se"],
        kasm.jackknife_se(random_maf_jk.T),
    )
    assert fixed.maf_components["hsq"].sum() == pytest.approx(
        fixed.hsq, rel=1e-9, abs=1e-11
    )


@pytest.mark.parametrize("partition", [None, "exon"])
def test_public_random_maf_components_expand_unstratified_burden(
    synthetic_dataset, monkeypatch, partition
):
    from dataclasses import replace

    from kasm.api import _maf_trace_data, _random_maf_components

    model = kasm.Model(
        synthetic_dataset,
        kasm.KernelOptions(
            maf_col="af",
            maf_bins={"low": (0, 0.01), "high": (0.01, 1)},
            burden_cols=["constant", "score"],
            burden_by_maf=False,
            overdispersion_cols=["constant"],
            overdispersion_by_maf=True,
            within_gene_group_col=partition,
        ),
    ).compute_suffstats()
    model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    fit = kasm.fit_random_effects(model, exclude_genes=[], n_jk_blocks=2)
    expanded = kasm.Model(
        model.dataset, replace(model.options, burden_by_maf=True)
    ).compute_suffstats()
    expected = _random_maf_components(
        model,
        fit,
        [],
        fit.hsq,
        expected_hsq_jk=fit.hsq_jk,
        trace_data=(expanded, expanded.suffstats.trace),
    )

    def no_full_reduction(*args, **kwargs):
        pytest.fail("MAF reporting must not compute full sufficient statistics")

    monkeypatch.setattr(kasm.Model, "compute_suffstats", no_full_reduction)
    np.testing.assert_allclose(
        _maf_trace_data(model, model.jk_suffstats)[1].sum(axis=0)[:, 0],
        expanded.suffstats.trace.sum(axis=0),
        rtol=1e-12,
        atol=1e-18,
    )
    np.testing.assert_allclose(
        fit.maf_components.select("hsq", "hsq_se").to_numpy(),
        expected.select("hsq", "hsq_se").to_numpy(),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        fit.maf_components["hsq_jk"].to_list(),
        expected["hsq_jk"].to_list(),
        rtol=1e-10,
        atol=1e-12,
    )


def test_unstratified_maf_reporting_preserves_custom_compact_blocks(test_dataset):
    model = _simple_model(test_dataset, maf_bins=True)
    stats = model.compute_jk_suffstats(
        exclude_genes=[],
        n_jk_blocks=2,
        gene_blocks=np.arange(len(model.genes)) % 2,
    )
    fit = kasm.fit_random_effects(stats, exclude_genes=[], n_jk_blocks=2)

    from kasm.random_effects import _assemble_random_reports

    fit = _assemble_random_reports(fit, model, stats)
    np.testing.assert_array_equal(fit.gene_blocks, stats.gene_blocks)

    assert fit.maf_components["hsq"].sum() == pytest.approx(
        fit.hsq, rel=1e-9, abs=1e-11
    )
    np.testing.assert_allclose(
        np.asarray(fit.maf_components["hsq_jk"].to_list()).sum(axis=0), fit.hsq_jk
    )


def test_public_random_maf_components_use_compact_partitioned_traces(
    synthetic_dataset,
):
    excluded = synthetic_dataset.genes[:1]
    model = kasm.Model(
        synthetic_dataset,
        kasm.KernelOptions(
            maf_col="af",
            maf_bins={"all": (0, 1)},
            burden_cols=["constant"],
            burden_by_maf=True,
            within_gene_group_col="exon",
            gene_annot_cols=["constraint_low", "constraint_high"],
        ),
    )
    model.compute_suffstats()
    stats = model.compute_jk_suffstats(exclude_genes=excluded, n_jk_blocks=2)
    fit = kasm.fit_random_effects(stats, exclude_genes=excluded, n_jk_blocks=2)

    from kasm.random_effects import _assemble_random_reports

    fit = _assemble_random_reports(fit, model, stats)

    assert fit.maf_components["hsq"].sum() == pytest.approx(
        fit.hsq, rel=1e-9, abs=1e-11
    )
    assert fit.gene_annot_components["hsq"].sum() == pytest.approx(
        fit.hsq, rel=1e-9, abs=1e-11
    )
    assert all(
        len(values) == fit.n_jk_blocks for values in fit.maf_components["hsq_jk"]
    )


def test_public_random_gene_annot_components_are_additive(test_dataset):
    annotations = [f"oebin{i}" for i in range(1, 6)]
    model = _simple_model(test_dataset, gene_annot_cols=annotations)
    model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    fit = kasm.fit_random_effects(model, exclude_genes=[], n_jk_blocks=2)

    components = fit.gene_annot_components
    assert components["gene_annot"].to_list() == annotations
    assert components["hsq"].sum() == pytest.approx(fit.hsq, rel=1e-9, abs=1e-11)
    assert components["n_genes"].sum() == fit.n_genes
    assert components["n_variants"].sum() == fit.n_variants
    assert components["het"].min() > 0
    hsq_jk = np.asarray(components["hsq_jk"].to_list())
    het_jk = np.asarray(components["het_jk"].to_list())
    assert hsq_jk.shape == het_jk.shape == (len(annotations), fit.n_jk_blocks)
    np.testing.assert_allclose(hsq_jk.sum(axis=0), fit.hsq_jk, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("fix_gene_coef", [False, True])
def test_public_joint_gene_coef_preserve_or_normalize_scale(
    test_dataset, fix_gene_coef
):
    model = _simple_model(test_dataset)
    model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    fits = kasm.fit_random_effects(
        {"low": model, "high": model},
        exclude_genes=[],
        init_gene_coef={"high": {"constant": 1.5}, "low": {"constant": 0.5}},
        fix_gene_coef=fix_gene_coef,
        n_jk_blocks=2,
    )
    for name, scale in (("low", 0.5), ("high", 1.5)):
        np.testing.assert_array_equal(
            fits[name].gene_coef, [[scale if fix_gene_coef else 1.0]]
        )
        np.testing.assert_array_equal(
            fits[name].gene_coef_jk, np.full((2, 1, 1), scale if fix_gene_coef else 1.0)
        )


def test_grouped_fit_accepts_annotated_independent_kernels(
    synthetic, synthetic_ldm_path, tmp_path
):
    frame = synthetic.variant_df.with_columns(
        (pl.col("score") >= 0.5).cast(pl.Float64).alias("class_m"),
        (pl.col("score") < 0.5).cast(pl.Float64).alias("class_l"),
    )
    dataset = kasm.Dataset(
        frame,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=100_000,
        ldm=synthetic_ldm_path,
    )
    configured_model = kasm.Model(
        dataset,
        kasm.KernelOptions(
            kernel_groups={
                "missense": {
                    "burden_cols": ["class_m"],
                    "independent_col": "class_m",
                },
                "plof": {
                    "burden_cols": ["class_l"],
                    "independent_col": "class_l",
                },
            },
            covariance_between_kernel_groups={"missense_x_plof": ("missense", "plof")},
        ),
    )
    configured_model.compute_suffstats()
    with pytest.raises(ValueError, match="cannot be combined in a model mapping"):
        kasm.fit_random_effects(
            {"configured": configured_model},
            exclude_genes={"configured": []},
            n_jk_blocks=1,
        )
    configured_stats = configured_model.compute_jk_suffstats(
        exclude_genes=[], n_jk_blocks=1
    )
    assert configured_stats.excluded_genes == ()
    cache_path = tmp_path / "grouped-flat.npz"
    configured_stats.save(cache_path)
    configured_stats = kasm.JKSuffStats.load(
        cache_path,
        exclude_genes={"missense": [], "plof": []},
        n_jk_blocks=1,
    )
    assert configured_stats.excluded_genes == ()
    with pytest.raises(ValueError, match="cannot be combined in a model mapping"):
        kasm.fit_random_effects(
            {"configured": configured_stats},
            exclude_genes={"configured": []},
            n_jk_blocks=1,
        )
    burden_only_model = kasm.Model(
        dataset,
        kasm.KernelOptions(
            kernel_groups={
                "missense": {"burden_cols": ["class_m"]},
                "plof": {"burden_cols": ["class_l"]},
            },
            covariance_between_kernel_groups={"missense_x_plof": ("missense", "plof")},
        ),
    )
    burden_only_model.compute_suffstats()
    burden_only_model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=1)
    burden_only_fit = kasm.fit_random_effects(
        burden_only_model,
        exclude_genes=[],
        init_gene_coef=None,
        n_jk_blocks=1,
    )
    configured_model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=1)
    default_fit = kasm.fit_random_effects(
        configured_model, exclude_genes=[], n_jk_blocks=1
    )
    cached_fit = kasm.fit_random_effects(
        configured_stats,
        exclude_genes={"missense": [], "plof": []},
        n_jk_blocks=1,
    )
    np.testing.assert_array_equal(cached_fit.kernel_coef, default_fit.kernel_coef)
    np.testing.assert_array_equal(cached_fit.gene_coef, default_fit.gene_coef)
    assert default_fit.gene_coef.shape == (1, 3)
    configured_fit = kasm.fit_random_effects(
        configured_model,
        exclude_genes=[],
        fix_gene_coef=True,
        init_gene_coef=burden_only_fit.gene_coef_dict,
        n_jk_blocks=1,
    )
    assert tuple(configured_fit.gene_coef_dict) == (
        "missense",
        "plof",
        "missense_x_plof",
    )
    independent = np.asarray(configured_model.layout.independent_indices)
    np.testing.assert_allclose(
        configured_fit.gene_coef,
        burden_only_fit.gene_coef,
    )
    np.testing.assert_allclose(
        configured_fit.gene_coef_jk,
        np.broadcast_to(
            configured_fit.gene_coef,
            configured_fit.gene_coef_jk.shape,
        ),
    )
    assert np.isfinite(configured_fit.kernel_coef[independent]).all()
    assert np.any(configured_fit.kernel_coef[independent] < 0)

    with pytest.raises(ValueError, match="keys must match kernel groups"):
        kasm.fit_random_effects(
            configured_model,
            exclude_genes={name: [] for name in configured_fit.gene_coef_dict},
            init_gene_coef=None,
            n_jk_blocks=1,
        )

    with pytest.raises(ValueError, match="keys must match"):
        kasm.fit_random_effects(
            configured_model,
            exclude_genes=[],
            fix_gene_coef=True,
            init_gene_coef={"missense": {"constant": 1.0}},
            n_jk_blocks=1,
        )
    with pytest.raises(ValueError, match="init_gene_coef.*missense.*keys must match"):
        kasm.fit_random_effects(
            configured_model,
            exclude_genes=[],
            fix_gene_coef=True,
            init_gene_coef={
                name: {"wrong": 1.0} for name in configured_fit.gene_coef_dict
            },
            n_jk_blocks=1,
        )
    simple_model = kasm.Model(
        dataset,
        kasm.KernelOptions(burden_cols=["constant"]),
    )
    simple_model.compute_suffstats()
    # Default contiguous blocks over 5 genes and 3 blocks are [0, 0, 1, 1, 2];
    # excluding both leading genes leaves block 0 empty.
    simple_model.compute_jk_suffstats(
        exclude_genes=simple_model.genes[:2], n_jk_blocks=3
    )
    fit_with_empty_block = kasm.fit_random_effects(
        simple_model,
        exclude_genes=simple_model.genes[:2],
        kernel_groups={"burden": ["constant"]},
        n_jk_blocks=3,
    )
    assert fit_with_empty_block.n_jk_blocks == 3
    np.testing.assert_allclose(
        fit_with_empty_block.kernel_coef_jk[0], fit_with_empty_block.kernel_coef
    )
    np.testing.assert_allclose(
        fit_with_empty_block.gene_coef_jk[0],
        fit_with_empty_block.gene_coef,
    )


def test_fit_time_grouping_keeps_overdispersion_parameters_distinct(
    synthetic_dataset, monkeypatch
):
    model = kasm.Model(
        synthetic_dataset,
        kasm.KernelOptions(
            burden_cols=["constant", "score"],
            overdispersion_cols=["constant", "score"],
        ),
    ).compute_suffstats()
    observed = {}

    class Captured(Exception):
        pass

    def capture(inputs, **kwargs):
        observed["kernel_coef_loc"] = inputs[0].kernel_coef_loc
        raise Captured

    monkeypatch.setattr(kasm.fitting, "alternating_fit", capture)
    model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=1)
    with pytest.raises(Captured):
        kasm.fit_random_effects(
            model,
            exclude_genes=[],
            kernel_groups={"all": ["constant", "score"]},
            n_jk_blocks=1,
        )

    np.testing.assert_array_equal(
        observed["kernel_coef_loc"], np.arange(model.layout.n_kernels)
    )


@pytest.mark.parametrize("by_maf", [True, False])
@pytest.mark.parametrize("independent", [True, False])
@pytest.mark.parametrize("shared", [True, False])
def test_grouped_fit_reports_additive_group_and_maf_components(
    synthetic, synthetic_ldm_path, monkeypatch, by_maf, independent, shared
):
    frame = synthetic.variant_df.with_columns(
        (pl.col("score") >= 0.5).cast(pl.Float64).alias("is_missense"),
        (pl.col("score") < 0.5).cast(pl.Float64).alias("is_plof"),
    )
    model = kasm.Model(
        kasm.Dataset(
            frame,
            gene_df=synthetic.gene_df,
            beta_col="beta",
            se_col="se",
            sample_size=100_000,
            ldm=synthetic_ldm_path,
        ),
        kasm.KernelOptions(
            maf_col="af",
            maf_bins={"rare": (0, 1)},
            gene_annot_cols=["constraint_low", "constraint_high"],
            kernel_groups={
                "missense": {
                    "burden_cols": ["is_missense"],
                    "independent_col": "is_missense" if independent else None,
                    "overdispersion_cols": ["is_missense"],
                },
                "plof": {
                    "burden_cols": ["is_plof"],
                    "independent_col": "is_plof" if independent else None,
                    "overdispersion_cols": ["is_plof"],
                },
            },
            burden_by_maf=by_maf,
            overdispersion_shared=shared,
            overdispersion_by_maf=True,
        ),
    ).compute_suffstats()
    exclusions = {"missense": model.genes[:1], "plof": model.genes[-1:]}
    model.compute_jk_suffstats(exclude_genes=exclusions, n_jk_blocks=1)
    fit = kasm.fit_random_effects(
        model,
        exclude_genes=exclusions,
        fix_gene_coef=True,
        init_gene_coef={
            "missense": {"constraint_low": 0.5, "constraint_high": 1.5},
            "plof": {"constraint_low": 1.5, "constraint_high": 0.5},
        },
        n_jk_blocks=1,
    )

    assert fit.point["kernel_group"].drop_nulls().to_list() == ["missense", "plof"]
    assert fit.point.filter(pl.col("kernel_group").is_not_null())[
        "hsq"
    ].sum() == pytest.approx(fit.hsq)
    group_jk = (
        fit.jk.filter(pl.col("kernel_group").is_not_null())["hsq"]
        .to_numpy()
        .reshape(len(fit.kernel_group_names), fit.n_jk_blocks)
    )
    np.testing.assert_allclose(group_jk.sum(axis=0), fit.hsq_jk)
    original_load_ldm = model.dataset.load_ldm
    loaded = []

    def record_load(gene):
        loaded.append(gene)
        return original_load_ldm(gene)

    monkeypatch.setattr(model.dataset, "load_ldm", record_load)
    maf_components = fit.maf_components
    assert loaded == []  # Geometry was reduced during fit assembly.
    assert maf_components["kernel_group"].to_list() == ["missense", "plof"]
    assert maf_components["hsq"].sum() == pytest.approx(fit.hsq)
    np.testing.assert_allclose(
        np.asarray(maf_components["hsq_jk"].to_list()).sum(axis=0),
        fit.hsq_jk,
    )
    assert set(fit.components["kernel_group"].drop_nulls()) == {"missense", "plof"}
    overdispersion = fit.overdispersion_components
    assert overdispersion["annotation"].to_list() == ["is_missense", "is_plof"]
    assert overdispersion["kernel_coef"].n_unique() == (1 if shared else 2)


def test_public_joint_combination_and_mapping_inputs(test_dataset):
    models = {}
    for label in ("missense", "lof"):
        frame = test_dataset.variant_df().filter(pl.col("variant_class") == label)
        dataset = kasm.Dataset(
            frame,
            gene_df=test_dataset.gene_df(),
            beta_col="BETA",
            se_col="SE",
            sample_size=test_dataset.sample_size,
            ldm=test_dataset.ldm_source,
        )
        models[label] = _simple_model(dataset, by_maf=True, overdispersion=True)
    excluded_by_model = {
        label: model.dataset.genes[:1] for label, model in models.items()
    }
    for label, model in models.items():
        model.compute_jk_suffstats(
            exclude_genes=excluded_by_model[label], n_jk_blocks=2
        )
    joint = kasm.fit_random_effects(
        models,
        exclude_genes={
            label: (gene for gene in genes)
            for label, genes in excluded_by_model.items()
        },
        n_jk_blocks=2,
    )
    assert set(joint) == set(models)
    combined = kasm.combine_random_hsq(joint)
    expected_jk = sum(fit.hsq_jk for fit in joint.values())
    assert combined[0] == pytest.approx(sum(fit.hsq for fit in joint.values()))
    assert combined[1] == pytest.approx(kasm.jackknife_se(expected_jk[:, None])[0])
    for label, fit in joint.items():
        components = fit.maf_components
        assert components["hsq"].sum() == pytest.approx(fit.hsq)
        np.testing.assert_allclose(
            np.asarray(components["hsq_jk"].to_list()).sum(axis=0),
            fit.hsq_jk,
            rtol=1e-9,
            atol=1e-11,
        )
    fixed = kasm.fit_fixed_effects(
        models,
        include_genes={
            label: (gene for gene in genes)
            for label, genes in excluded_by_model.items()
        },
        method="burden",
    )
    assert isinstance(fixed, dict)
    assert set(fixed) == set(models)
    for fit in fixed.values():
        assert fit.maf_components["hsq"].sum() == pytest.approx(fit.hsq)
    with pytest.raises(ValueError, match="requires a single model"):
        kasm.fit_fixed_effects(
            models,
            include_genes=excluded_by_model,
            extra_enrich_col="netsurfp.rsa_G1",
        )
    mixed = {
        label: kasm.combine_mixed_effects(random_fit=random, fixed_fit=fixed[label])
        for label, random in joint.items()
    }
    assert sum(hsq for hsq, _ in mixed.values()) == pytest.approx(
        combined[0] + sum(fit.hsq for fit in fixed.values())
    )
    for label, (hsq, hsq_se) in mixed.items():
        assert hsq == pytest.approx(joint[label].hsq + fixed[label].hsq)
        assert hsq_se == pytest.approx(
            np.hypot(joint[label].hsq_se, fixed[label].hsq_se)
        )


def test_mixed_fit_requires_exact_split_without_model_identity_checks(test_dataset):
    model = _simple_model(test_dataset)
    excluded = model.dataset.genes[:2]
    model.compute_jk_suffstats(exclude_genes=excluded, n_jk_blocks=2)
    random_fit = kasm.fit_random_effects(model, exclude_genes=excluded, n_jk_blocks=2)

    incomplete_fixed = kasm.fit_fixed_effects(model, include_genes=excluded[:1])
    with pytest.raises(ValueError, match="exactly match"):
        kasm.combine_mixed_effects(random_fit=random_fit, fixed_fit=incomplete_fixed)

    other_model = _simple_model(test_dataset)
    other_fixed = kasm.fit_fixed_effects(other_model, include_genes=excluded)
    assert (
        kasm.combine_mixed_effects(random_fit=random_fit, fixed_fit=other_fixed)[0]
        == random_fit.hsq + other_fixed.hsq
    )

    model.compute_suffstats()
    current_fixed = kasm.fit_fixed_effects(model, include_genes=excluded)
    assert (
        kasm.combine_mixed_effects(random_fit=random_fit, fixed_fit=current_fixed)[0]
        == random_fit.hsq + current_fixed.hsq
    )


@pytest.mark.parametrize("shape", ["ordinary", "grouped", "mapping"])
@pytest.mark.parametrize(
    "fix_gene_coef, max_iter", [(False, 1), (False, 200), (True, 1)]
)
def test_gene_initialization_alignment_and_point_start_deletes(
    synthetic_dataset, monkeypatch, shape, fix_gene_coef, max_iter
):
    annotations = ["constraint_low", "constraint_high"]
    model = kasm.Model(
        synthetic_dataset,
        kasm.KernelOptions(
            gene_annot_cols=annotations,
            **(
                {"kernel_groups": {"all": {"burden_cols": ["constant"]}}}
                if shape == "grouped"
                else {"burden_cols": ["constant"]}
            ),
        ),
    ).compute_suffstats()
    values = {"constraint_high": 6.0, "constraint_low": 2.0}
    supplied = {"all": values} if shape == "grouped" else values
    models = {"first": model, "second": model} if shape == "mapping" else model
    if shape == "mapping":
        supplied = {"second": values.copy(), "first": values.copy()}
    original = kasm.fitting.alternating_fit
    calls, results = [], []

    def record(inputs, **kwargs):
        calls.append(kwargs["init_gene_coef"])
        results.append(original(inputs, **kwargs))
        return results[-1]

    monkeypatch.setattr(kasm.fitting, "alternating_fit", record)
    model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    fitted = kasm.fit_random_effects(
        models,
        exclude_genes=[],
        init_gene_coef=supplied,
        fix_gene_coef=fix_gene_coef,
        n_jk_blocks=2,
        max_iter=max_iter,
    )
    assert len(calls) == 3
    for index, fit in enumerate(fitted.values() if shape == "mapping" else [fitted]):
        np.testing.assert_array_equal(calls[0][index], [[2.0], [6.0]])
        for start in calls[1:]:
            np.testing.assert_array_equal(start[index], fit.gene_coef)
        if fix_gene_coef:
            np.testing.assert_array_equal(fit.gene_coef, [[2.0], [6.0]])
            np.testing.assert_array_equal(fit.gene_coef_jk, [[[2.0], [6.0]]] * 2)
        else:
            np.testing.assert_allclose(fit.gene_coef.mean(axis=0), 1.0)
            np.testing.assert_allclose(fit.gene_coef_jk.mean(axis=1), 1.0)
            assert np.any(fit.gene_coef_jk != fit.gene_coef)
    assert values == {"constraint_high": 6.0, "constraint_low": 2.0}


@pytest.mark.parametrize(
    "values, error, message",
    [
        ({"cell": None}, TypeError, "dictionary"),
        (
            {"cell": {"wrong": 1.0}},
            ValueError,
            "missing labels.*constant.*extra labels.*wrong",
        ),
        (
            {"wrong": {"constant": 1.0}},
            ValueError,
            "missing labels.*cell.*extra labels.*wrong",
        ),
    ],
)
def test_model_initial_values_require_complete_hierarchy(
    test_dataset, values, error, message
):
    model = _simple_model(test_dataset)
    model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=1)
    with pytest.raises(error, match=message):
        kasm.fit_random_effects(
            {"cell": model},
            exclude_genes=[],
            init_gene_coef=values,
            n_jk_blocks=1,
        )


@pytest.mark.parametrize("fix_gene_coef", [False, True])
def test_compact_mapping_accepts_gene_values(synthetic_dataset, fix_gene_coef):
    model = kasm.Model(
        synthetic_dataset,
        kasm.KernelOptions(
            burden_cols=["constant"],
            gene_annot_cols=["constraint_low", "constraint_high"],
        ),
    ).compute_suffstats()
    stats = model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    options = dict(
        init_gene_coef={"cell": {"constraint_high": 6.0, "constraint_low": 2.0}},
        fix_gene_coef=fix_gene_coef,
        n_jk_blocks=2,
    )
    compact = kasm.fit_random_effects({"cell": stats}, **options)["cell"]
    retained = kasm.fit_random_effects({"cell": model}, exclude_genes=[], **options)[
        "cell"
    ]
    for field in (
        "kernel_coef",
        "kernel_coef_jk",
        "gene_coef",
        "gene_coef_jk",
        "hsq",
        "hsq_se",
    ):
        np.testing.assert_array_equal(getattr(compact, field), getattr(retained, field))


def test_group_dictionary_order_and_one_group_result_reuse(synthetic_dataset):
    model = kasm.Model(
        synthetic_dataset,
        kasm.KernelOptions(
            gene_annot_cols=["constraint_low", "constraint_high"],
            kernel_groups={
                "first": {"burden_cols": ["constant"]},
                "second": {"burden_cols": ["score"]},
            },
        ),
    ).compute_suffstats()
    values = {
        "second": {"constraint_high": 4.0, "constraint_low": 12.0},
        "first": {"constraint_high": 6.0, "constraint_low": 2.0},
    }
    model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    fit = kasm.fit_random_effects(
        model,
        exclude_genes=[],
        init_gene_coef=values,
        fix_gene_coef=True,
        n_jk_blocks=2,
    )
    np.testing.assert_array_equal(fit.gene_coef, [[2, 12], [6, 4]])
    np.testing.assert_array_equal(fit.gene_coef_jk, [[[2, 12], [6, 4]]] * 2)
    assert fit.gene_coef_dict == values

    single_group = kasm.Model(
        synthetic_dataset,
        kasm.KernelOptions(
            gene_annot_cols=["constraint_low", "constraint_high"],
            kernel_groups={"all": {"burden_cols": ["constant"]}},
        ),
    ).compute_suffstats()
    single_group.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=1)
    point = kasm.fit_random_effects(single_group, exclude_genes=[], n_jk_blocks=1)
    assert set(point.gene_coef_dict) == {"all"}
    reused = kasm.fit_random_effects(
        single_group,
        exclude_genes=[],
        init_gene_coef=point.gene_coef_dict,
        fix_gene_coef=True,
        n_jk_blocks=1,
    )
    np.testing.assert_array_equal(reused.gene_coef, point.gene_coef)
    np.testing.assert_array_equal(reused.kernel_coef, point.kernel_coef)


@pytest.mark.parametrize("n_blocks", [1, 2])
@pytest.mark.parametrize("per_kernel_group_exclusions", [False, True])
def test_random_summary_support_and_source_free_roundtrip(
    synthetic_dataset, tmp_path, monkeypatch, n_blocks, per_kernel_group_exclusions
):
    model = kasm.Model(
        synthetic_dataset,
        kasm.KernelOptions(
            kernel_groups={
                "first": {"burden_cols": ["constant"]},
                "second": {"burden_cols": ["score"]},
            },
            maf_col="af",
            maf_bins={"all": (0, 1)},
        ),
    ).compute_suffstats()
    exclusions = (
        {"first": model.genes[:1], "second": model.genes[-1:]}
        if per_kernel_group_exclusions
        else {"first": [], "second": []}
    )
    model.compute_jk_suffstats(
        exclude_genes=exclusions if per_kernel_group_exclusions else [],
        n_jk_blocks=n_blocks,
    )
    fit = kasm.fit_random_effects(
        model,
        exclude_genes=exclusions if per_kernel_group_exclusions else [],
        init_gene_coef={"first": {"constant": 2.0}, "second": {"constant": 3.0}},
        fix_gene_coef=True,
        n_jk_blocks=n_blocks,
    )
    frame = synthetic_dataset.variant_df()
    supports = [
        frame.filter(~pl.col("gene").is_in(exclusions[name]) & (pl.col(column) != 0))
        for name, column in (("first", "constant"), ("second", "score"))
    ]
    assert fit.point.schema == {
        "kernel_group": pl.String,
        "hsq": pl.Float64,
        "hsq_se": pl.Float64,
        "n_genes": pl.Int64,
        "n_variants": pl.Int64,
    }
    assert fit.jk.schema == {
        "block": pl.Int64,
        "kernel_group": pl.String,
        "hsq": pl.Float64,
    }
    assert fit.point["kernel_group"].to_list() == [None, "first", "second"]
    expected_total = (
        (None, None)
        if per_kernel_group_exclusions
        else (
            synthetic_dataset.n_genes,
            synthetic_dataset.n_variants,
        )
    )
    assert fit.point.row(0, named=True)["n_genes"] == expected_total[0]
    assert fit.point.row(0, named=True)["n_variants"] == expected_total[1]
    for row, support in zip(list(fit.point.iter_rows(named=True))[1:], supports):
        assert row["n_genes"] == support["gene"].n_unique()
        assert row["n_variants"] == support.height
    for index, name in enumerate(fit.kernel_group_names):
        deleted = fit.variance_jk[:, fit.kernel_groups == index].sum(axis=1)
        row = fit.point.filter(pl.col("kernel_group") == name).row(0, named=True)
        np.testing.assert_equal(row["hsq_se"], kasm.jackknife_se(deleted[:, None])[0])
        np.testing.assert_array_equal(
            fit.jk.filter(pl.col("kernel_group") == name)["hsq"], deleted
        )
    assert fit.point["hsq"][0] == fit.hsq
    np.testing.assert_array_equal(
        fit.jk.filter(pl.col("kernel_group").is_null())["hsq"], fit.hsq_jk
    )
    with pytest.raises(ValueError, match="single kernel group"):
        fit.score_variants(synthetic_dataset)
    path = tmp_path / "random.npz"
    fit.save(path)
    loaded = kasm.RandomFit.load(path)

    def no_training_geometry(*args, **kwargs):
        raise AssertionError("returned results must not read training geometry")

    monkeypatch.setattr(model, "gene_annot_matrix", no_training_geometry)
    monkeypatch.setattr(synthetic_dataset, "load_ldm", no_training_geometry)
    for name in (
        "point",
        "jk",
        "components",
        "maf_components",
        "overdispersion_components",
    ):
        assert getattr(loaded, name).equals(getattr(fit, name))
    with pytest.raises(ValueError, match="single kernel group"):
        loaded.score_variants(synthetic_dataset)
    assert "kernel_coef" in loaded.components.columns
    assert "kernel_coef_se" in loaded.components.columns
    assert "coefficient" not in loaded.components.columns
    if n_blocks == 1:
        assert fit.point["hsq_se"].is_nan().all()
        np.testing.assert_array_equal(fit.kernel_coef_jk[0], fit.kernel_coef)


def test_explicit_fitting_defaults_and_compact_block_validation(synthetic_dataset):
    model = kasm.Model(synthetic_dataset, kasm.KernelOptions(burden_cols=["constant"]))
    model.compute_suffstats()
    stats = model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    with pytest.raises(ValueError, match="n_jk_blocks does not match"):
        kasm.fit_random_effects({"cached": stats}, n_jk_blocks=3)
    fit = kasm.fit_random_effects({"cached": stats}, n_jk_blocks=stats.n_jk_blocks)[
        "cached"
    ]
    assert fit.n_jk_blocks == 2
    model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=len(model.genes))
    ordinary = kasm.fit_random_effects(model, exclude_genes=[])
    assert ordinary.n_jk_blocks == len(model.genes)
    assert ordinary.point["kernel_group"].to_list() == [None]
    assert ordinary.point.schema == fit.point.schema
    assert ordinary.jk.schema == fit.jk.schema


@pytest.mark.parametrize("method", ["burden", "burden_1d", "indep"])
def test_fixed_fit_owns_results_and_does_not_require_jackknife_stats(
    synthetic_dataset, monkeypatch, method
):
    from polars.testing import assert_frame_equal

    from kasm.fixed_effects import FixedFit, _fit_fixed_effects

    model = kasm.Model(
        synthetic_dataset,
        kasm.KernelOptions(
            burden_cols=["constant", "score"],
            maf_col="af",
            maf_bins={"all": (0, 1)},
        ),
    ).compute_suffstats()
    model.compute_gene_pvalue()
    genes = model.genes
    expected = kasm.fit_fixed_effects(model, include_genes=genes, method=method)
    model.compute_jk_suffstats(exclude_genes=genes, n_jk_blocks=1)
    assert isinstance(model.suffstats, kasm.SuffStats)
    model.compute_gene_pvalue()
    model.suffstats = None
    monkeypatch.setattr(
        model, "_require_suffstats", lambda: pytest.fail("fixed fit consulted stats")
    )
    observed = kasm.fit_fixed_effects(model, include_genes=genes, method=method)
    numerical = _fit_fixed_effects(model, include_genes=genes, method=method)
    assert type(observed) is type(numerical) is FixedFit is kasm.FixedFit
    assert {"hsq", "hsq_se", "per_gene", "method", "n_genes"} <= vars(observed).keys()
    assert observed._model is model
    assert_frame_equal(observed.point, expected.point)
    assert_frame_equal(observed.per_gene, expected.per_gene)
    assert_frame_equal(observed.maf_components, expected.maf_components)


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("n_blocks", [1, 3])
def test_random_result_assembly_matches_scalar_moment_and_raw_delete_reference(
    synthetic_dataset, conditional, n_blocks
):
    model = kasm.Model(
        synthetic_dataset,
        kasm.KernelOptions(
            burden_cols=None if conditional else ["constant"],
            kernel_groups={"all": {"burden_cols": ["constant"]}}
            if conditional
            else None,
            gene_annot_cols=["constraint_low", "constraint_high"],
        ),
    ).compute_suffstats()
    coefficients = {"constraint_low": 2.0, "constraint_high": 4.0}
    excluded = model.genes[:1]
    exclusion_geometry = {"all": excluded} if conditional else excluded
    model.compute_jk_suffstats(exclude_genes=exclusion_geometry, n_jk_blocks=n_blocks)
    fit = kasm.fit_random_effects(
        model,
        exclude_genes=exclusion_geometry,
        fix_gene_coef=True,
        init_gene_coef={"all": coefficients} if conditional else coefficients,
        n_jk_blocks=n_blocks,
    )
    stats = model.suffstats
    scales = model.gene_annot_matrix() @ np.array([2.0, 4.0])
    blocks = kasm.fitting.assign_jackknife_blocks(len(model.genes), n_blocks)
    point_and_deletes = []
    for deleted in (None, *range(n_blocks)):
        keep = np.array([gene not in excluded for gene in model.genes])
        if deleted is not None and n_blocks > 1:
            keep &= blocks != deleted
        pair = np.sum(stats.pair_trace_upper[keep, 0] * scales[keep] ** 2)
        trace = np.sum(stats.trace[keep, 0] * scales[keep])
        quad = np.sum(stats.quadform[keep, 0] * scales[keep])
        coefficient = (quad - trace / model.dataset.sample_size) / pair
        point_and_deletes.append((coefficient, coefficient * trace))
    reference = np.asarray(point_and_deletes)
    np.testing.assert_allclose(
        [fit.kernel_coef[0], fit.hsq], reference[0], rtol=1e-12, atol=1e-16
    )
    np.testing.assert_allclose(
        np.column_stack((fit.kernel_coef_jk[:, 0], fit.hsq_jk)),
        reference[1:],
        rtol=1e-12,
        atol=1e-16,
    )
    np.testing.assert_array_equal(
        fit.gene_coef_jk, np.broadcast_to([[2.0], [4.0]], (n_blocks, 2, 1))
    )
    if n_blocks == 1:
        assert np.isnan(fit.hsq_se)
        assert np.isnan(fit.kernel_coef_se).all()
    else:
        centered = reference[1:] - reference[1:].mean(axis=0)
        expected_se = np.sqrt((n_blocks - 1) / n_blocks * np.sum(centered**2, axis=0))
        np.testing.assert_allclose(
            [fit.kernel_coef_se[0], fit.hsq_se], expected_se, rtol=1e-12, atol=1e-16
        )
    assert fit.n_iter == 0
    assert fit.converged and fit.converged_jk.all()


@pytest.mark.parametrize("enabled", [False, True])
def test_boolean_psd_option_reaches_solver(synthetic_dataset, monkeypatch, enabled):
    from kasm import fitting

    model = kasm.Model(
        synthetic_dataset,
        kasm.KernelOptions(burden_cols=["constant", "score"]),
    )
    original = fitting.alternating_fit
    observed = []

    def capture(inputs, **kwargs):
        observed.extend(inputs)
        return original(inputs, **kwargs)

    monkeypatch.setattr(fitting, "alternating_fit", capture)
    kasm.fit_random_effects(
        model, exclude_genes=[], n_jk_blocks=1, psd_burden_coef=enabled
    )
    assert len(observed) == 1
    if enabled:
        assert observed[0].psd_kernel_basis is not None
        np.testing.assert_array_equal(
            observed[0].psd_annotation_patterns,
            np.ones((1, model.gene_annot_matrix().shape[1])),
        )
    else:
        assert observed[0].psd_kernel_basis is None
        assert observed[0].psd_annotation_patterns is None
