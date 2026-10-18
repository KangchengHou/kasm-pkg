"""Covar extension statistics and random covar."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from extension_helpers import extension_designs
from scipy import sparse

import kasm
from kasm.ldm import LDM


def distance_covar_kernels(frame):
    coordinate = frame["position_coordinate"].to_numpy().astype(float)
    distance = np.abs(coordinate[:, None] - coordinate[None, :])
    covar_kernels = {
        "near": (distance > 0) & (distance <= 1),
        "far": distance > 1,
    }
    return {
        name: sparse.csr_matrix(mask, dtype=float)
        for name, mask in covar_kernels.items()
    }


def class_a_covar_kernels(frame):
    selected = frame["is_a"].to_numpy().astype(bool)
    covar_kernel = selected[:, None] & selected[None, :]
    np.fill_diagonal(covar_kernel, False)
    return {"within": sparse.csr_matrix(covar_kernel, dtype=float)}


BUILDER_CALLS = {"a": 0, "b": 0}


def counted_class_a_covar_kernels(frame):
    BUILDER_CALLS["a"] += 1
    covar_kernel = class_a_covar_kernels(frame)["within"]
    return {"within": covar_kernel, "within_copy": covar_kernel.copy()}


def counted_class_b_covar_kernels(frame):
    BUILDER_CALLS["b"] += 1
    selected = frame["is_b"].to_numpy().astype(bool)
    covar_kernel = selected[:, None] & selected[None, :]
    np.fill_diagonal(covar_kernel, False)
    return {"within": sparse.csr_matrix(covar_kernel, dtype=float)}


def unsupported_group_covar_kernels(frame):
    covar_kernel = np.ones((len(frame), len(frame)), dtype=float)
    np.fill_diagonal(covar_kernel, False)
    return {"outside": sparse.csr_matrix(covar_kernel)}


def class_a_near_covar_kernels(frame):
    selected = frame["is_a"].to_numpy().astype(bool)
    index = np.arange(len(frame))
    mask = (
        selected[:, None]
        & selected[None, :]
        & (np.abs(index[:, None] - index[None, :]) <= 1)
    )
    np.fill_diagonal(mask, False)
    return {"within": sparse.csr_matrix(mask.astype(float))}


def empty_covar_kernels(frame):
    return {"empty": sparse.csr_matrix((len(frame), len(frame)), dtype=float)}


def _covar_model(dataset, *, maf_scaled=False):
    frame = dataset.variant_df().with_columns(
        (pl.col("gene").cum_count().over("gene") - 1)
        .cast(pl.Float64)
        .alias("position_coordinate")
    )
    annotated = kasm.Dataset(
        frame,
        gene_df=dataset.gene_df(),
        beta_col=dataset.beta_col,
        se_col=dataset.se_col,
        sample_size=dataset.sample_size,
        ldm=dataset.ldm_source,
    )
    return kasm.Model(
        annotated,
        kasm.KernelOptions(
            maf_col="af" if maf_scaled else None,
            burden_by_maf=False,
            kernel_groups={
                "all": {
                    "burden_cols": ["constant", "score"],
                    "independent_col": "constant",
                    "extra_covar_kernel_fun": distance_covar_kernels,
                }
            },
            variant_variance_from_maf=maf_scaled,
        ),
    )


def _base_dense_kernels(model, gene):
    burden_wgts = model.burden_wgts(gene)
    diagonal = model.diag_kernel_weights(gene)
    kernels = [None] * model.layout.n_kernels
    for position, index in enumerate(model.layout.burden_indices):
        kernels[index] = np.outer(burden_wgts[:, position], burden_wgts[:, position])
    for (left, right), index in zip(
        model.layout.cross_pairs, model.layout.cross_indices
    ):
        kernels[index] = np.outer(
            burden_wgts[:, left], burden_wgts[:, right]
        ) + np.outer(burden_wgts[:, right], burden_wgts[:, left])
    for position, index in enumerate(model.layout.diagonal_indices):
        kernels[index] = np.diag(diagonal[:, position])
    return kernels


def test_covar_statistics_match_dense_matrix_oracles(
    synthetic_dataset,
):
    model = _covar_model(synthetic_dataset)
    model.compute_suffstats()
    covar = model.suffstats.covar_extensions

    assert tuple(covar) == ("all:near", "all:far")
    assert all(
        extension.pair_trace.shape == (model.dataset.n_genes, 1, model.n_kernels + 1)
        for extension in covar.values()
    )

    for gene_index, gene in enumerate(model.dataset.genes):
        rows = model.dataset.gene_slice(gene)
        beta_tilde = model.dataset.beta[rows] / (
            model.dataset.se[rows] ** 2 * model.dataset.sample_size
        )
        residual = (
            model.dataset.load_ldm(gene).scale(model.variant_scale(gene)).toarray()
        )
        base = _base_dense_kernels(model, gene)
        covar_kernels = distance_covar_kernels(model.dataset.variant_df()[rows])
        for covar_kernel_index, name in enumerate(covar):
            kernel = covar_kernels[name.removeprefix("all:")].toarray()
            assert covar[name].trace[gene_index, 0] == pytest.approx(
                np.trace(kernel @ residual)
            )
            assert covar[name].quadform[gene_index, 0] == pytest.approx(
                beta_tilde @ kernel @ beta_tilde
            )
            assert covar[name].pair_trace[gene_index, 0, -1] == pytest.approx(
                np.trace(kernel @ residual @ kernel @ residual)
            )
            expected_cross = np.array(
                [
                    np.trace(base_kernel @ residual @ kernel @ residual)
                    for base_kernel in base
                ]
            )
            np.testing.assert_allclose(
                covar[name].pair_trace[gene_index, 0, :-1],
                expected_cross,
                rtol=1e-9,
                atol=1e-24,
            )


def test_random_suffstats_never_materialize_dense_ld(
    synthetic_dataset,
    monkeypatch,
):
    model = _covar_model(synthetic_dataset)

    def reject_dense(self):
        raise AssertionError("random covar materialized dense LD")

    monkeypatch.setattr(LDM, "toarray", reject_dense)
    model.compute_suffstats()
    model.compute_jk_suffstats(exclude_genes={"all": []}, n_jk_blocks=2)
    kasm.fit_random_effects(
        model,
        extra_covar_kernel="all:near",
        exclude_genes={"all": []},
        fix_gene_coef=True,
        init_gene_coef={"all": {"constant": 1.0}},
        n_jk_blocks=2,
    )


def test_covariance_kernels_must_be_binary():
    weighted = sparse.csr_matrix([[0.0, 0.5], [0.5, 0.0]])

    with pytest.raises(ValueError, match="must be binary"):
        kasm.covar._validate_covar_kernel(weighted, 2, "weighted", "g")


def test_grouped_random_covar_has_aligned_additive_moments(
    synthetic_dataset_no_ld,
    tmp_path,
):
    model = _grouped_covar_kernel_model(synthetic_dataset_no_ld)
    options = {
        "exclude_genes": {"missense": []},
        "fix_gene_coef": True,
        "init_gene_coef": {"missense": {"constraint_low": 0.5, "constraint_high": 1.5}},
        "n_jk_blocks": 2,
    }
    model.compute_jk_suffstats(
        exclude_genes=options["exclude_genes"], n_jk_blocks=options["n_jk_blocks"]
    )
    fit, covar = kasm.fit_random_effects(
        model,
        extra_covar_kernel="missense:near",
        **options,
    )
    result = covar

    assert result.jk[..., 0].shape == (2,)

    expected_scores = fit.score_variants(model.dataset)
    path = tmp_path / "covar.npz"
    fit.save(path)
    loaded = kasm.RandomFit.load(path)
    for name in ("point", "jk", "components"):
        assert getattr(loaded, name).equals(getattr(fit, name))
    covar_path = tmp_path / "moments.npz"
    result.save(covar_path)
    loaded_covar = kasm.CovarResult.load(covar_path)
    np.testing.assert_array_equal(loaded_covar.point, result.point)
    np.testing.assert_array_equal(loaded_covar.jk, result.jk)
    assert loaded_covar.n_genes == result.n_genes
    assert loaded_covar.n_pairs == result.n_pairs
    with np.load(covar_path) as archive:
        assert "diagnostics" not in archive.files
    assert loaded.score_variants(model.dataset).equals(expected_scores)


def _grouped_covar_kernel_model(synthetic_dataset_no_ld):
    ordinary = _covar_model(synthetic_dataset_no_ld)
    model = kasm.Model(
        ordinary.dataset,
        kasm.KernelOptions(
            gene_annot_cols=["constraint_low", "constraint_high"],
            kernel_groups={
                "missense": {
                    "burden_cols": ["constant", "score"],
                    "independent_col": "constant",
                    "extra_covar_kernel_fun": distance_covar_kernels,
                    "overdispersion_cols": ["constant"],
                }
            },
            overdispersion_shared=True,
        ),
    )
    model.compute_suffstats()
    return model


def test_grouped_covariance_rejects_entries_outside_group_support(
    synthetic_dataset_no_ld,
):
    frame = synthetic_dataset_no_ld.variant_df().with_columns(
        ((pl.col("gene").cum_count().over("gene") - 1) % 2 == 0)
        .cast(pl.Float64)
        .alias("is_a")
    )
    model = kasm.Model(
        kasm.Dataset(
            frame,
            gene_df=synthetic_dataset_no_ld.gene_df(),
            beta_col=synthetic_dataset_no_ld.beta_col,
            se_col=synthetic_dataset_no_ld.se_col,
            sample_size=synthetic_dataset_no_ld.sample_size,
        ),
        kasm.KernelOptions(
            kernel_groups={
                "a": {
                    "burden_cols": ["is_a"],
                    "independent_col": "is_a",
                    "extra_covar_kernel_fun": unsupported_group_covar_kernels,
                }
            }
        ),
    )
    with pytest.raises(ValueError, match="outside its kernel group support"):
        model.compute_suffstats()


def test_grouped_random_covar_rejects_rank_deficient_gram(
    synthetic_dataset_no_ld,
    monkeypatch,
):
    ordinary = _covar_model(synthetic_dataset_no_ld)
    model = kasm.Model(
        ordinary.dataset,
        kasm.KernelOptions(
            gene_annot_cols=["constraint_low", "constraint_high"],
            kernel_groups={
                "missense": {
                    "burden_cols": ["constant", "score"],
                    "independent_col": "constant",
                    "extra_covar_kernel_fun": distance_covar_kernels,
                    "overdispersion_cols": ["constant"],
                }
            },
            overdispersion_shared=True,
        ),
    )
    model.compute_suffstats()

    from kasm import fitting

    original = fitting._gram_solve

    def deficient(system, right, **kwargs):
        value, rank = original(system, right, **kwargs)
        return value, max(0, rank - 1)

    monkeypatch.setattr(fitting, "_gram_solve", deficient)
    model.compute_jk_suffstats(exclude_genes={"missense": []}, n_jk_blocks=2)
    with pytest.raises(
        np.linalg.LinAlgError, match=r"point.*rank deficient.*live parameters"
    ):
        kasm.fit_random_effects(
            model,
            extra_covar_kernel="missense:near",
            exclude_genes={"missense": []},
            fix_gene_coef=True,
            init_gene_coef={
                "missense": {"constraint_low": 0.5, "constraint_high": 1.5}
            },
            n_jk_blocks=2,
        )


@pytest.mark.parametrize("with_indep", [False, True])
@pytest.mark.parametrize("shared_exclusion", [False, True])
def test_grouped_random_system_retains_cross_group_ld_and_exclusion_intersection(
    synthetic_dataset,
    monkeypatch,
    tmp_path,
    with_indep,
    shared_exclusion,
):
    frame = (
        synthetic_dataset.variant_df()
        .with_columns(
            ((pl.col("gene").cum_count().over("gene") - 1) % 2 == 0)
            .cast(pl.Float64)
            .alias("is_a")
        )
        .with_columns((1 - pl.col("is_a")).alias("is_b"))
    )
    dataset = kasm.Dataset(
        frame,
        gene_df=synthetic_dataset.gene_df(),
        beta_col=synthetic_dataset.beta_col,
        se_col=synthetic_dataset.se_col,
        sample_size=synthetic_dataset.sample_size,
        ldm=synthetic_dataset.ldm_source,
    )
    model = kasm.Model(
        dataset,
        kasm.KernelOptions(
            maf_col="af",
            maf_bins={"all": (0, 1)},
            gene_annot_cols=["constraint_low", "constraint_high"],
            kernel_groups={
                "a": {
                    "burden_cols": ["is_a"],
                    "independent_col": "is_a",
                    "extra_covar_kernel_fun": class_a_covar_kernels,
                    "extra_covar_with_indep": with_indep,
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
    ).compute_suffstats()
    excluded = {"a": {model.genes[0]}, "b": {model.genes[1]}}
    if shared_exclusion:
        for values in excluded.values():
            values.add(model.genes[-1])
    supplied = {
        "a": {"constraint_low": 0.5, "constraint_high": 1.5},
        "b": {"constraint_low": 1.5, "constraint_high": 0.5},
    }
    observed = {}

    class Captured(Exception):
        pass

    def capture(system, _right, **_kwargs):
        observed["system"] = system.copy()
        raise Captured

    monkeypatch.setattr(kasm.fitting, "_gram_solve", capture)
    stats = model.compute_jk_suffstats(exclude_genes=excluded, n_jk_blocks=2)
    assert stats.n_genes is None
    stats.save(tmp_path / "grouped.npz")
    restored = kasm.JKSuffStats.load(tmp_path / "grouped.npz", exclude_genes=excluded)
    assert restored.excluded_genes == stats.excluded_genes
    for key in stats.blocks:
        np.testing.assert_array_equal(restored.blocks[key], stats.blocks[key])
    per_gene = model.suffstats.covar_extensions["a:within"]
    retained = np.array([gene not in excluded["a"] for gene in model.genes])
    assert len(per_gene.new_indices) == (2 if with_indep else 1)
    for column in range(len(per_gene.new_indices)):
        expected_trace = np.bincount(
            stats.gene_blocks,
            weights=per_gene.trace[:, column] * retained,
            minlength=2,
        )
        np.testing.assert_allclose(
            restored.covar_extensions["a:within"].trace[:, column, 0],
            expected_trace,
        )
    with pytest.raises(Captured):
        kasm.fit_random_effects(
            model,
            extra_covar_kernel="a:within",
            exclude_genes=excluded,
            fix_gene_coef=True,
            init_gene_coef=supplied,
            n_jk_blocks=2,
        )

    annotations = model.gene_annot_matrix()
    scale_a = annotations @ np.array([0.5, 1.5])
    scale_b = annotations @ np.array([1.5, 0.5])
    keep = np.array([gene not in excluded["a"] | excluded["b"] for gene in model.genes])
    expected = np.sum(model.suffstats.pair_trace()[:, 0, 1] * scale_a * scale_b * keep)
    assert expected != 0
    assert observed["system"][0, 1] == pytest.approx(expected)


def test_grouped_random_covariance_matches_dense_shared_overdispersion_system(
    synthetic_dataset,
):
    frame = (
        synthetic_dataset.variant_df()
        .with_columns((pl.col("gene").cum_count().over("gene") - 1).alias("gene_index"))
        .with_columns(
            pl.when(pl.col("gene") == "GENE_C")
            .then(pl.col("gene_index") < 3)
            .when(pl.col("gene") == "GENE_A")
            .then(pl.col("gene_index") == 0)
            .otherwise(pl.col("gene_index") % 2 == 0)
            .cast(pl.Float64)
            .alias("is_a")
        )
        .with_columns((1 - pl.col("is_a")).alias("is_b"))
        .drop("gene_index")
    )
    model = kasm.Model(
        kasm.Dataset(
            frame,
            gene_df=synthetic_dataset.gene_df(),
            beta_col=synthetic_dataset.beta_col,
            se_col=synthetic_dataset.se_col,
            sample_size=synthetic_dataset.sample_size,
            ldm=synthetic_dataset.ldm_source,
        ),
        kasm.KernelOptions(
            gene_annot_cols=["constraint_low", "constraint_high"],
            kernel_groups={
                "a": {
                    "burden_cols": ["is_a"],
                    "independent_col": "is_a",
                    "extra_covar_kernel_fun": class_a_near_covar_kernels,
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
    ).compute_suffstats()
    excluded = {"a": {model.genes[0]}, "b": {model.genes[1]}}
    supplied = {
        "a": {"constraint_low": 0.5, "constraint_high": 1.5},
        "b": {"constraint_low": 1.5, "constraint_high": 0.5},
    }
    model.compute_jk_suffstats(exclude_genes=excluded, n_jk_blocks=1)
    fit, covar = kasm.fit_random_effects(
        model,
        extra_covar_kernel="a:within",
        exclude_genes=excluded,
        fix_gene_coef=True,
        init_gene_coef=supplied,
        n_jk_blocks=1,
    )

    groups = ("a", "b")
    group_index = {group: index for index, group in enumerate(groups)}
    annotations = model.gene_annot_matrix()
    eta = np.column_stack(
        [
            annotations
            @ np.array(
                [
                    supplied[group]["constraint_low"],
                    supplied[group]["constraint_high"],
                ]
            )
            for group in groups
        ]
    )
    keep = np.column_stack(
        [[gene not in excluded[group] for gene in model.genes] for group in groups]
    )
    owner = np.asarray(model.layout.kernel_groups)
    scale = np.ones((len(model.genes), model.layout.n_kernels))
    heritable = np.asarray(model.layout.heritable_indices)
    scale[:, heritable] = eta[:, owner[heritable]]
    owned = owner >= 0
    scale[:, owned] *= keep[:, owner[owned]]
    for index in model.layout.overdispersion_indices:
        scale[:, index] *= keep[:, group_index[model.layout.kernel_ids[index].col[3:]]]

    n_base = model.layout.n_kernels
    system = np.zeros((n_base + 1, n_base + 1))
    right = np.zeros(n_base + 1)
    zero_cross_system = np.zeros_like(system)
    zero_cross_right = np.zeros_like(right)
    local_system = np.zeros_like(system)
    local_right = np.zeros_like(right)
    dense = []

    def add_normal(left, rhs, correlation, beta_tilde, kernels, selected=None):
        if selected is None:
            selected = np.ones(len(beta_tilde), dtype=bool)
        local_ld = correlation[np.ix_(selected, selected)]
        local_beta = beta_tilde[selected]
        local_kernels = [kernel[np.ix_(selected, selected)] for kernel in kernels]
        for i, kernel_i in enumerate(local_kernels):
            rhs[i] += (
                float(local_beta @ (kernel_i @ local_beta))
                - np.trace(kernel_i @ local_ld) / model.dataset.sample_size
            )
            for j, kernel_j in enumerate(local_kernels):
                left[i, j] += np.trace(kernel_i @ local_ld @ kernel_j @ local_ld)

    for gene_position, gene in enumerate(model.genes):
        rows = model.dataset.gene_slice(gene)
        gene_frame = model.dataset.variant_df()[rows]
        masks = [gene_frame[f"is_{group}"].to_numpy().astype(bool) for group in groups]
        residual = (
            model.dataset.load_ldm(gene).scale(model.variant_scale(gene)).toarray()
        )
        beta_tilde = model.dataset.beta[rows] / (
            model.dataset.se[rows] ** 2 * model.dataset.sample_size
        )
        base = [
            kernel * scale[gene_position, index]
            for index, kernel in enumerate(_base_dense_kernels(model, gene))
        ]
        covar_kernel = class_a_near_covar_kernels(gene_frame)["within"].toarray()
        covar_kernel *= eta[gene_position, group_index["a"]] * keep[gene_position, 0]
        kernels = [*base, covar_kernel]

        add_normal(system, right, residual, beta_tilde, kernels)
        zero_cross = residual.copy()
        zero_cross[np.ix_(masks[0], masks[1])] = 0
        zero_cross[np.ix_(masks[1], masks[0])] = 0
        add_normal(
            zero_cross_system,
            zero_cross_right,
            zero_cross,
            beta_tilde,
            kernels,
        )
        for mask in masks:
            add_normal(local_system, local_right, residual, beta_tilde, kernels, mask)
        dense.append((gene_position, masks[0], kernels))

    assert system[0, 2] != 0
    np.testing.assert_allclose(zero_cross_system, local_system, atol=1e-12)
    np.testing.assert_allclose(zero_cross_right, local_right, atol=1e-12)
    parameter_keys = [
        ("overdispersion", model.layout.kernel_ids[index].maf_label)
        if index in model.layout.overdispersion_indices
        else ("kernel", index)
        for index in range(n_base)
    ]
    parameter_index = {
        key: index for index, key in enumerate(dict.fromkeys(parameter_keys))
    }
    base_design = np.zeros((n_base, len(parameter_index)))
    base_design[np.arange(n_base), [parameter_index[key] for key in parameter_keys]] = 1
    covar_kernel_design = np.pad(base_design, ((0, 1), (0, 1)))
    covar_kernel_design[-1, -1] = 1
    covar_kernel_coef = covar_kernel_design @ np.linalg.solve(
        covar_kernel_design.T @ system @ covar_kernel_design,
        covar_kernel_design.T @ right,
    )
    np.testing.assert_allclose(fit.kernel_coef, covar_kernel_coef)
    overdispersion_coef = fit.kernel_coef[list(model.layout.overdispersion_indices)]
    np.testing.assert_allclose(overdispersion_coef, overdispersion_coef[0])

    expected = np.zeros(2)
    for gene_position, target, kernels in dense:
        if not keep[gene_position, 0]:
            continue
        covar_kernel = kernels[-1]
        ordered = covar_kernel.sum()
        if ordered == 0:
            continue
        effect = sum(
            covar_kernel_coef[index] * kernels[index]
            for index in [*model.layout.heritable_indices, n_base]
        )
        degree = covar_kernel.sum(axis=1)
        expected += [
            np.sum(covar_kernel * effect) / ordered,
            degree @ np.diag(effect) / ordered,
        ]
    np.testing.assert_allclose(covar.point, expected)


def _projection_model(*, cross=True):
    rng = np.random.default_rng(71)
    n_genes, size = 6, 8
    a = np.tile(np.r_[np.ones(4), np.zeros(4)], n_genes)
    frame = pl.DataFrame(
        {
            "gene": np.repeat([f"g{i}" for i in range(n_genes)], size),
            "variant": [f"1:{i + 1}:A:C" for i in range(n_genes * size)],
            "beta": rng.normal(0.1, 0.03, n_genes * size),
            "se": rng.uniform(0.02, 0.04, n_genes * size),
            "a": a,
            "b": 1 - a,
        }
    )
    return kasm.Model(
        kasm.Dataset(frame, beta_col="beta", se_col="se", sample_size=1000),
        kasm.KernelOptions(
            kernel_groups={
                "a": {"burden_cols": ["a"], "independent_col": "a"},
                "b": {"burden_cols": ["b"], "independent_col": "b"},
            },
            covariance_between_kernel_groups={"a_x_b": ("a", "b")} if cross else None,
        ),
    ).compute_suffstats()


@pytest.mark.parametrize("selection", [("a", "a"), ("a", "b"), ("b", "b")])
@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize("per_group", [False, True])
@pytest.mark.parametrize("n_blocks", [1, 3])
def test_compute_random_covar_uses_existing_coefficients_and_exclusions(
    cross, per_group, n_blocks, monkeypatch, tmp_path, selection
):
    from dataclasses import replace

    model = _projection_model(cross=cross)
    excluded = {"a": ["g0"], "b": ["g1"]} if per_group else ["g0"]
    fit_input = model
    if not per_group:
        # A noncontiguous partition must survive fitting and persistence.
        fit_input = model.compute_jk_suffstats(
            exclude_genes=excluded,
            gene_blocks=np.arange(6) % n_blocks,
            n_jk_blocks=n_blocks,
        )
    else:
        model.compute_jk_suffstats(
            exclude_genes=excluded,
            n_jk_blocks=n_blocks,
        )

    def unexpected(*args, **kwargs):
        pytest.fail("ordinary fitting must not compute covariance projections")

    with monkeypatch.context() as context:
        context.setattr(kasm.covar, "_group_projection", unexpected)
        fit = kasm.fit_random_effects(
            fit_input,
            exclude_genes=excluded,
            fix_gene_coef=True,
            init_gene_coef={
                name: {"constant": 1.0} for name in model.layout.kernel_group_names
            },
            n_jk_blocks=n_blocks,
        )
    if cross and per_group:
        row = fit.point.filter(pl.col("kernel_group") == "a_x_b").row(0, named=True)
        assert row["n_genes"] == 4
        assert row["n_variants"] == 8 * row["n_genes"]
    assert isinstance(fit, kasm.RandomFit)
    assert fit.kernel_coef.shape == (model.n_kernels,)
    # Convergence is diagnostic-only: finite last iterates must still project.
    fit = replace(fit, converged=False, converged_jk=np.zeros(n_blocks, dtype=bool))
    path = tmp_path / "fit.npz"
    fit.save(path)
    fit = kasm.RandomFit.load(path)
    with np.load(path) as stored:
        assert not any("covar" in name for name in stored.files)

    def no_refit(*args, **kwargs):
        pytest.fail("projection must not refit or recompute sufficient statistics")

    monkeypatch.setattr(kasm.fitting, "moment_solve", no_refit)
    monkeypatch.setattr(kasm.Model, "compute_suffstats", no_refit)
    monkeypatch.setattr(kasm, "fit_random_effects", no_refit)
    # An equivalent caller-owned Model is sufficient; no identity requirement.
    corresponding = kasm.Model(model.dataset, model.options)
    result = kasm.compute_random_covar(
        corresponding,
        fit,
        **(
            {"group_col": selection[0]}
            if selection[0] == selection[1]
            else {"group_col1": selection[0], "group_col2": selection[1]}
        ),
    )
    point, jk = result.point, result.jk
    assert isinstance(point, np.ndarray) and isinstance(jk, np.ndarray)

    def dense(theta, eta, block=None):
        total = np.zeros(5)
        for index, gene in enumerate(model.genes):
            if block is not None and n_blocks > 1 and fit.gene_blocks[index] == block:
                continue
            scales = eta[0, fit.kernel_groups].copy()
            for column, owner in enumerate(fit.kernel_groups):
                omitted = (
                    (
                        set(excluded["a"]) | set(excluded["b"])
                        if fit.kernel_group_names[owner] == "a_x_b"
                        else excluded[fit.kernel_group_names[owner]]
                    )
                    if per_group
                    else excluded
                )
                if gene in omitted:
                    scales[column] = 0
            covariance = np.einsum(
                "k,kij->ij", theta * scales, _base_dense_kernels(model, gene)
            )
            a, b = covariance[:4, :4], covariance[4:, 4:]
            total += [
                (a.sum() - np.trace(a)) / 12,
                covariance[:4, 4:].mean(),
                (b.sum() - np.trace(b)) / 12,
                np.trace(a) / 4,
                np.trace(b) / 4,
            ]
        columns = {("a", "a"): [0, 3], ("a", "b"): [1], ("b", "b"): [2, 4]}
        return total[columns[selection]]

    np.testing.assert_allclose(point, dense(fit.kernel_coef, fit.gene_coef), atol=1e-14)
    for block in range(n_blocks):
        np.testing.assert_allclose(
            jk[block],
            dense(fit.kernel_coef_jk[block], fit.gene_coef_jk[block], block),
            atol=1e-14,
        )
    if n_blocks == 1:
        np.testing.assert_array_equal(jk[0], point)
        assert np.isnan(kasm.jackknife_se(jk)).all()
    else:
        assert np.isfinite(kasm.jackknife_se(jk)).all()


def test_covariance_selector_requires_one_string(synthetic_dataset_no_ld):
    model = _grouped_covar_kernel_model(synthetic_dataset_no_ld)
    selection = {"exclude_genes": []}
    model.compute_jk_suffstats(exclude_genes=[])
    with pytest.raises(TypeError, match="extra_covar_kernel must be a string"):
        kasm.fit_random_effects(
            model, extra_covar_kernel=["missense:near"], **selection
        )


def test_covariance_request_cannot_be_combined_with_enrichment(synthetic_dataset_no_ld):
    model = _grouped_covar_kernel_model(synthetic_dataset_no_ld)
    selection = {"exclude_genes": []}
    model.compute_jk_suffstats(exclude_genes=[])
    with pytest.raises(ValueError, match="enrichment cannot be combined"):
        kasm.fit_random_effects(
            model,
            extra_covar_kernel="missense:near",
            extra_enrich_col="score",
            **selection,
        )


@pytest.mark.parametrize("values", [[0, 2], [0, None], [0, float("nan")], ["0", "1"]])
def test_covariance_indicator_validation(values):
    frame = pl.DataFrame({"indicator": values}, strict=False)
    model = SimpleNamespace(
        layout=SimpleNamespace(
            diagonal_indices=(), burden_indices=(), cross_pairs=(), cross_indices=()
        ),
        genes=("g",),
        dataset=SimpleNamespace(variant_df=lambda gene: frame),
    )
    with pytest.raises(ValueError, match="0/1"):
        kasm.covar._group_projection(model, "indicator", "indicator")


def test_covar_extension_jk_cache_preserves_rectangular_moments_and_support(
    synthetic_dataset_no_ld, monkeypatch, tmp_path
):
    model = _grouped_covar_kernel_model(synthetic_dataset_no_ld)
    stats = model.suffstats
    blocks = np.array([2, 0, 2, 1, 0])
    excluded = [model.genes[1]]
    monkeypatch.setattr(
        model.dataset, "load_ldm", lambda *args: pytest.fail("JK conversion loaded LD")
    )
    monkeypatch.setattr(
        kasm.suffstats,
        "compute_gene_suffstats",
        lambda **kwargs: pytest.fail("JK conversion recomputed gene moments"),
    )
    jk = model.compute_jk_suffstats(
        exclude_genes=excluded, gene_blocks=blocks, n_jk_blocks=4
    )
    assert jk.covar_extensions
    assert model.suffstats is stats
    assert tuple(jk.layout.kernel_ids) == tuple(model.layout.kernel_ids)
    for _, name, selected, cached in extension_designs(
        jk, extra_covar_kernels=list(stats.covar_extensions)
    ):
        value = stats.covar_extensions[name]
        assert cached["kernel_group"] == value.payload.kernel_group
        assert selected.blocks["pt0"].shape == (
            4,
            model.n_kernels + 1,
            model.n_kernels + 1,
        )
        for block in range(4):
            keep = (blocks == block) & np.array(
                [gene not in excluded for gene in model.genes]
            )
            annot = model.gene_annot_matrix()[keep]
            for prefix, key in (
                ("pt", "pair_trace"),
                ("tr", "trace"),
                ("qf", "quadform"),
            ):
                np.testing.assert_allclose(
                    selected.blocks[prefix + "0"][block, -1:],
                    getattr(value, key)[keep].sum(axis=0),
                    atol=1e-16,
                )
                expected = np.einsum("g...,ga->...a", getattr(value, key)[keep], annot)
                np.testing.assert_allclose(
                    selected.blocks[prefix + "1"][block, -1:], expected, atol=1e-16
                )
            for key in kasm.suffstats.COVAR_PROJECTION_KEYS:
                np.testing.assert_allclose(
                    cached[key][block],
                    getattr(value.payload, key)[keep].T @ annot,
                    atol=1e-16,
                )
            assert cached["n_genes"][block] == value.payload.n_genes[keep].sum()
            assert cached["n_pairs"][block] == value.payload.n_pairs[keep].sum()
        path = tmp_path / f"{name.replace(':', '-')}.npz"
        selected.save(path)
        loaded = kasm.JKSuffStats.load(path, exclude_genes=excluded, n_jk_blocks=4)
        np.testing.assert_array_equal(loaded.gene_blocks, blocks)
        assert loaded.layout == selected.layout
        for key, expected in selected.blocks.items():
            np.testing.assert_array_equal(loaded.blocks[key], expected)
        parent_path = tmp_path / "parent.npz"
        jk.save(parent_path)
        parent = kasm.JKSuffStats.load(parent_path)
        for key, expected in cached.items():
            np.testing.assert_array_equal(
                getattr(parent.covar_extensions[name].payload, key), expected
            )
    with pytest.raises(ValueError, match="exclusions do not match"):
        kasm.fit_random_effects(
            loaded,
            exclude_genes=[],
            n_jk_blocks=4,
            init_gene_coef={
                "missense": {"constraint_low": 1.0, "constraint_high": 1.0}
            },
            fix_gene_coef=True,
        )


def test_burden_and_covar_extension_names_roundtrip_independently(
    synthetic_dataset_no_ld, tmp_path
):
    from dataclasses import replace

    source = _covar_model(synthetic_dataset_no_ld).dataset
    dataset = kasm.Dataset(
        source.variant_df().with_columns(pl.col("buried").alias("all:near")),
        gene_df=source.gene_df(),
        beta_col=source.beta_col,
        se_col=source.se_col,
        sample_size=source.sample_size,
    )
    grouped = _covar_model(dataset).compute_suffstats()
    model = kasm.Model(
        dataset,
        kasm.KernelOptions(
            burden_cols=["constant", "score"], independent_col="constant"
        ),
    ).compute_suffstats(extra_enrich_cols=["all:near"])
    parent = replace(
        model.suffstats, covar_extensions=grouped.suffstats.covar_extensions
    )
    path = tmp_path / "both.npz"
    parent.save(path)
    loaded = kasm.SuffStats.load(path)
    assert set(loaded.burden_extensions) == {"all:near"}
    assert set(loaded.covar_extensions) == {"all:near", "all:far"}
    assert loaded.burden_extensions["all:near"].payload.burden_cov is not None
    model.suffstats = loaded
    jk = model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=2)
    jk.save(tmp_path / "both-jk.npz")
    restored = kasm.JKSuffStats.load(tmp_path / "both-jk.npz")
    assert restored.layout == jk.layout
    designs = list(
        extension_designs(
            restored,
            extra_enrich_cols=["all:near"],
            extra_covar_kernels=["all:near", "all:far"],
        )
    )
    assert [(kind, name) for kind, name, _, _ in designs] == [
        ("burden", "all:near"),
        ("covar", "all:near"),
        ("covar", "all:far"),
    ]
    assert all(
        not value.burden_extensions and not value.covar_extensions
        for _, _, value, _ in designs
    )


_CLASS_GROUPS = {
    "missense": {"independent_col": "is_missense"},
    "plof": {"independent_col": "is_plof"},
}


def test_group_projection_matches_dense_kernel_oracle():
    burden_wgts = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    diagonal = np.array(
        [[1.0, 0.0, 2.0], [1.0, 0.0, 3.0], [0.0, 1.0, 4.0], [0.0, 1.0, 5.0]]
    )
    mask = np.array([True, True, False, False])
    layout = SimpleNamespace(
        n_kernels=5,
        burden_indices=(0,),
        cross_indices=(1,),
        cross_pairs=((0, 1),),
        independent_indices=(2, 3),
        overdispersion_indices=(4,),
        diagonal_indices=(2, 3, 4),
    )
    dataset = SimpleNamespace(
        variant_df=lambda gene: pl.DataFrame({"is_missense": mask, "is_plof": ~mask}),
    )
    model = SimpleNamespace(
        layout=layout,
        dataset=dataset,
        genes=("g",),
        options=SimpleNamespace(kernel_groups=_CLASS_GROUPS),
        burden_wgts=lambda gene: burden_wgts,
        diag_kernel_weights=lambda gene: diagonal,
    )
    kernels = [
        np.outer(burden_wgts[:, 0], burden_wgts[:, 0]),
        np.outer(burden_wgts[:, 0], burden_wgts[:, 1])
        + np.outer(burden_wgts[:, 1], burden_wgts[:, 0]),
        np.diag(diagonal[:, 0]),
        np.diag(diagonal[:, 1]),
        np.zeros((4, 4)),
    ]
    m, l = mask.sum(), (~mask).sum()
    expected = []
    for kernel in kernels:
        expected.append(
            [
                (
                    kernel[np.ix_(mask, mask)].sum()
                    - np.trace(kernel[np.ix_(mask, mask)])
                )
                / (m * (m - 1)),
                kernel[np.ix_(mask, ~mask)].sum() / (m * l),
                (
                    kernel[np.ix_(~mask, ~mask)].sum()
                    - np.trace(kernel[np.ix_(~mask, ~mask)])
                )
                / (l * (l - 1)),
                np.trace(kernel[np.ix_(mask, mask)]) / m,
                np.trace(kernel[np.ix_(~mask, ~mask)]) / l,
            ]
        )
    for group, group2, columns, n_pairs in [
        ("missense", "missense", [0, 3], 1),
        ("missense", "plof", [1], 4),
        ("plof", "missense", [1], 4),
        ("plof", "plof", [2, 4], 1),
    ]:
        projection, pairs = kasm.covar._group_projection(
            model, f"is_{group}", f"is_{group2}"
        )
        np.testing.assert_allclose(projection[0], np.asarray(expected)[:, columns])
        np.testing.assert_array_equal(pairs, [n_pairs])


def test_random_projections_are_gene_additive_with_refitted_exact_complements(
    monkeypatch,
):
    projection = np.arange(9, dtype=float).reshape(3, 3, 1) / 10
    annotations = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    groups = np.array([0, 1, -1])
    point_gene_coef = np.array([[1.0, 2.0], [3.0, 4.0]])
    point_theta = np.array([0.5, 2.0, 100.0])
    delete_gene_coef = np.stack([point_gene_coef + i for i in range(3)])
    delete_theta = np.array(
        [[10.0, 20.0, 100.0], [30.0, 40.0, 100.0], [50.0, 60.0, 100.0]]
    )
    model = SimpleNamespace(
        genes=("g0", "g1", "g2"),
        n_kernels=3,
        options=SimpleNamespace(
            kernel_groups={"a": {}, "b": {}}, within_gene_group_col=None
        ),
        layout=SimpleNamespace(
            kernel_ids=[SimpleNamespace(label=label) for label in ("B:a", "B:b", "O:c")]
        ),
        gene_annot_matrix=lambda: annotations,
    )
    blocks = np.arange(3)
    fit = SimpleNamespace(
        excluded_genes=("g1",),
        gene_blocks=blocks,
        gene_coef=point_gene_coef,
        kernel_group_names=("a", "b"),
        kernel_options={},
        kernel_labels=("B:a", "B:b", "O:c"),
        heritable_labels=("B:a", "B:b"),
        kernel_groups=groups,
        kernel_coef=point_theta,
        kernel_coef_jk=delete_theta,
        gene_coef_jk=delete_gene_coef,
        n_jk_blocks=3,
    )
    monkeypatch.setattr(
        kasm.covar,
        "_group_projection",
        lambda _model, _group, _group2: (projection, np.ones(3, dtype=int)),
    )

    result = kasm.compute_random_covar(model, fit, group_col1="a", group_col2="b")
    point, jk = result.point, result.jk
    assert point.shape == (1,) and jk.shape == (3, 1)

    def expected(theta, gene_coef, keep):
        eta = annotations @ gene_coef
        scale = np.zeros((3, 3))
        scale[:, :2] = eta
        return np.einsum("gk,gkm,k->m", scale[keep], projection[keep], theta)

    np.testing.assert_allclose(
        point, expected(point_theta, point_gene_coef, [True, False, True])
    )
    for block in range(3):
        keep = np.array([True, False, True]) & (blocks != block)
        np.testing.assert_allclose(
            jk[block],
            expected(delete_theta[block], delete_gene_coef[block], keep),
        )


@pytest.mark.parametrize("with_indep", [False, True])
@pytest.mark.parametrize("n_blocks", [1, 3])
def test_selected_self_matches_all_pairs_extension(
    synthetic_dataset, with_indep, n_blocks
):
    """Compare real LD/MAF/annotation/JK contractions with the former baseline path."""
    source = synthetic_dataset
    frame = (
        source.variant_df()
        .with_columns(
            ((pl.col("gene").cum_count().over("gene") - 1) % 2 == 0)
            .cast(pl.Float64)
            .alias("is_a")
        )
        .with_columns((1 - pl.col("is_a")).alias("is_b"))
    )
    dataset = kasm.Dataset(
        frame,
        gene_df=source.gene_df(),
        beta_col=source.beta_col,
        se_col=source.se_col,
        sample_size=source.sample_size,
        ldm=source.ldm_source,
    )
    model = kasm.Model(
        dataset,
        kasm.KernelOptions(
            maf_col="af",
            maf_bins={"all": (0, 1)},
            gene_annot_cols=["constraint_low", "constraint_high"],
            kernel_groups={
                "a": {
                    "burden_cols": ["is_a"],
                    "independent_col": "is_a",
                    "overdispersion_cols": ["is_a"],
                    "extra_covar_kernel_fun": class_a_covar_kernels,
                    "extra_covar_with_indep": with_indep,
                },
                "b": {
                    "burden_cols": ["is_b"],
                    "independent_col": "is_b",
                    "overdispersion_cols": ["is_b"],
                },
            },
            overdispersion_shared=True,
        ),
    ).compute_suffstats()
    stats = model.compute_jk_suffstats(
        exclude_genes={"a": {model.genes[0]}, "b": {model.genes[1]}},
        n_jk_blocks=1,
    )
    fit = kasm.fit_random_effects(
        stats,
        fix_gene_coef=True,
        init_gene_coef={
            "a": {"constraint_low": 0.5, "constraint_high": 1.5},
            "b": {"constraint_low": 1.5, "constraint_high": 0.5},
        },
    )
    # Isolate projection equivalence from rank-deficient deletes in this tiny LD fixture.
    # Other tests above exercise actual fitted delete coefficients against dense kernels.
    from dataclasses import replace

    stats = model.compute_jk_suffstats(
        exclude_genes={"a": {model.genes[0]}, "b": {model.genes[1]}},
        n_jk_blocks=n_blocks,
    )
    fit = replace(
        fit,
        gene_blocks=stats.gene_blocks,
        n_jk_blocks=n_blocks,
        kernel_coef_jk=np.stack(
            [fit.kernel_coef * (1 + block / 10) for block in range(n_blocks)]
        ),
        gene_coef_jk=np.repeat(fit.gene_coef[None], n_blocks, axis=0),
    )
    old = kasm.covar._project_covar_fit(stats, fit, stats.covar_extensions["a:within"])
    new = kasm.compute_random_covar(
        model, fit, group_col="is_a", include_genes=set(model.genes) - {model.genes[0]}
    )
    np.testing.assert_allclose(new.point[:2], old.point, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(new.jk[:, :2], old.jk, rtol=1e-11, atol=1e-13)
    assert (new.n_genes, new.n_pairs) == (old.n_genes, old.n_pairs)
    dataset.close()


def test_selected_readout_support_and_group_symmetry():
    source = _projection_model()
    # g0 has one B, g1 has no B, g2 has one A; other genes retain four each.
    frame = source.dataset.variant_df().filter(
        ~(
            (pl.col("gene") == "g0")
            & (pl.col("b") > 0)
            & (pl.col("variant") != "1:5:A:C")
        )
        & ~((pl.col("gene") == "g1") & (pl.col("b") > 0))
        & ~(
            (pl.col("gene") == "g2")
            & (pl.col("a") > 0)
            & (pl.col("variant") != "1:17:A:C")
        )
    )
    model = kasm.Model(
        kasm.Dataset(frame, beta_col="beta", se_col="se", sample_size=1000),
        source.options,
    )
    fit = kasm.fit_random_effects(
        model,
        exclude_genes=set(),
        n_jk_blocks=3,
        fix_gene_coef=True,
        init_gene_coef={
            name: {"constant": 1.0} for name in model.layout.kernel_group_names
        },
    )
    aa = kasm.compute_random_covar(model, fit, group_col="a")
    ab = kasm.compute_random_covar(model, fit, group_col1="a", group_col2="b")
    ba = kasm.compute_random_covar(model, fit, group_col1="b", group_col2="a")
    assert (aa.n_genes, aa.n_pairs) == (5, 30)
    assert (ab.n_genes, ab.n_pairs) == (5, 56)
    np.testing.assert_allclose(ab.point, ba.point)
    np.testing.assert_allclose(ab.jk, ba.jk)
    common = {"g3", "g4", "g5"}
    aa = kasm.compute_random_covar(model, fit, group_col="a", include_genes=common)
    bb = kasm.compute_random_covar(model, fit, group_col="b", include_genes=common)
    ab = kasm.compute_random_covar(
        model, fit, group_col1="a", group_col2="b", include_genes=common
    )
    assert aa.point.shape == bb.point.shape == (2,)
    assert ab.point.shape == (1,)
    assert aa.n_genes == bb.n_genes == ab.n_genes == 3
    for kwargs, message in [
        ({}, "provide group"),
        ({"group_col1": "a"}, "provide group"),
        ({"group_col": "a", "group_col2": "b"}, "not both"),
        ({"group_col1": "a", "group_col2": "a"}, "use group"),
        ({"group_col": "missing"}, "unknown covariance indicator"),
        ({"group_col": "a", "include_genes": {"missing"}}, "unknown projection genes"),
        ({"group_col": "a", "include_genes": {"g2"}}, "no eligible genes"),
    ]:
        with pytest.raises(ValueError, match=message):
            kasm.compute_random_covar(model, fit, **kwargs)
    model.dataset.close()
    source.dataset.close()


@pytest.mark.parametrize("with_independent", [False, True])
def test_indicator_readouts_without_kernel_groups(with_independent):
    source = _projection_model()
    frame = source.dataset.variant_df().with_columns(
        pl.col("a").alias("selected_a"),
        pl.col("b").alias("selected_b"),
        pl.lit(1.0).alias("constant"),
    )
    model = kasm.Model(
        kasm.Dataset(frame, beta_col="beta", se_col="se", sample_size=1000),
        kasm.KernelOptions(
            burden_cols=["a", "b"],
            independent_col="constant" if with_independent else None,
        ),
    )
    fit = kasm.fit_random_effects(
        model,
        exclude_genes={"g0"},
        n_jk_blocks=3,
        fix_gene_coef=True,
        init_gene_coef={"constant": 1.0},
    )
    aa = kasm.compute_random_covar(model, fit, group_col="selected_a")
    ab = kasm.compute_random_covar(
        model, fit, group_col1="selected_a", group_col2="selected_b"
    )
    assert not model.options.kernel_groups
    assert aa.n_genes == ab.n_genes == 5

    def expected(theta, eta, block=None):
        result = np.zeros(3)
        for i, gene in enumerate(model.genes):
            if gene == "g0" or (block is not None and fit.gene_blocks[i] == block):
                continue
            effect = np.einsum(
                "k,kij->ij",
                theta * eta[0, fit.kernel_groups],
                _base_dense_kernels(model, gene),
            )
            a = effect[:4, :4]
            result += [
                (a.sum() - np.trace(a)) / 12,
                np.trace(a) / 4,
                effect[:4, 4:].mean(),
            ]
        return result

    np.testing.assert_allclose(
        np.r_[aa.point, ab.point], expected(fit.kernel_coef, fit.gene_coef), atol=1e-13
    )
    for block in range(3):
        np.testing.assert_allclose(
            np.r_[aa.jk[block], ab.jk[block]],
            expected(fit.kernel_coef_jk[block], fit.gene_coef_jk[block], block),
            atol=1e-13,
        )
    with pytest.raises(ValueError, match="overlap"):
        kasm.compute_random_covar(model, fit, group_col1="a", group_col2="selected_a")
    model.dataset.close()
    source.dataset.close()
