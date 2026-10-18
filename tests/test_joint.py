"""Joint hsq across models under the fixed sharing contract."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from conftest import GENE_ANNOT_COLS, MAF_BINS

from kasm import (
    Dataset,
    KernelOptions,
    Model,
    fit_random_effects,
    fitting,
)
from kasm.random_effects import (
    _fit_random_effects as estimate_joint_random_hsq,
)


def class_options(**overrides) -> KernelOptions:
    settings = {
        "maf_col": "AF",
        "maf_bins": MAF_BINS,
        "gene_annot_cols": GENE_ANNOT_COLS,
        "burden_cols": ["constant"],
        "burden_by_maf": True,
        "overdispersion_cols": ["constant"],
        "overdispersion_by_maf": True,
    }
    settings.update(overrides)
    return KernelOptions(**settings)


def class_model(test_dataset, variant_class, **overrides) -> Model:
    frame = test_dataset.variant_df()
    subset = frame.filter(pl.col("variant_class") == variant_class)
    dataset = Dataset(
        subset,
        gene_df=test_dataset.gene_df(),
        beta_col="BETA",
        se_col="SE",
        sample_size=test_dataset.sample_size,
        ldm=test_dataset.ldm_source,
    )
    model = Model(dataset, class_options(**overrides))
    model.compute_suffstats()
    return model


def prepare_jk(models, exclude_genes, n_jk_blocks):
    exclusions = (
        exclude_genes
        if isinstance(exclude_genes, dict)
        else {label: exclude_genes for label in models}
    )
    return {
        label: model.compute_jk_suffstats(
            exclude_genes=exclusions[label], n_jk_blocks=n_jk_blocks
        )
        for label, model in models.items()
    }


@pytest.fixture
def class_models(test_dataset):
    return {
        label: class_model(test_dataset, label)
        for label in ("missense", "lof", "synonymous")
    }


def test_joint_gene_coefficient_update_is_nonnegative():
    totals = {
        "pt0": np.zeros((1, 1)),
        "pt1": np.array([[[1.0, 1.0]]]),
        "tr0": np.zeros(1),
        "tr1": np.zeros((1, 2)),
        "qf0": np.zeros(1),
        "qf1": np.array([[-1.0, 3.0]]),
        "ga1": np.ones(2),
        "ng": 2,
    }
    item = fitting.RandomEffectsInput(
        totals, np.array([0]), np.array([], dtype=np.intp), 1.0, np.array([0])
    )
    pair, trace, quad, groups = fitting.gene_coef_system(item, np.array([1.0]))
    constrained = fitting._constrained_grouped_simplex_solve(pair, quad - trace, groups)
    np.testing.assert_allclose(constrained, [0.0, 2.0], atol=1e-8)


def test_joint_gene_coef_have_mean_one_per_model():
    def solve(amplitude):
        totals = {
            "pt0": np.zeros((1, 1)),
            "pt1": np.ones((1, 1, 2)),
            "tr0": np.zeros(1),
            "tr1": np.zeros((1, 2)),
            "qf0": np.zeros(1),
            "qf1": amplitude * np.array([[1.0, 3.0]]),
            "ga1": np.ones(2),
            "ng": 2,
        }
        item = fitting.RandomEffectsInput(
            totals, np.array([0]), np.array([], dtype=int), 1.0, np.array([0])
        )
        pair, trace, quad, groups = fitting.gene_coef_system(item, np.ones(1))
        return fitting._constrained_grouped_simplex_solve(pair, quad - trace, groups)

    np.testing.assert_allclose([solve(1).mean(), solve(10).mean()], 1)


def test_joint_gene_update_profiles_model_local_overdispersion():
    def make_item(diagonal, cross, overdispersion, right):
        pt0 = np.zeros((2, 2))
        pt0[1, 1] = overdispersion
        pt1 = np.zeros((2, 2, 2))
        pt1[0, 0] = diagonal
        pt1[0, 1] = cross
        totals = {
            "pt0": pt0,
            "pt1": pt1,
            "tr0": np.zeros(2),
            "tr1": np.zeros((2, 2)),
            "qf0": np.array([0.0, right[2]]),
            "qf1": np.array([right[:2], np.zeros(2)]),
            "ga1": np.ones(2),
            "ng": 2,
        }
        return totals

    models = {
        "first": make_item([4.0, 3.0], [1.0, -0.2], 2.0, [-1.0, 4.0, 0.5]),
        "second": make_item([2.0, 5.0], [-0.5, 0.8], 3.0, [3.0, -2.0, -0.3]),
    }
    weights = {"first": 0.5, "second": 2.0}
    items = {
        label: fitting.RandomEffectsInput(
            totals, np.array([0, -1]), np.array([1]), 1.0, np.array([0, 1])
        )
        for label, totals in models.items()
    }
    systems = {
        label: fitting.gene_coef_system(item, np.array([1.0, 0.0]))
        for label, item in items.items()
    }
    updated = {
        label: fitting._constrained_grouped_simplex_solve(
            system, quadform - trace, groups
        )[:2]
        for label, (system, trace, quadform, groups) in systems.items()
    }

    import cvxpy

    value = cvxpy.Variable(6)
    objective = 0
    for label, (system, trace, quadform, _groups) in systems.items():
        indices = [0, 1, 4] if label == "first" else [2, 3, 5]
        local = value[indices]
        objective += weights[label] * (
            0.5 * cvxpy.quad_form(local, cvxpy.psd_wrap(system))
            - (quadform - trace) @ local
        )
    problem = cvxpy.Problem(
        cvxpy.Minimize(objective),
        [
            value[:2] >= 0,
            cvxpy.sum(value[:2]) == 2,
            value[2:4] >= 0,
            cvxpy.sum(value[2:4]) == 2,
        ],
    )
    problem.solve(solver=cvxpy.CLARABEL)

    np.testing.assert_allclose(updated["first"], value.value[:2], atol=1e-7)
    np.testing.assert_allclose(updated["second"], value.value[2:4], atol=1e-7)


def test_joint_one_block_is_a_point_only_request(class_models):
    result = estimate_joint_random_hsq(
        prepare_jk(class_models, [], 1),
        n_jk_blocks=1,
    )

    for fit in result.values():
        np.testing.assert_array_equal(fit.hsq_jk, [fit.hsq])
        assert np.isnan(fit.hsq_se)
        assert np.isnan(fit.kernel_coef_se).all()


def test_joint_nonconvergence_retains_finite_standard_errors(class_models, monkeypatch):
    original = fitting.alternating_fit
    calls = 0

    def report_failed_delete_block(*args, **kwargs):
        nonlocal calls
        calls += 1
        results = original(*args, **kwargs)
        if calls == 2:
            results = [{**result, "converged": False} for result in results]
        return results

    monkeypatch.setattr(fitting, "alternating_fit", report_failed_delete_block)
    result = estimate_joint_random_hsq(
        prepare_jk(class_models, [], 3),
        n_jk_blocks=3,
    )

    for fit in result.values():
        assert not fit.converged_jk.all()
        assert np.isfinite(fit.hsq_se)


def test_two_model_fit_and_jackknife_respects_weights(class_models):
    models = {label: class_models[label] for label in ("missense", "lof")}
    weights = {"missense": 0.5, "lof": 2.0}
    weighted = estimate_joint_random_hsq(
        prepare_jk(models, [], 3),
        weights=weights,
        n_jk_blocks=3,
    )
    unweighted = estimate_joint_random_hsq(prepare_jk(models, [], 3), n_jk_blocks=3)
    for label in models:
        assert weighted[label].weight == pytest.approx(
            weights[label] / np.mean(list(weights.values()))
        )
        assert weighted[label].hsq_jk.shape == (3,)
    assert any(weighted[label].hsq != unweighted[label].hsq for label in models)


def test_fixed_contract_shares_burden_but_keeps_other_terms_local(class_models):
    result = estimate_joint_random_hsq(prepare_jk(class_models, [], 2), n_jk_blocks=2)
    reference_label = next(iter(class_models))
    reference = result[reference_label]
    reference_model = class_models[reference_label]
    for index, kernel_id in enumerate(reference_model.layout.kernel_ids):
        if kernel_id.kind != "B":
            continue
        np.testing.assert_allclose(
            [
                fit.kernel_coef[model.layout.kernel_ids.index(kernel_id)]
                for fit, model in zip(
                    result.values(), class_models.values(), strict=True
                )
            ],
            reference.kernel_coef[index],
        )
    assert any(
        not np.array_equal(
            fit.kernel_coef[list(model.layout.overdispersion_indices)],
            reference.kernel_coef[list(reference_model.layout.overdispersion_indices)],
        )
        for fit, model in list(
            zip(result.values(), class_models.values(), strict=True)
        )[1:]
    )
    assert any(
        not np.array_equal(fit.gene_coef, reference.gene_coef)
        for fit in list(result.values())[1:]
    )


def test_per_model_heritabilities_are_reported(class_models):
    result = estimate_joint_random_hsq(prepare_jk(class_models, [], 4), n_jk_blocks=4)
    assert set(result) == set(class_models)
    assert all(fit.n_genes > 0 for fit in result.values())
    assert all(len(fit.hsq_jk) == fit.n_jk_blocks for fit in result.values())
    for fit in result.values():
        assert fit.residual == pytest.approx(1.0 - fit.variance.sum())
        assert fit.hsq_se == pytest.approx(fitting.jackknife_se(fit.hsq_jk))


def test_joint_accepts_model_specific_gene_exclusions(class_models):
    excluded = {label: [model.genes[0]] for label, model in class_models.items()}
    result = estimate_joint_random_hsq(
        prepare_jk(class_models, excluded, 3),
        n_jk_blocks=3,
    )
    assert {label: fit.n_genes for label, fit in result.items()} == {
        label: len(model.genes) - 1 for label, model in class_models.items()
    }


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_requires_models_and_valid_explicit_exclusions(class_models):
    from kasm import fit_random_effects

    with pytest.raises(ValueError, match="must not be empty"):
        fit_random_effects({}, exclude_genes=[])
    with pytest.raises(KeyError, match="not in this model"):
        prepare_jk(class_models, ["NOT_A_GENE"], 1)


def test_combination_leaves_variant_overlap_responsibility_to_caller(class_models):
    model = class_models["missense"]
    result = estimate_joint_random_hsq(
        prepare_jk({"first": model, "second": model}, [], 2),
        n_jk_blocks=2,
    )
    assert len(result) == 2
    from kasm.api import combine_random_hsq

    hsq, hsq_se = combine_random_hsq(result)
    assert hsq == sum(fit.hsq for fit in result.values())
    assert hsq_se == pytest.approx(
        fitting.jackknife_se(sum(fit.hsq_jk for fit in result.values())[:, None])[0]
    )


def test_shared_burden_requires_matching_maf_bounds(test_dataset):
    first = class_model(test_dataset, "missense")
    incompatible_bins = dict(MAF_BINS)
    incompatible_bins["AF_RARE"] = (1e-5, 2e-4)
    incompatible_bins["AF_LF"] = (2e-4, 1)
    second = class_model(test_dataset, "synonymous", maf_bins=incompatible_bins)

    with pytest.raises(ValueError, match="inconsistent MAF bounds"):
        estimate_joint_random_hsq(
            prepare_jk({"first": first, "second": second}, [], 2),
            n_jk_blocks=2,
        )


def test_blocks_are_left_out_together_across_models(class_models):
    """Joint replicate b deletes each model's own local block b."""
    result = estimate_joint_random_hsq(prepare_jk(class_models, [], 4), n_jk_blocks=4)
    for fit in result.values():
        assert fit.n_jk_blocks == 4
        assert fit.hsq_jk.shape == (4,)
        assert not np.any(np.isclose(fit.hsq_jk, fit.hsq, rtol=1e-12))


def test_each_model_owns_its_block_assignment_when_genes_differ(test_dataset):
    full = class_model(test_dataset, "missense")
    missing = set(full.dataset.genes[4:8])
    frame = full.dataset.variant_df()
    keep = ~pl.col("gene").is_in(missing)
    reduced = Model(
        Dataset(
            frame.filter(keep),
            gene_df=full.dataset.gene_df(),
            beta_col="BETA",
            se_col="SE",
            sample_size=full.dataset.sample_size,
            ldm=full.dataset.ldm_source,
        ),
        class_options(),
    )

    reduced.compute_suffstats()
    models = {"full": full, "reduced": reduced}
    block_map = {
        name: fitting.assign_jackknife_blocks(len(model.genes), 4)
        for name, model in models.items()
    }
    explicit = fit_random_effects(
        prepare_jk(models, [], 4), exclude_genes=[], n_jk_blocks=4
    )
    for model in models.values():
        model.jk_suffstats = None
    automatic = fit_random_effects(
        models, exclude_genes=[], n_jk_blocks=4, gene_blocks=block_map
    )

    expected_full = fitting.assign_jackknife_blocks(len(full.dataset.genes), 4)
    expected_reduced = fitting.assign_jackknife_blocks(len(reduced.dataset.genes), 4)
    np.testing.assert_array_equal(full.jk_suffstats.gene_blocks, expected_full)
    np.testing.assert_array_equal(reduced.jk_suffstats.gene_blocks, expected_reduced)
    for name in models:
        np.testing.assert_allclose(automatic[name].hsq_jk, explicit[name].hsq_jk)

    shared = set(full.dataset.genes) & set(reduced.dataset.genes)
    full_labels = dict(zip(full.dataset.genes, expected_full))
    reduced_labels = dict(zip(reduced.dataset.genes, expected_reduced))
    assert any(full_labels[gene] != reduced_labels[gene] for gene in shared)


def test_excluded_gene_may_be_absent_from_one_model(class_models):
    full = class_models["missense"]
    absent = full.dataset.genes[5]
    other = class_models["lof"]
    frame = other.dataset.variant_df()
    reduced = Model(
        Dataset(
            frame.filter(pl.col("gene") != absent),
            gene_df=other.dataset.gene_df(),
            beta_col="BETA",
            se_col="SE",
            sample_size=other.dataset.sample_size,
            ldm=other.dataset.ldm_source,
        ),
        class_options(),
    )
    reduced.compute_suffstats()

    exclusions = {"full": [absent], "reduced": []}
    result = estimate_joint_random_hsq(
        prepare_jk({"full": full, "reduced": reduced}, exclusions, 2),
        n_jk_blocks=2,
    )

    assert result["full"].n_genes == full.dataset.n_genes - 1
    assert result["reduced"].n_genes == reduced.dataset.n_genes


def test_joint_gene_coefficient_start_is_model_local(class_models):
    result = estimate_joint_random_hsq(
        prepare_jk(class_models, [], 1),
        init_gene_coef={
            name: dict(
                zip(
                    GENE_ANNOT_COLS,
                    np.roll(np.linspace(0.5, 1.5, 5), index),
                    strict=True,
                )
            )
            for index, name in enumerate(class_models)
        },
        n_jk_blocks=1,
    )
    assert all(np.isfinite(fit.hsq) for fit in result.values())
    assert all(np.isfinite(fitted.gene_coef).all() for fitted in result.values())
