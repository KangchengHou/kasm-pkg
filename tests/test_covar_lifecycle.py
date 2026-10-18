"""Independent checks of incremental preparation and the common random lifecycle."""

from extension_helpers import extension_designs

import numpy as np
import pytest
from test_covar import (
    _base_dense_kernels,
    _covar_model,
    _grouped_covar_kernel_model,
    distance_covar_kernels,
)
from test_suffstats import dense_oracle

import kasm
from kasm import fitting, suffstats


def test_one_gene_call_prepares_burden_and_covar_extensions(
    synthetic_dataset_no_ld, monkeypatch, tmp_path
):
    model = _covar_model(synthetic_dataset_no_ld)
    reduced, loaded = [], []
    original = suffstats.compute_gene_suffstats
    load = model.dataset.load_ldm

    def base(**kwargs):
        reduced.append(kwargs["index"].n_kernels)
        return original(**kwargs)

    def ld(gene):
        loaded.append(gene)
        return load(gene)

    monkeypatch.setattr(suffstats, "compute_gene_suffstats", base)
    monkeypatch.setattr(model.dataset, "load_ldm", ld)
    model.compute_suffstats(extra_enrich_cols=["buried"])
    assert reduced == [model.n_kernels] * len(model.genes)
    assert loaded == list(model.genes)
    stats = model.suffstats
    assert tuple(stats.burden_extensions) == ("buried",)
    assert tuple(stats.covar_extensions) == ("all:near", "all:far")
    selected = stats.with_burden_extension("buried")
    extended = model._with_burden_column("buried")
    for gene in model.genes:
        reference = dense_oracle(extended, gene)
        np.testing.assert_allclose(
            selected.pair_trace(gene), reference["pair_trace"], rtol=1e-12, atol=1e-16
        )
    stats.save(tmp_path / "gene.npz")
    model.suffstats = kasm.SuffStats.load(tmp_path / "gene.npz")
    monkeypatch.setattr(
        model.dataset, "load_ldm", lambda *args: pytest.fail("aggregation loaded LD")
    )
    jk = model.compute_jk_suffstats(
        exclude_genes={"all": [model.genes[0]]}, n_jk_blocks=2
    )
    jk.save(tmp_path / "blocks.npz")
    restored = kasm.JKSuffStats.load(tmp_path / "blocks.npz")
    assert restored.excluded_genes == (model.genes[0],)
    assert restored.burden_extensions
    assert restored.covar_extensions
    for _, name, selected, payload in extension_designs(
        jk, extra_enrich_cols=["buried"], extra_covar_kernels=["all:near", "all:far"]
    ):
        selected.save(tmp_path / f"{name}.npz")
        recovered = kasm.JKSuffStats.load(tmp_path / f"{name}.npz")
        assert recovered.layout == selected.layout
        for key, value in selected.blocks.items():
            np.testing.assert_array_equal(recovered.blocks[key], value)
        kind = "burden" if name == "buried" else "covar"
        saved_payload = getattr(restored, kind + "_extensions")[name].payload
        for key, value in payload.items():
            if key not in ("annotation", "total_het"):
                np.testing.assert_array_equal(getattr(saved_payload, key), value)


@pytest.mark.parametrize("n_blocks", [1, 2])
def test_every_augmented_coefficient_matches_dense_point_and_jk(
    synthetic_dataset_no_ld, monkeypatch, tmp_path, n_blocks
):
    model = _grouped_covar_kernel_model(synthetic_dataset_no_ld)
    exclude = {"missense": [model.genes[0]]}
    initial = {"missense": {"constraint_low": 0.5, "constraint_high": 1.5}}
    names = ["missense:near", "missense:far"]
    solves, aggregations = [], []
    original_solve = fitting.alternating_fit
    original_aggregate = model.compute_jk_suffstats

    def solve(*args, **kwargs):
        result = original_solve(*args, **kwargs)
        solves.append(result[0]["theta"].copy())
        return result

    def aggregate(**kwargs):
        aggregations.append(kwargs)
        return original_aggregate(**kwargs)

    monkeypatch.setattr(fitting, "alternating_fit", solve)
    monkeypatch.setattr(model, "compute_jk_suffstats", aggregate)
    kwargs = dict(
        exclude_genes=exclude,
        init_gene_coef=initial,
        fix_gene_coef=True,
        extra_covar_kernel=names[0],
        n_jk_blocks=n_blocks,
    )
    model.compute_jk_suffstats(exclude_genes=exclude, n_jk_blocks=n_blocks)
    baseline, covar = kasm.fit_random_effects(model, **kwargs)
    assert len(aggregations) == 1
    expected_calls = 1 if n_blocks == 1 else n_blocks + 1
    assert len(solves) == expected_calls
    scales = model.gene_annot_matrix() @ np.array([0.5, 1.5])
    references = []
    for extension in ["near"]:
        systems, targets = [], []
        for g, gene in enumerate(model.genes):
            kernels = _base_dense_kernels(model, gene)
            kernels = [
                kernel * (scales[g] if k in model.layout.heritable_indices else 1)
                for k, kernel in enumerate(kernels)
            ]
            if extension is not None:
                kernels.append(
                    distance_covar_kernels(model.dataset.variant_df(gene))[
                        extension
                    ].toarray()
                    * scales[g]
                )
            rows = model.dataset.gene_slice(gene)
            ld = model.dataset.load_ldm(gene).scale(model.variant_scale(gene)).toarray()
            beta = model.dataset.beta[rows] / (
                model.dataset.se[rows] ** 2 * model.dataset.sample_size
            )
            products = np.asarray([kernel @ ld for kernel in kernels])
            systems.append(np.einsum("kij,lji->kl", products, products))
            targets.append(
                [
                    beta @ kernel @ beta
                    - np.trace(kernel @ ld) / model.dataset.sample_size
                    for kernel in kernels
                ]
            )
        for block in [None] if n_blocks == 1 else [None, *range(n_blocks)]:
            keep = np.array([gene not in exclude["missense"] for gene in model.genes])
            if block is not None:
                keep &= baseline.gene_blocks != block
            left = np.asarray(systems)[keep].sum(axis=0)
            right = np.asarray(targets)[keep].sum(axis=0)
            live = np.diag(left) > 0
            coefficient = np.zeros(len(right))
            coefficient[live] = np.linalg.solve(left[np.ix_(live, live)], right[live])
            references.append(coefficient)
    for actual, expected in zip(solves, references, strict=True):
        np.testing.assert_allclose(actual, expected)
    assert any(np.any(value < 0) for value in solves)
    assert baseline.n_iter == 0
    assert np.all(baseline.n_iter_jk == 0)
    # A matching, explicitly aligned cache is sufficient for the same fits and scoring.
    blocks = (n_blocks - 1) - baseline.gene_blocks
    jk = original_aggregate(
        exclude_genes=exclude, gene_blocks=blocks, n_jk_blocks=n_blocks
    )
    jk.save(tmp_path / "jk.npz")
    designs = {
        name: (selected, payload)
        for _, name, selected, payload in extension_designs(
            jk, extra_covar_kernels=names
        )
    }
    selected = designs[names[0]][0]
    selected.save(tmp_path / "selected.npz")
    restored = kasm.JKSuffStats.load(tmp_path / "selected.npz")
    assert restored.layout == selected.layout
    assert restored.layout.kernel_ids[-1].kind == "C"
    from kasm.random_effects import _fit_random_effects

    fit_args = dict(
        init_gene_coef=initial,
        fix_gene_coef=True,
        n_jk_blocks=n_blocks,
    )
    selected_fit = _fit_random_effects(selected, **fit_args)["model"]
    restored_fit = _fit_random_effects(restored, **fit_args)["model"]
    np.testing.assert_array_equal(
        selected_fit.kernel_coef_jk, restored_fit.kernel_coef_jk
    )
    restored_base = kasm.JKSuffStats.load(tmp_path / "jk.npz")
    monkeypatch.setattr(
        model.dataset, "load_ldm", lambda *args: pytest.fail("cached fitting loaded LD")
    )
    cached_fit = kasm.fit_random_effects(restored_base, **fit_args)
    implicit_support = kasm.fit_random_effects(
        restored_base,
        **{key: value for key, value in fit_args.items() if key != "exclude_genes"},
    )
    np.testing.assert_array_equal(
        implicit_support.kernel_coef_jk, cached_fit.kernel_coef_jk
    )
    cached_fit, cached_covar = kasm.fit_random_effects(
        restored_base, extra_covar_kernel=names[0], **fit_args
    )
    np.testing.assert_array_equal(cached_fit.kernel_coef, baseline.kernel_coef)
    np.testing.assert_array_equal(
        cached_fit.kernel_coef_jk, baseline.kernel_coef_jk[::-1]
    )
    np.testing.assert_array_equal(cached_covar.point, covar.point)
    np.testing.assert_array_equal(cached_covar.jk, covar.jk[::-1])
    assert cached_fit.score_variants(model.dataset).equals(
        baseline.score_variants(model.dataset)
    )
    with pytest.raises(ValueError, match="exclusions do not match"):
        kasm.fit_random_effects(
            restored_base, **{**fit_args, "exclude_genes": {"missense": []}}
        )
