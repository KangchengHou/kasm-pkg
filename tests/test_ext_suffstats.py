"""Covar independent adjustment checked against dense, nonidentity-LD moment systems."""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from scipy import sparse

import kasm
from kasm.ldm import LDM
from kasm.suffstats import ExtraEnrichCols, CovarPayload, _select_jk_extension
from test_covar import _base_dense_kernels


def covar_kernels(frame):
    kernel = np.zeros((len(frame), len(frame)))
    kernel[0, 1] = kernel[1, 0] = 1
    kernel[1, 2] = kernel[2, 1] = 1
    return {
        "edges": sparse.csr_matrix(kernel),
        "empty": sparse.csr_matrix(kernel.shape),
    }


def complete_covar_kernel(frame):
    return {
        "all": sparse.csr_matrix(np.ones((len(frame), len(frame))) - np.eye(len(frame)))
    }


@pytest.fixture
def covar_indep_model(monkeypatch):
    rng = np.random.default_rng(41)
    n_genes, size = 12, 6
    frame = pl.DataFrame(
        {
            "variant": [f"1:{1000 + i}:A:G" for i in range(n_genes * size)],
            "gene": np.repeat([f"G{i}" for i in range(n_genes)], size),
            "beta": rng.normal(0, 0.02, n_genes * size),
            "se": rng.uniform(0.02, 0.07, n_genes * size),
            "constant": np.ones(n_genes * size),
            "score": rng.uniform(0.2, 1.5, n_genes * size),
            "annot": np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], n_genes),
            "noise": rng.uniform(0.3, 1.8, n_genes * size),
        }
    )
    dataset = kasm.Dataset(frame, beta_col="beta", se_col="se", sample_size=10000)
    correlations = {}
    for gene in dataset.genes:
        factor = rng.normal(size=(size, size))
        covariance = factor @ factor.T + 4 * np.eye(size)
        scale = np.sqrt(np.diag(covariance))
        correlations[gene] = covariance / np.outer(scale, scale)
    monkeypatch.setattr(
        dataset,
        "load_ldm",
        lambda gene: LDM(sparse.csr_matrix(correlations[gene]), np.zeros(size)),
    )
    model = kasm.Model(
        dataset,
        kasm.KernelOptions(
            kernel_groups={
                "all": {
                    "burden_cols": ["score"],
                    "independent_col": "constant",
                    "extra_covar_kernel_fun": covar_kernels,
                    "extra_covar_with_indep": True,
                    "overdispersion_cols": ["noise"],
                }
            },
            overdispersion_shared=True,
        ),
    )
    # Grouped shared overdispersion uses support, retaining separate gene scales.
    return model.compute_suffstats(extra_enrich_cols=["annot"])


def fit_options(blocks=3):
    return dict(
        n_jk_blocks=blocks,
        init_gene_coef={"all": {"constant": 1.7}},
        fix_gene_coef=True,
    )


def test_covar_indep_rectangular_moments_and_matched_fits(
    covar_indep_model, monkeypatch, tmp_path
):
    model = covar_indep_model
    extension = model.suffstats.covar_extensions["all:edges"]
    matrices, traces, targets, projection = [], [], [], []
    for g, gene in enumerate(model.genes):
        covar_kernel = covar_kernels(model.dataset.variant_df(gene))["edges"].toarray()
        diagonal = np.diag((covar_kernel.sum(axis=1) > 0).astype(float))
        kernels = np.array([*_base_dense_kernels(model, gene), diagonal, covar_kernel])
        ld = model.dataset.load_ldm(gene).scale(model.variant_scale(gene)).toarray()
        rows = model.dataset.gene_slice(gene)
        beta = model.dataset.beta[rows] / (
            model.dataset.se[rows] ** 2 * model.dataset.sample_size
        )
        pair = np.einsum("kij,lji->kl", kernels @ ld, kernels @ ld)
        trace = np.einsum("kij,ji->k", kernels, ld)
        quad = np.einsum("i,kij,j->k", beta, kernels, beta)
        np.testing.assert_allclose(extension.pair_trace[g], pair[-2:], atol=1e-18)
        np.testing.assert_allclose(extension.trace[g], trace[-2:], atol=1e-18)
        np.testing.assert_allclose(extension.quadform[g], quad[-2:], atol=1e-18)
        assert (
            abs(pair[-2, -1]) > 1e-12
        )  # A zero-diagonal K still has D-K moments under LD.
        degree, ordered = covar_kernel.sum(axis=1), covar_kernel.sum()
        weights = np.array(
            [
                np.einsum("ij,kij->k", covar_kernel, kernels) / ordered,
                np.einsum("i,kii->k", degree, kernels) / ordered,
            ]
        )
        # Overdispersion does not contribute to heritable variance.
        weights[:, model.layout.overdispersion_indices] = 0
        np.testing.assert_allclose(extension.payload.numerator_weights[g], weights[0])
        np.testing.assert_allclose(extension.payload.denominator_weights[g], weights[1])
        projection.append(weights)
        scales = np.where(
            np.array([kid.is_heritable for kid in extension.layout.kernel_ids]),
            1.7,
            1.0,
        )
        matrices.append(pair * np.outer(scales, scales))
        traces.append(trace * scales)
        targets.append((quad - trace / model.dataset.sample_size) * scales)
    blocks = np.arange(len(model.genes)) % 3
    excluded = [model.genes[2]]
    jk = model.compute_jk_suffstats(
        exclude_genes={"all": excluded}, gene_blocks=blocks, n_jk_blocks=3
    )
    jk.save(tmp_path / "jk.npz")
    model.suffstats.save(tmp_path / "gene.npz")
    restored_gene = kasm.SuffStats.load(tmp_path / "gene.npz")
    assert isinstance(restored_gene.burden_extensions["annot"].payload, ExtraEnrichCols)
    assert isinstance(restored_gene.covar_extensions["all:edges"].payload, CovarPayload)
    restored = kasm.JKSuffStats.load(tmp_path / "jk.npz")
    monkeypatch.setattr(
        model.dataset, "load_ldm", lambda *_: pytest.fail("loaded JK fit accessed LD")
    )
    baseline, result = kasm.fit_random_effects(
        restored, extra_covar_kernel="all:edges", **fit_options()
    )
    full = kasm.fit_random_effects(
        _select_jk_extension(restored, restored.covar_extensions["all:edges"]),
        **fit_options(),
    )
    keep = np.array([gene not in excluded for gene in model.genes])
    moments = []
    for block in [None, 0, 1, 2]:
        selected = keep if block is None else keep & (blocks != block)
        pair, target = (
            np.asarray(matrices)[selected].sum(axis=0),
            np.asarray(targets)[selected].sum(axis=0),
        )
        full_coef = np.linalg.solve(pair, target)
        actual_full = full.kernel_coef if block is None else full.kernel_coef_jk[block]
        np.testing.assert_allclose(actual_full, full_coef, rtol=1e-10, atol=1e-12)
        weights = np.asarray(projection)[selected].sum(axis=0) * scales
        moments.append(
            [
                weights[0] @ full_coef,
                weights[1] @ full_coef,
            ]
        )
    np.testing.assert_allclose(result.point, moments[0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.jk, moments[1:], rtol=1e-10, atol=1e-12)
    assert len(baseline.kernel_coef) == extension.layout.n_kernels
    assert len(restored.burden_extensions) == 1 and len(restored.covar_extensions) == 2
    assert restored.burden_extensions["annot"].payload.burden_cov is None


@pytest.mark.parametrize("covariance", [0.0, 0.15])
def test_covar_indep_population_null_and_positive_control(
    covar_indep_model, covariance
):
    model = covar_indep_model
    base = model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=3)
    extension = base.covar_extensions["all:edges"]
    selected = _select_jk_extension(base, extension)
    theta = np.zeros(selected.layout.n_kernels)
    theta[0], theta[1], theta[-2], theta[-1] = 0.0, 0.3, 0.8, covariance
    scales = np.where(
        np.array([kid.is_heritable for kid in selected.layout.kernel_ids]), 1.7, 1.0
    )
    qf = (
        np.einsum("bij,j->bi", selected.blocks["pt0"], theta * scales)
        + selected.blocks["tr0"] / base.sample_size
    )
    changed_base = replace(
        base,
        blocks={
            **base.blocks,
            "qf0": qf[:, : model.n_kernels],
            "qf1": qf[:, : model.n_kernels, None],
        },
    )
    changed_ext = replace(extension, quadform=np.repeat(qf[:, -2:, None], 2, axis=-1))
    population = replace(changed_base, covar_extensions={"all:edges": changed_ext})
    full = kasm.fit_random_effects(
        _select_jk_extension(population, changed_ext), **fit_options()
    )
    np.testing.assert_allclose(full.kernel_coef, theta, atol=1e-10)
    np.testing.assert_allclose(full.kernel_coef_jk, np.tile(theta, (3, 1)), atol=1e-10)
    _, result = kasm.fit_random_effects(
        population, extra_covar_kernel="all:edges", **fit_options()
    )
    assert np.isfinite(result.point).all()
    if covariance == 0:
        np.testing.assert_allclose(result.point[0], 0, atol=1e-10)
    else:
        assert result.point[0] == pytest.approx(len(model.genes) * 1.7 * covariance)
        assert result.point[0] / result.point[1] > 0


def test_covar_indep_option_requires_callback():
    with pytest.raises(ValueError, match="requires extra_covar_kernel_fun"):
        kasm.KernelOptions(
            kernel_groups={
                "all": {
                    "burden_cols": ["x"],
                    "independent_col": "x",
                    "extra_covar_with_indep": True,
                }
            }
        )


def test_covar_indep_one_block(covar_indep_model):
    model = covar_indep_model
    stats = model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=1)
    _, result = kasm.fit_random_effects(
        stats, extra_covar_kernel="all:edges", **fit_options(1)
    )
    np.testing.assert_array_equal(result.point, result.jk[0])


def test_covar_indep_redundancy_and_delete_rank_failure(covar_indep_model):
    model = covar_indep_model
    groups = {
        name: {**options, "extra_covar_kernel_fun": complete_covar_kernel}
        for name, options in model.options.kernel_groups.items()
    }
    redundant = kasm.Model(
        model.dataset, replace(model.options, burden_cols=None, kernel_groups=groups)
    ).compute_suffstats()
    redundant.compute_jk_suffstats(
        exclude_genes={"all": []}, n_jk_blocks=fit_options()["n_jk_blocks"]
    )
    with pytest.raises(
        (ValueError, np.linalg.LinAlgError), match="redundant|singular|rank"
    ):
        kasm.fit_random_effects(
            redundant,
            exclude_genes=[],
            extra_covar_kernel="all:all",
            **fit_options(),
        )
    # Delete 0 leaves only genes whose covar independent kernel equals class indep.
    # Remaining genes use the partial covar kernel, so the point system is full rank.
    original, complete = model.suffstats, redundant.suffstats
    partial = original.covar_extensions["all:edges"]
    full = complete.covar_extensions["all:all"]
    n = len(model.genes) // 2
    extension = replace(
        partial,
        trace=np.concatenate((partial.trace[:n], full.trace[n:])),
        quadform=np.concatenate((partial.quadform[:n], full.quadform[n:])),
        pair_trace=np.concatenate((partial.pair_trace[:n], full.pair_trace[n:])),
        payload=replace(
            partial.payload,
            **{
                key: np.concatenate(
                    (getattr(partial.payload, key)[:n], getattr(full.payload, key)[n:])
                )
                for key in (
                    "numerator_weights",
                    "denominator_weights",
                    "n_genes",
                    "n_pairs",
                )
            },
        ),
    )
    model.suffstats = replace(original, covar_extensions={"all:edges": extension})
    stats = model.compute_jk_suffstats(
        exclude_genes=[], n_jk_blocks=2, gene_blocks=np.repeat([0, 1], n)
    )
    with pytest.raises(np.linalg.LinAlgError, match="singular|rank"):
        kasm.fit_random_effects(stats, extra_covar_kernel="all:edges", **fit_options(2))


def test_parent_validation_and_selection(covar_indep_model, monkeypatch):
    model = covar_indep_model
    calls = []
    original = kasm.fitting.annotation_block_sums

    def aggregate(*args, **kwargs):
        calls.append(np.shape(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(kasm.fitting, "annotation_block_sums", aggregate)
    parent = model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=3)
    assert len(calls) == 4  # baseline, one burden, two covar kernels
    saved = {key: value.copy() for key, value in parent.blocks.items()}
    selected = _select_jk_extension(parent, parent.covar_extensions["all:edges"])
    assert not selected.burden_extensions and not selected.covar_extensions
    selected.blocks["pt0"][0, 0, 0] += 1
    for key, value in saved.items():
        np.testing.assert_array_equal(parent.blocks[key], value)
    kasm.fit_random_effects(parent, extra_covar_kernel="all:edges", **fit_options())
    kasm.fit_random_effects(parent, extra_covar_kernel="all:empty", **fit_options())
    assert len(calls) == 4
    for kwargs in (
        {"extra_covar_kernel": "all:missing"},
        {"extra_enrich_col": "missing"},
    ):
        with pytest.raises((KeyError, ValueError), match="unknown|grouping"):
            kasm.fit_random_effects(parent, **kwargs, **fit_options())
    with pytest.raises(ValueError, match="n_jk_blocks"):
        kasm.fit_random_effects(
            parent, extra_covar_kernel="all:edges", **fit_options(2)
        )
    with pytest.raises(ValueError, match="exclusion"):
        kasm.fit_random_effects(
            parent,
            exclude_genes=[model.genes[0]],
            extra_covar_kernel="all:edges",
            **fit_options(),
        )
    ext = parent.covar_extensions["all:edges"]
    for changed in (
        replace(ext, pair_trace=ext.pair_trace[..., :1]),
        replace(ext, base_to_extended=np.zeros(model.n_kernels, dtype=int)),
        replace(
            ext,
            payload=replace(
                ext.payload, numerator_weights=ext.payload.numerator_weights[:, :-1]
            ),
        ),
    ):
        with pytest.raises(ValueError, match="extension|mapping"):
            replace(parent, covar_extensions={"all:edges": changed})
    with pytest.raises(ValueError, match="gene_blocks"):
        replace(parent, gene_blocks=np.zeros(1, dtype=int))


def test_near_singular_diagnostic_preserves_unconstrained_solution(caplog):
    system = np.array([[1.0, 1.0 - 1e-11], [1.0 - 1e-11, 1.0]])
    truth = np.array([1.0, -1.0])
    actual, rank = kasm.fitting._gram_solve(system, system @ truth)
    assert rank == 2
    np.testing.assert_allclose(actual, truth, atol=1e-8)
    assert "near_singular_moment_system" in caplog.text


def test_compact_enrichment_continuous_fit(covar_indep_model):
    model = kasm.Model(
        covar_indep_model.dataset,
        kasm.KernelOptions(burden_cols=["constant"], independent_col="constant"),
    ).compute_suffstats(extra_enrich_cols=["score"])
    parent = model.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=3)
    options = dict(n_jk_blocks=3, init_gene_coef={"constant": 1.7}, fix_gene_coef=True)
    compact, info = kasm.fit_random_effects(parent, extra_enrich_col="score", **options)
    assert info.annotation == "score"
    extended = model.with_burden_extension("score")
    extended.compute_jk_suffstats(exclude_genes=[], n_jk_blocks=3)
    ordinary = kasm.fit_random_effects(extended, exclude_genes=[], **options)
    np.testing.assert_allclose(
        compact.kernel_coef_jk, ordinary.kernel_coef_jk, rtol=1e-12
    )
    baseline = kasm.fit_random_effects({"cell": parent}, n_jk_blocks=3)["cell"]
    assert len(baseline.kernel_coef) == model.n_kernels
