"""Low-rank fixed-effect covariance contractions against dense definitions."""

import numpy as np
import pytest
from scipy import sparse

from kasm.fixed_effects import (
    _allocation_covariance,
    _fixed_quadform_variance,
    _quadratic_form_covariance,
)


def _dense_allocation(y, residual, operator, masks, sample_size, rank):
    symmetrized = np.stack(
        [(mask[:, None] * operator + operator * mask[None, :]) / 2 for mask in masks]
    )
    transformed = symmetrized @ residual
    trace = np.einsum("gij,hji->gh", transformed, transformed)
    projected = symmetrized @ y
    signal = projected @ residual @ projected.T
    hsq = (sample_size * y @ operator @ y - rank) / (sample_size - rank)
    covariance = 2 * trace / sample_size**2
    if signal.sum() > 0 and hsq > 0:
        covariance += 4 * hsq / sample_size * signal / signal.sum()
    return (covariance + covariance.T) / 2


def _dense_quadratic(y, residual, left, right, sample_size, left_rank, right_rank):
    if left_rank >= sample_size or right_rank >= sample_size:
        return 0.0
    left_residual, right_residual = left @ residual, right @ residual
    trace = np.trace(left_residual @ right_residual)
    left_y, right_y = left @ y, right @ y
    left_signal, right_signal = left_y @ residual @ left_y, right_y @ residual @ right_y
    left_variance = _fixed_quadform_variance(y @ left_y, left_rank, sample_size)
    right_variance = _fixed_quadform_variance(y @ right_y, right_rank, sample_size)
    if left_signal > 0 and right_signal > 0:
        correlation = left_y @ residual @ right_y / np.sqrt(left_signal * right_signal)
    elif trace:
        left_trace = np.trace(left_residual @ left_residual)
        right_trace = np.trace(right_residual @ right_residual)
        correlation = trace / np.sqrt(left_trace * right_trace)
    else:
        correlation = 0.0
    return np.sqrt(left_variance * right_variance) * np.clip(correlation, -1, 1)


@pytest.fixture
def low_rank_case():
    rng = np.random.default_rng(281)
    design = rng.normal(size=(9, 3))
    inverse = np.linalg.inv(design.T @ design + np.eye(3))
    base = rng.normal(size=(9, 9))
    residual = base @ base.T / 9 + np.eye(9)
    return rng.normal(size=9), residual, (design, inverse)


@pytest.mark.parametrize("signal_case", ["zero_signal", "zero_hsq", "positive"])
def test_low_rank_allocation_matches_dense_partition_oracle(low_rank_case, signal_case):
    y, residual, factor = low_rank_case
    operator = factor[0] @ factor[1] @ factor[0].T
    if signal_case == "zero_signal":
        y[:] = 0.0  # Also exercises the negative-hsq signal fallback.
    elif signal_case == "zero_hsq":
        y *= np.sqrt(3 / (40 * y @ operator @ y))
    else:
        y *= 2.0
    masks = [np.arange(9) % 3 == group for group in range(3)]
    expected = _dense_allocation(y, residual, operator, masks, 40, 3)
    actual = _allocation_covariance(
        y, sparse.csr_matrix(residual), factor, masks, 40, 3
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-14)


@pytest.mark.parametrize("signal_scale", [0.0, 0.02, 1.0])
def test_low_rank_quadratic_covariance_matches_dense_oracle(
    low_rank_case, signal_scale
):
    y, residual, left = low_rank_case
    y *= signal_scale
    right = (left[0][:, :2], left[1][:2, :2])
    left_dense = left[0] @ left[1] @ left[0].T
    right_dense = right[0] @ right[1] @ right[0].T
    expected = _dense_quadratic(y, residual, left_dense, right_dense, 30, 3, 2)
    actual = _quadratic_form_covariance(
        y, sparse.csr_matrix(residual), left, right, 30, 3, 2
    )
    assert actual == pytest.approx(expected, rel=2e-12, abs=2e-14)


def test_quadratic_covariance_is_zero_when_rank_reaches_sample_size(low_rank_case):
    y, residual, operator = low_rank_case
    assert _quadratic_form_covariance(y, residual, operator, operator, 3, 3, 2) == 0


def test_burden_reporting_does_not_materialize_dense_ld(synthetic_dataset, monkeypatch):
    import kasm

    model = kasm.Model(
        synthetic_dataset,
        kasm.KernelOptions(burden_cols=["constant"], burden_by_maf=False),
    )

    def no_dense_ld(*args, **kwargs):
        pytest.fail("burden reporting must use sparse LD contractions")

    monkeypatch.setattr(kasm.LDM, "toarray", no_dense_ld)
    fit = kasm.fit_fixed_effects(model, include_genes=model.genes)
    assert fit.decompose(["exon"])["hsq"].sum() == pytest.approx(fit.hsq)
    enrichment = kasm.fit_fixed_effects(
        model, include_genes=model.genes, extra_enrich_col="buried"
    )[1]
    assert np.isfinite(enrichment.enrichment_se())
