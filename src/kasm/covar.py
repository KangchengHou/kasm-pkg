"""Covar extension statistics and effect-covar estimators.

Each named covar kernel is fitted as ``base + one covar kernel``. The
callable's sparse matrices therefore live only for one gene and are reduced
immediately; no covar kernel-by-covar kernel trace tensor is retained.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps

from kasm import fitting
from kasm.suffstats import _diagonal_moments


@dataclass(frozen=True)
class CovarResult:
    """Additive covariance moments for one caller-selected pair set.

    Self/kernel readouts contain (covariance, variance); cross-group readouts
    contain only covariance. ``point`` has shape (n_moments,) and ``jk``
    has shape (n_blocks, n_moments), with n_moments equal to 1 or 2.
    The caller owns covar kernel identity.
    """

    point: np.ndarray
    jk: np.ndarray
    n_genes: int
    n_pairs: int

    def __post_init__(self):
        if (
            self.point.shape not in ((1,), (2,))
            or self.jk.ndim != 2
            or self.jk.shape[1:] != self.point.shape
        ):
            raise ValueError("point/jk must contain one or two aligned covar moments")
        if not np.isfinite(self.point).all() or not np.isfinite(self.jk).all():
            raise ValueError("point/jk moments must be finite")

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            point=self.point,
            jk=self.jk,
            n_genes=self.n_genes,
            n_pairs=self.n_pairs,
        )

    @classmethod
    def load(cls, path) -> CovarResult:
        with np.load(path, allow_pickle=False) as stored:
            return cls(
                point=stored["point"],
                jk=stored["jk"],
                n_genes=int(stored["n_genes"]),
                n_pairs=int(stored["n_pairs"]),
            )


def _produce_covar_kernels(model, frame):
    names, kernel_groups, kernels = [], [], []
    for kernel_group, function in model.options.extra_covar_kernel_functions():
        produced = function(frame.clone())
        if not isinstance(produced, Mapping) or not produced:
            raise ValueError(
                "extra_covar_kernel_fun must return a non-empty mapping of names "
                "to sparse matrices"
            )
        for local_name, kernel in produced.items():
            if not isinstance(local_name, str) or not local_name or ":" in local_name:
                raise ValueError(
                    "covar kernel names must be non-empty and cannot contain ':'"
                )
            names.append(f"{kernel_group}:{local_name}")
            kernel_groups.append(kernel_group)
            kernels.append(kernel)
    if not names:
        raise ValueError("no extra_covar_kernel_fun is configured")
    if len(set(names)) != len(names):
        raise ValueError("covar kernel names must be unique")
    return tuple(names), tuple(kernel_groups), kernels


def _validate_covar_kernel(
    matrix, size: int, name: str, gene: str, support=None
) -> sps.csr_matrix:
    kernel = sps.csr_matrix(matrix, dtype=float, copy=True)
    if kernel.shape != (size, size):
        raise ValueError(
            f"covar kernel {name!r} for {gene!r} has shape "
            f"{kernel.shape}, expected {(size, size)}"
        )
    kernel.sum_duplicates()
    kernel.eliminate_zeros()
    if not np.isfinite(kernel.data).all():
        raise ValueError(f"covar kernel {name!r} for {gene!r} must be finite")
    if (kernel.data < 0).any():
        raise ValueError(f"covar kernel {name!r} for {gene!r} must be nonnegative")
    if np.any(kernel.data != 1):
        raise ValueError(f"covar kernel {name!r} for {gene!r} must be binary")
    if np.any(kernel.diagonal() != 0):
        raise ValueError(f"covar kernel {name!r} for {gene!r} must have zero diagonal")
    difference = kernel - kernel.T
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-12:
        raise ValueError(f"covar kernel {name!r} for {gene!r} must be symmetric")
    if support is not None and kernel[np.flatnonzero(~support)].nnz:
        raise ValueError(
            f"covar kernel {name!r} for {gene!r} has entries outside its "
            "kernel group support"
        )
    return kernel


def _covar_extension_inputs(model, gene):
    """Generate each sparse covar extension on its owning group's support."""
    if model.options.within_gene_group_col is not None:
        raise ValueError(
            "extra_covar_kernel_fun cannot be combined with within_gene_group_col"
        )
    frame = model.dataset.variant_df(gene)
    names, groups, kernels = _produce_covar_kernels(model, frame)
    for name, group, kernel in zip(names, groups, kernels, strict=True):
        support = (
            frame[model.options.kernel_groups[group]["independent_col"]]
            .to_numpy()
            .astype(bool)
        )
        yield (
            name,
            {
                "kernel_group": group,
                "kernel": _validate_covar_kernel(
                    kernel, len(frame), name, gene, support
                ),
                "independent_indices": model.layout.independent_indices,
                "with_indep": model.options.kernel_groups[group][
                    "extra_covar_with_indep"
                ],
            },
        )


def _compute_covar_ext(
    *,
    kernel,
    independent_indices,
    ldm,
    beta_tilde,
    burden_wgts,
    diag_weights,
    index,
    propagated_burden_wgts,
    with_indep=False,
):
    """One sparse covar extension's rectangular moments and projection weights."""
    n_base = index.n_kernels
    n_added = 2 if with_indep else 1
    covar_ext = {
        "pair_trace": np.zeros((n_added, n_base + n_added)),
        "trace": np.zeros(n_added),
        "quadform": np.zeros(n_added),
        "numerator_weights": np.zeros(n_base + n_added),
        "denominator_weights": np.zeros(n_base + n_added),
        "pair_counts": round(float(kernel.sum()) / 2),
        "eligible": bool(kernel.nnz),
    }
    ordered = float(kernel.sum())
    if ordered <= 0:
        return covar_ext
    kernel_stats = ldm.kernel_trace_stats(kernel)
    covar_ext["trace"][-1] = kernel_stats.trace
    covar_ext["quadform"][-1] = float(beta_tilde @ (kernel @ beta_tilde))
    covar_ext["pair_trace"][-1, -1] = kernel_stats.self_pair_trace
    projection = propagated_burden_wgts.T @ (kernel @ propagated_burden_wgts)
    for position, base_index in enumerate(index.burden_indices):
        covar_ext["pair_trace"][-1, base_index] = projection[position, position]
    for left, right, base_index in zip(
        index.cross_left, index.cross_right, index.cross_indices
    ):
        covar_ext["pair_trace"][-1, base_index] = 2.0 * projection[left, right]
    if index.diag_indices.size:
        covar_ext["pair_trace"][-1, index.diag_indices] = (
            diag_weights.T @ kernel_stats.diag_pair_weights
        )
    kernel_projection = burden_wgts.T @ (kernel @ burden_wgts) / ordered
    pair_degree = np.asarray(kernel.sum(axis=1)).ravel()
    denominator_projection = (
        burden_wgts.T @ (pair_degree[:, None] * burden_wgts) / ordered
    )
    for key, projection in zip(
        ("numerator_weights", "denominator_weights"),
        (kernel_projection, denominator_projection),
        strict=True,
    ):
        for position, base_index in enumerate(index.burden_indices):
            covar_ext[key][base_index] = projection[position, position]
        for left, right, base_index in zip(
            index.cross_left, index.cross_right, index.cross_indices
        ):
            covar_ext[key][base_index] = 2.0 * projection[left, right]
    diagonal_position = {
        base_index: position for position, base_index in enumerate(index.diag_indices)
    }
    for independent in independent_indices:
        covar_ext["denominator_weights"][independent] = float(
            pair_degree @ diag_weights[:, diagonal_position[independent]] / ordered
        )
    covar_ext["numerator_weights"][-1] = 1.0
    if with_indep:
        covar_membership = (pair_degree > 0).astype(float)
        # Match the independent kernel convention: binary membership in the
        # unscaled kernel; the existing scaled LD supplies the variant scale.
        trace, quadform, pair = _diagonal_moments(
            ldm,
            beta_tilde,
            covar_membership[:, None],
            propagated_burden_wgts[:, :, None],
            index,
        )
        covar_ext["trace"][0] = trace[0]
        covar_ext["quadform"][0] = quadform[0]
        covar_ext["pair_trace"][0, :n_base] = pair[0]
        diagonal = ldm.diag_pair_trace(
            covar_membership[:, None], np.column_stack((diag_weights, covar_membership))
        )[0]
        covar_ext["pair_trace"][0, index.diag_indices] = diagonal[:-1]
        covar_ext["pair_trace"][0, -2] = diagonal[-1]
        covar_ext["pair_trace"][0, -1] = covar_ext["pair_trace"][-1, -2] = (
            covar_membership @ kernel_stats.diag_pair_weights
        )
        covar_ext["denominator_weights"][-2] = pair_degree @ covar_membership / ordered
    return covar_ext


# ---------------------------------------------------------------------------
# Kernel-group covariance moments
# ---------------------------------------------------------------------------


def _group_projection(model, group_col, group_col2):
    """Compute selected additive moments directly, without pair matrices."""
    is_self = group_col == group_col2
    layout = model.layout
    diagonal_position = {
        index: position for position, index in enumerate(layout.diagonal_indices)
    }
    burden_indices = [
        *((index, p, p) for p, index in enumerate(layout.burden_indices)),
        *(
            (index, left, right)
            for (left, right), index in zip(layout.cross_pairs, layout.cross_indices)
        ),
    ]
    rows, pairs = [], []
    for gene in model.genes:
        frame = model.dataset.variant_df(gene)
        masks = {}
        for column in dict.fromkeys((group_col, group_col2)):
            if column not in frame.columns:
                raise ValueError(f"unknown covariance indicator column: {column!r}")
            values = frame[column].to_numpy()
            if frame[column].null_count() or not np.isin(values, [0, 1]).all():
                raise ValueError(
                    f"covariance indicator {column!r} must contain only non-null 0/1 values"
                )
            masks[column] = values.astype(bool)
        ma, mb = masks[group_col], masks[group_col2]
        if not is_self and (ma & mb).any():
            raise ValueError("cross-covariance indicator columns must not overlap")
        na, nb = int(ma.sum()), int(mb.sum())
        count = na * (na - 1) // 2 if is_self else na * nb
        table = np.zeros((layout.n_kernels, 2 if is_self else 1))
        pairs.append(count)
        if count:
            burden = model.burden_wgts(gene)
            diagonal = model.diag_kernel_weights(gene) if is_self else None
            for kernel, left, right in burden_indices:
                u, v = burden[:, left], burden[:, right]
                factor = 2.0 if kernel in layout.cross_indices else 1.0
                if is_self:
                    covariance = (u[ma].sum() * v[ma].sum() - (u[ma] * v[ma]).sum()) / (
                        na * (na - 1)
                    )
                else:
                    covariance = (
                        u[ma].sum() * v[mb].sum() + v[ma].sum() * u[mb].sum()
                    ) / (2 * na * nb)
                table[kernel, 0] = factor * covariance
                if is_self:
                    table[kernel, 1] = factor * (u[ma] * v[ma]).mean()
            if is_self:
                for index in layout.independent_indices:
                    table[index, 1] = diagonal[ma, diagonal_position[index]].mean()
        rows.append(table)
    return np.stack(rows), np.asarray(pairs, dtype=np.int64)


def compute_random_covar(
    model,
    fit,
    *,
    group_col: str | None = None,
    group_col1: str | None = None,
    group_col2: str | None = None,
    include_genes=None,
) -> CovarResult:
    """Project one baseline random fit on selected all-pairs indicator support.

    group_col selects self pairs (>=2 variants); group_col1/group_col2 select cross pairs with >=1
    variant at each endpoint. Self returns covariance and variance; cross returns
    only covariance. The caller owns normalization and matching support across
    calls. include_genes optionally restricts reporting to a
    common support without refitting. Uses the fit's exact point/JK coefficients
    and block labels. Overdispersion is sampling noise and contributes zero.
    """
    if model.options.within_gene_group_col is not None:
        raise ValueError(
            "covariance projection does not support within_gene_group_col "
            "(including exon burden groups)"
        )
    if group_col is not None:
        if group_col1 is not None or group_col2 is not None:
            raise ValueError(
                "use group_col for self or group_col1 and group_col2 for cross, not both"
            )
        group_col2 = group_col
    else:
        if group_col1 is None or group_col2 is None:
            raise ValueError(
                "provide group_col for self or both group_col1 and group_col2 for cross"
            )
        if group_col1 == group_col2:
            raise ValueError("use group_col for a self readout")
        group_col = group_col1
    if tuple(fit.kernel_labels) != tuple(
        kernel.label for kernel in model.layout.kernel_ids
    ):
        raise ValueError(
            "covariance projection requires a baseline fit, without an added extension"
        )
    genes = list(model.genes)
    if fit.gene_blocks is None:
        raise ValueError("projection requires fit gene blocks")
    block_of, _, _ = fitting.jackknife_blocks(genes, fit.gene_blocks, fit.n_jk_blocks)
    from kasm.suffstats import _get_gene_group_mask

    kernel_groups = np.asarray(fit.kernel_groups, dtype=np.intp)
    projection, pair_counts = _group_projection(model, group_col, group_col2)
    gene_group_mask = _get_gene_group_mask(
        genes,
        fit.excluded_genes,
        fit.kernel_group_names or (None,),
        fit.kernel_options.get("covariance_between_kernel_groups"),
    )
    keep = (pair_counts > 0) & gene_group_mask.any(axis=1)
    if include_genes is not None:
        include_genes = set(include_genes)
        if unknown := include_genes - set(genes):
            raise ValueError(f"unknown projection genes: {sorted(unknown)}")
        keep &= np.asarray([gene in include_genes for gene in genes])
    if not keep.any():
        raise ValueError("no eligible genes for the requested covariance readout")
    annot = model.gene_annot_matrix()
    heritable = np.isin(fit.kernel_labels, fit.heritable_labels)

    def contract(gene_coef, kernel_coef, selected):
        eta = (annot @ np.asarray(gene_coef, dtype=float)) * gene_group_mask
        scales = np.zeros((len(genes), model.n_kernels))
        scales[:, heritable] = eta[:, kernel_groups[heritable]]
        return np.einsum(
            "gk,gks,k->s",
            scales[selected],
            projection[selected],
            np.asarray(kernel_coef, dtype=float),
        )

    point = contract(fit.gene_coef, fit.kernel_coef, keep)
    jk = np.stack(
        [
            contract(
                fit.gene_coef_jk[block],
                fit.kernel_coef_jk[block],
                keep & ((fit.n_jk_blocks == 1) | (block_of != block)),
            )
            for block in range(fit.n_jk_blocks)
        ]
    )
    return CovarResult(point, jk, int(keep.sum()), int(pair_counts[keep].sum()))


def _project_covar_fit(stats, fit, extension):
    """Project additive moments from one full augmented fit."""
    n_blocks = stats.n_jk_blocks

    positions = [extension.kernel_labels.index(label) for label in fit.kernel_labels]
    scales = np.ones((len(stats.gene_annot_names), len(fit.kernel_groups)))
    heritable = np.isin(fit.kernel_labels, fit.heritable_labels)
    scales[:, heritable] = fit.gene_coef[:, fit.kernel_groups[heritable]]

    def contract(key):
        weights = getattr(extension.payload, key)[:, positions]
        total = weights.sum(axis=0)
        value = np.einsum("ka,ak,k->", total, scales, fit.kernel_coef)
        delete = total[None] if n_blocks == 1 else total - weights
        replicates = np.einsum("bka,ak,bk->b", delete, scales, fit.kernel_coef_jk)
        return value, replicates

    values = [
        contract("numerator_weights"),
        contract("denominator_weights"),
    ]
    n_pairs = float(extension.payload.n_pairs.sum())
    if not n_pairs.is_integer():
        raise ValueError("aggregated covar pair count must be integral")
    return CovarResult(
        point=np.asarray([value[0] for value in values]),
        jk=np.stack([value[1] for value in values], axis=-1),
        n_genes=int(extension.payload.n_genes.sum()),
        n_pairs=int(n_pairs),
    )
