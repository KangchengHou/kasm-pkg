"""Public fitting API over KASM's numerical estimators.

The numerical modules own moment construction and optimization.  This module
owns user-facing dispatch, stable result objects, and random/fixed composition.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import polars as pl

from kasm import covar as _covar
from kasm import fitting
from kasm.dataset import Dataset
from kasm.enrichment import (
    EnrichResult,
    _fixed_enrich_result,
    _random_enrich_result,
    _validate_burden_annotation,
)
from kasm.fixed_effects import (
    FixedFit,
    _fit_fixed_effects,
)
from kasm.model import Model
from kasm.random_effects import (
    RandomFit,
    _assemble_random_reports,
    _fit_random_effects,
    make_kernel_groups,
)
from kasm.suffstats import (
    JKSuffStats,
    _select_covar_extension,
    _select_jk_extension,
)


def _require_suffstats(model) -> None:
    if model.suffstats is None:
        model.compute_suffstats()


def _snapshot_gene_selection(selection):
    """Materialize one-shot iterables once at the public API boundary."""
    if isinstance(selection, Mapping):
        return {label: tuple(genes) for label, genes in selection.items()}
    return tuple(selection)


def _has_group_specific_exclusions(selection):
    """Whether a group mapping describes genuinely different support."""
    return (
        isinstance(selection, Mapping)
        and len({tuple(sorted(set(genes))) for genes in selection.values()}) > 1
    )


def _overdispersion_table(
    layout,
    kernel_coef,
    kernel_coef_se,
    variance,
    variance_se,
    *,
    model_name=None,
):
    rows = []
    for index in layout.overdispersion_indices:
        kernel = layout.kernel_ids[index]
        row = {
            "annotation": kernel.col,
            "maf_bin": kernel.maf_label,
            "kernel_coef": float(kernel_coef[index]),
            "kernel_coef_se": float(kernel_coef_se[index]),
            "variance": float(variance[index]),
            "variance_se": float(variance_se[index]),
        }
        if model_name is not None:
            row["model"] = model_name
        rows.append(row)
    schema = {
        **({"model": pl.String} if model_name is not None else {}),
        "annotation": pl.String,
        "maf_bin": pl.String,
        "kernel_coef": pl.Float64,
        "kernel_coef_se": pl.Float64,
        "variance": pl.Float64,
        "variance_se": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def _maf_trace_data(model, stats):
    """Return reporting traces retained during the shared gene reduction."""
    if model.options.burden_by_maf:
        return model, stats.blocks["tr1"]
    trace_model = Model(
        model.dataset,
        replace(
            model.options,
            burden_by_maf=True,
            burden_cols=None
            if model.options.kernel_groups
            else model.options.burden_cols,
        ),
    )
    if stats.maf_trace is None:
        raise ValueError(
            "MAF reporting traces are missing; recompute sufficient statistics"
        )
    return trace_model, stats.maf_trace


def _random_maf_components(
    model,
    fitted_model,
    exclude_genes,
    expected_hsq,
    *,
    expected_hsq_jk,
    trace_data,
    target_kernel_group=None,
    independent_trace=None,
) -> pl.DataFrame:
    """Allocate point and delete-block random hsq across MAF bins."""
    bins = model.options.maf_bins
    kernel_coef_jk, gene_coef_jk = (
        fitted_model.kernel_coef_jk,
        fitted_model.gene_coef_jk,
    )
    n_blocks = fitted_model.n_jk_blocks
    source_position = {
        label: position for position, label in enumerate(fitted_model.kernel_labels)
    }
    source_by_burden = {}
    for position, kernel in enumerate(model.layout.kernel_ids):
        if kernel.kind == "B":
            source_by_burden[(kernel.col, kernel.col2)] = position

    trace_model, traces = trace_data
    excluded = set(exclude_genes)
    keep = np.asarray([gene not in excluded for gene in model.genes])
    annotations = trace_model.gene_annot_matrix()
    block_of_gene = fitted_model.gene_blocks
    fitting.require_nonempty_blocks(block_of_gene[keep], n_blocks)
    if traces.ndim == 3:
        trace_by_block = traces
    else:
        trace_by_block = np.zeros(
            (n_blocks, trace_model.layout.n_kernels, annotations.shape[1])
        )
        for block in range(n_blocks):
            selected = keep & (block_of_gene == block)
            trace_by_block[block] = np.einsum(
                "gk,ga->ka", traces[selected], annotations[selected]
            )
    trace = trace_by_block.sum(axis=0)
    trace_jk = trace[None] if n_blocks == 1 else trace - trace_by_block
    names = tuple(bins)
    position_of = {name: position for position, name in enumerate(names)}
    hsq = np.zeros(len(names))
    hsq_jk = np.zeros((n_blocks, len(names)))
    for target, kernel in enumerate(trace_model.layout.kernel_ids):
        if kernel.kind != "B":
            continue
        if model.options.burden_by_maf:
            source = source_position[kernel.label]
        else:
            source = source_by_burden[
                (kernel.col, None if kernel.col2 == kernel.col else kernel.col2)
            ]
        kernel_group = fitted_model.kernel_groups[source]
        if target_kernel_group is not None and kernel_group != target_kernel_group:
            continue
        contribution = float(fitted_model.kernel_coef[source]) * float(
            trace[target] @ fitted_model.gene_coef[:, kernel_group]
        )
        replicate = kernel_coef_jk[:, source] * np.einsum(
            "ba,ba->b", trace_jk[:, target], gene_coef_jk[:, :, kernel_group]
        )
        labels = tuple(
            dict.fromkeys(
                label
                for label in (kernel.maf_label, kernel.maf_label2)
                if label is not None
            )
        )
        for label in labels:
            position = position_of[label]
            hsq[position] += contribution / len(labels)
            hsq_jk[:, position] += replicate / len(labels)

    overdispersion = {
        (kernel.col, kernel.maf_label): index
        for index, kernel in enumerate(trace_model.layout.kernel_ids)
        if kernel.kind == "O"
    }
    for independent_position, independent in enumerate(
        model.layout.independent_indices
    ):
        if independent_position >= (
            0 if independent_trace is None else independent_trace.shape[1]
        ):
            continue
        source = source_position[model.layout.kernel_ids[independent].label]
        kernel_group = fitted_model.kernel_groups[source]
        if target_kernel_group is not None and kernel_group != target_kernel_group:
            continue
        column = model.layout.kernel_ids[independent].col
        overdispersion_indices = [overdispersion.get((column, name)) for name in names]
        if all(index is not None for index in overdispersion_indices):
            independent_blocks = trace_by_block[:, overdispersion_indices]
        else:
            if independent_trace is None:
                raise ValueError(
                    "MAF independent traces are missing; recompute sufficient statistics"
                )
            independent_blocks = independent_trace[:, independent_position]
        independent_total = independent_blocks.sum(axis=0)
        hsq += float(fitted_model.kernel_coef[source]) * (
            independent_total @ fitted_model.gene_coef[:, kernel_group]
        )
        independent_trace_jk = (
            independent_total[None]
            if n_blocks == 1
            else independent_total - independent_blocks
        )
        hsq_jk += kernel_coef_jk[:, source, None] * np.einsum(
            "bma,ba->bm", independent_trace_jk, gene_coef_jk[:, :, kernel_group]
        )

    if not np.isclose(
        hsq.sum(),
        expected_hsq,
        rtol=1e-9,
        atol=1e-11,
    ):
        raise RuntimeError(
            "MAF components do not recover fitted hsq: "
            f"observed {hsq.sum()}, expected {expected_hsq}"
        )
    if not np.allclose(hsq_jk.sum(axis=1), expected_hsq_jk, rtol=1e-9, atol=1e-11):
        raise RuntimeError("MAF delete-block components do not recover fitted hsq")
    return pl.DataFrame(
        {
            "maf_bin": names,
            "hsq": hsq,
            "hsq_se": fitting.jackknife_se(hsq_jk),
            "hsq_jk": hsq_jk.T.tolist(),
        },
        schema={
            "maf_bin": pl.String,
            "hsq": pl.Float64,
            "hsq_se": pl.Float64,
            "hsq_jk": pl.List(pl.Float64),
        },
    )


def _random_gene_annot_components(model, result, exclude_genes):
    """Return self-contained LD-weighted components by gene annotation."""
    genes = list(model.genes)
    excluded = set(exclude_genes)
    keep = np.asarray([gene not in excluded for gene in genes])
    annotations = model.gene_annot_matrix()[keep]
    membership = annotations.sum(axis=1)
    if not np.isin(annotations, (0.0, 1.0)).all() or not np.all(membership == 1):
        raise ValueError(
            "gene_annot_components require every retained gene to belong to "
            "exactly one gene annotation"
        )
    codes = annotations.argmax(axis=1)
    names = tuple(model.options.gene_annot_cols or ("constant",))
    block_of_gene = result.gene_blocks[keep]
    het = np.zeros(len(names))
    het_by_block = np.zeros((result.n_jk_blocks, len(names)))
    n_variants = np.zeros(len(names), dtype=np.int64)
    n_genes = np.bincount(codes, minlength=len(names))
    for gene, code, block in zip(np.asarray(genes)[keep], codes, block_of_gene):
        variant_het = model.variant_scale(str(gene)) ** 2
        value = float(variant_het.sum())
        het[code] += value
        het_by_block[block, code] += value
        n_variants[code] += len(variant_het)
    het_jk = het[None] if result.n_jk_blocks == 1 else het - het_by_block
    return pl.DataFrame(
        {
            "gene_annot": names,
            "hsq": result.gene_annot_hsq,
            "hsq_se": result.gene_annot_hsq_se,
            "hsq_jk": result.gene_annot_hsq_jk.T.tolist(),
            "het": het,
            "het_jk": het_jk.T.tolist(),
            "n_genes": n_genes,
            "n_variants": n_variants,
        },
        schema={
            "gene_annot": pl.String,
            "hsq": pl.Float64,
            "hsq_se": pl.Float64,
            "hsq_jk": pl.List(pl.Float64),
            "het": pl.Float64,
            "het_jk": pl.List(pl.Float64),
            "n_genes": pl.Int64,
            "n_variants": pl.Int64,
        },
    )


def fit_random_effects(
    model_or_models,
    *,
    exclude_genes=None,
    extra_enrich_col: str | None = None,
    weights=None,
    n_jk_blocks: int | None = None,
    gene_blocks=None,
    kernel_groups=None,
    init_gene_coef=None,
    fix_gene_coef: bool = False,
    extra_covar_kernel: str | None = None,
    psd_burden_coef: bool = False,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> (
    RandomFit
    | dict[str, RandomFit]
    | tuple[RandomFit, EnrichResult | _covar.CovarResult]
):
    """Fit one model or a mapping of model/statistic inputs.

    Selecting ``extra_enrich_col`` returns ``(fit, enrichment)`` from one
    augmented fit. Selecting ``extra_covar_kernel`` returns ``(fit, covar)``
    from one full covar-augmented fit. Otherwise return a RandomFit or the existing
    model mapping.
    """
    if type(psd_burden_coef) is not bool:
        raise TypeError("psd_burden_coef must be Boolean")
    if type(fix_gene_coef) is not bool:
        raise TypeError("fix_gene_coef must be a scalar Boolean")
    if fix_gene_coef and init_gene_coef is None:
        raise ValueError("fix_gene_coef=True requires init_gene_coef")
    if exclude_genes is not None:
        exclude_genes = _snapshot_gene_selection(exclude_genes)
    if isinstance(model_or_models, Mapping) and extra_enrich_col is not None:
        raise ValueError("extra_enrich_col requires a single model")
    if extra_enrich_col is not None and not isinstance(model_or_models, Mapping):
        if len(model_or_models.layout.kernel_group_names) != 1:
            raise ValueError("enrichment requires a single kernel group")
    if isinstance(model_or_models, Model):
        _require_suffstats(model_or_models)
        if model_or_models.jk_suffstats is not None:
            cached = model_or_models._get_jk_suffstats(
                exclude_genes=exclude_genes,
                n_jk_blocks=n_jk_blocks,
                gene_blocks=gene_blocks,
            )
            if exclude_genes is None:
                exclude_genes = cached.excluded_genes
            n_jk_blocks = cached.n_jk_blocks
    elif isinstance(model_or_models, JKSuffStats):
        model_or_models.validate_jk_shape(
            exclude_genes=exclude_genes,
            n_jk_blocks=n_jk_blocks,
            gene_blocks=gene_blocks,
        )
        n_jk_blocks = (
            model_or_models.n_jk_blocks if n_jk_blocks is None else n_jk_blocks
        )
    configured_groups = (
        not isinstance(model_or_models, Mapping)
        and getattr(getattr(model_or_models, "options", None), "kernel_groups", None)
        is not None
    )
    grouped = kernel_groups is not None or psd_burden_coef or configured_groups
    if extra_covar_kernel is not None and not isinstance(extra_covar_kernel, str):
        raise TypeError("extra_covar_kernel must be a string")
    if extra_covar_kernel is not None and isinstance(model_or_models, Mapping):
        raise ValueError("extra covariance kernels require a single Model")
    mapping = isinstance(model_or_models, Mapping)
    if mapping:
        base, exclude_genes, n_jk_blocks = _prepare_random_mapping(
            model_or_models,
            exclude_genes=exclude_genes,
            n_jk_blocks=n_jk_blocks,
            gene_blocks=gene_blocks,
            grouped=grouped,
        )
    else:
        model = model_or_models
        if weights is not None:
            raise ValueError("weights require a model mapping")
        if isinstance(model, JKSuffStats):
            if not fix_gene_coef and (
                isinstance(model.excluded_genes, dict)
                or any(kernel.kind == "C" for kernel in model.layout.kernel_ids)
            ):
                raise ValueError(
                    "per-group support and covar designs require fix_gene_coef=True"
                )
            if kernel_groups is not None or psd_burden_coef:
                raise ValueError(
                    "fit-time grouping and PSD constraints require a Model"
                )
            if extra_enrich_col is not None and extra_covar_kernel is not None:
                raise ValueError(
                    "enrichment cannot be combined with extra covariance kernels"
                )
            if extra_covar_kernel is not None and not fix_gene_coef:
                raise ValueError("extra covariance fits require fix_gene_coef=True")
            base = model
        else:
            if exclude_genes is None:
                raise ValueError("exclude_genes is required")
            if configured_groups and kernel_groups is not None:
                raise ValueError(
                    "fit-time kernel_groups cannot override KernelOptions.kernel_groups"
                )
            if extra_covar_kernel is not None and not configured_groups:
                raise ValueError(
                    "extra covariance kernels require KernelOptions.kernel_groups"
                )
            if extra_enrich_col is not None:
                if kernel_groups is not None or psd_burden_coef:
                    raise ValueError(
                        "enrichment cannot be combined with kernel grouping"
                    )
                if extra_covar_kernel is not None or isinstance(exclude_genes, Mapping):
                    raise ValueError(
                        "enrichment cannot be combined with extra covariance kernels or "
                        "per-group fits"
                    )
                _validate_burden_annotation(model, extra_enrich_col)
            else:
                if extra_covar_kernel is not None or _has_group_specific_exclusions(
                    exclude_genes
                ):
                    if not configured_groups:
                        raise ValueError(
                            "per-group exclude_genes and extra covariance fits require "
                            "KernelOptions.kernel_groups"
                        )
                    if not fix_gene_coef:
                        raise ValueError(
                            "per-group exclude_genes and extra covariance fits require fix_gene_coef=True"
                        )
                if extra_covar_kernel is not None and not isinstance(
                    exclude_genes, Mapping
                ):
                    exclude_genes = {
                        name: exclude_genes for name in model.options.kernel_groups
                    }
                if kernel_groups is not None:
                    kernel_groups = make_kernel_groups(model, kernel_groups)
                if extra_covar_kernel is not None and psd_burden_coef:
                    raise ValueError(
                        "covar extension fitting cannot be combined with PSD constraints"
                    )
            base = model._get_jk_suffstats(
                exclude_genes=exclude_genes,
                n_jk_blocks=n_jk_blocks,
                gene_blocks=gene_blocks,
            )

    fit_options = dict(
        n_jk_blocks=n_jk_blocks,
        init_gene_coef=init_gene_coef,
        fix_gene_coef=fix_gene_coef,
        max_iter=max_iter,
        tol=tol,
    )
    extension_result = None
    if extra_enrich_col is not None or extra_covar_kernel is not None:
        result, extension_result = _fit_extensions(
            base,
            extra_enrich_col=extra_enrich_col,
            extra_covar_kernel=extra_covar_kernel,
            **fit_options,
        )
    else:
        result = _fit_random_effects(
            base,
            weights=weights,
            kernel_groups=kernel_groups,
            psd_burden_coef=psd_burden_coef,
            **fit_options,
        )
        if not mapping:
            result = result["model"]

    if mapping:
        return {
            name: _assemble_random_reports(result[name], source, base[name])
            if isinstance(source, Model)
            else result[name]
            for name, source in model_or_models.items()
        }
    if isinstance(model_or_models, Model):
        report_model = model_or_models
        report_stats = base
        if extension_result is not None:
            extension = (
                base.burden_extensions[extra_enrich_col]
                if extra_enrich_col is not None
                else _select_covar_extension(base.covar_extensions, extra_covar_kernel)
            )
            report_stats = _select_jk_extension(base, extension)
            if extra_enrich_col is not None:
                report_model = report_model._with_burden_column(extra_enrich_col)
        result = _assemble_random_reports(
            result, report_model, report_stats, include_maf=extension_result is None
        )
    return result if extension_result is None else (result, extension_result)


def _prepare_random_mapping(
    model_or_models, *, exclude_genes, n_jk_blocks, gene_blocks, grouped
):
    """Coordinate exclusions and block counts before creating joint-fit caches."""
    models = dict(model_or_models)
    if not models:
        raise ValueError("model mapping must not be empty")
    if any(
        (isinstance(model, JKSuffStats) and model.kernel_options.get("kernel_groups"))
        or getattr(getattr(model, "options", None), "kernel_groups", None)
        for model in models.values()
    ):
        raise ValueError("grouped inputs cannot be combined in a model mapping")
    if grouped:
        raise ValueError("kernel grouping requires a single model")
    if not all(isinstance(model, (Model, JKSuffStats)) for model in models.values()):
        raise TypeError("mapping values must be Models or JKSuffStats")
    compact = all(isinstance(model, JKSuffStats) for model in models.values())
    if not compact and any(isinstance(model, JKSuffStats) for model in models.values()):
        raise TypeError("mapping values must be either all Models or all JKSuffStats")
    labels = set(models)
    if exclude_genes is None:
        if any(
            isinstance(value, Model) and value.jk_suffstats is None
            for value in models.values()
        ):
            raise ValueError("exclude_genes is required")
        exclusions = {
            name: (
                (
                    value if isinstance(value, JKSuffStats) else value.jk_suffstats
                ).excluded_genes
            )
            for name, value in models.items()
        }
    elif isinstance(exclude_genes, Mapping):
        if set(exclude_genes) != labels:
            raise ValueError("exclude_genes keys must match the model labels")
        exclusions = dict(exclude_genes)
    else:
        requested = set(() if exclude_genes is None else exclude_genes)
        union = set().union(*(set(value.genes) for value in models.values()))
        if unknown := requested - union:
            raise KeyError(f"excluded genes are not in any model: {sorted(unknown)}")
        exclusions = {
            name: tuple(requested & set(value.genes)) for name, value in models.items()
        }
    if gene_blocks is not None:
        if not isinstance(gene_blocks, Mapping) or set(gene_blocks) != labels:
            raise ValueError("gene_blocks keys must match the model labels")
        blocks_by_model = dict(gene_blocks)
    else:
        blocks_by_model = {name: None for name in models}

    prepared_counts = {
        value.n_jk_blocks for value in models.values() if isinstance(value, JKSuffStats)
    } | {
        value.jk_suffstats.n_jk_blocks
        for value in models.values()
        if isinstance(value, Model) and value.jk_suffstats is not None
    }
    if len(prepared_counts) > 1:
        raise ValueError("all prepared inputs must use the same n_jk_blocks")
    if prepared_counts:
        prepared_count = next(iter(prepared_counts))
        if n_jk_blocks is not None and n_jk_blocks != prepared_count:
            raise ValueError("prepared n_jk_blocks does not match request")
        common_blocks = prepared_count
    else:
        common_blocks = (
            min(100, *(len(value.genes) for value in models.values()))
            if n_jk_blocks is None
            else n_jk_blocks
        )

    # Validate support and materialize per-gene statistics before creating any
    # JK caches, so ordinary request errors do not partially prepare a mapping.
    for name, value in models.items():
        if isinstance(value, Model):
            _require_suffstats(value)
            unknown = set(exclusions[name]) - set(value.genes)
            if unknown:
                raise KeyError(
                    f"excluded genes are not in model {name!r}: {sorted(unknown)}"
                )
        has_prepared = (
            value.jk_suffstats is not None if isinstance(value, Model) else True
        )
        if has_prepared:
            stat = value if isinstance(value, JKSuffStats) else value.jk_suffstats
            stat.validate_jk_shape(
                exclude_genes=exclusions[name],
                n_jk_blocks=common_blocks,
                gene_blocks=blocks_by_model[name],
            )
    stats = {}
    for name, value in models.items():
        requested = exclusions[name]
        stat = (
            value
            if isinstance(value, JKSuffStats)
            else value._get_jk_suffstats(
                exclude_genes=requested,
                n_jk_blocks=common_blocks,
                gene_blocks=blocks_by_model[name],
            )
        )
        stats[name] = stat
    return stats, exclusions, common_blocks


def _fit_extensions(
    base, *, extra_enrich_col=None, extra_covar_kernel=None, **fit_options
):
    """Select one determined system per fit; keep the common solver lifecycle."""
    if extra_enrich_col is not None:
        if len(base.layout.kernel_group_names) != 1:
            raise ValueError("enrichment requires a single kernel group")
        if extra_enrich_col not in base.burden_extensions:
            raise KeyError(f"unknown burden extension: {extra_enrich_col!r}")
        extension = base.burden_extensions[extra_enrich_col]
        fit = _fit_random_effects(_select_jk_extension(base, extension), **fit_options)[
            "model"
        ]
        attribution = dict(
            vars(extension.payload),
            annotation=extra_enrich_col,
            total_het=base.total_het,
        )
        return fit, _random_enrich_result(attribution, fit)
    extension = _select_covar_extension(base.covar_extensions, extra_covar_kernel)
    selected = _select_jk_extension(base, extension)
    if len(extension.new_indices) > 1:
        # Exact D/indep redundancy is a rank failure, including in deletes.
        # Diagnose it before the common solve gives its generic rank error.
        d = extension.new_indices[0]
        for i in base.layout.independent_indices:
            difference = (
                selected.blocks["pt0"][:, d, d]
                + selected.blocks["pt0"][:, i, i]
                - 2 * selected.blocks["pt0"][:, d, i]
            )
            if selected.blocks["pt0"][:, d, d].sum() > 0 and np.array_equal(
                difference, np.zeros_like(difference)
            ):
                raise ValueError(
                    f"covar extension {extra_covar_kernel!r}: covar independent kernel is exactly redundant with {base.kernel_labels[i]}"
                )
    fit = _fit_random_effects(selected, **fit_options)["model"]
    return fit, _covar._project_covar_fit(base, fit, extension)


def fit_fixed_effects(
    model_or_models,
    *,
    include_genes,
    method: str = "burden",
    extra_enrich_col: str | None = None,
    burden_1d_wgts=None,
    burden_indices=None,
    kernel_options=None,
):
    """Fit explicitly selected genes as fixed effects.

    Pass either a prepared ``Model`` or a ``Dataset`` with ``kernel_options``.
    """
    if include_genes is None:
        raise ValueError("include_genes is required")
    include_genes = _snapshot_gene_selection(include_genes)
    if burden_1d_wgts is not None:
        burden_1d_wgts = dict(burden_1d_wgts)
    if burden_indices is not None:
        burden_indices = tuple(burden_indices)
    if isinstance(model_or_models, Dataset):
        if kernel_options is None:
            raise ValueError("kernel_options is required with a Dataset")
        model_or_models = Model(model_or_models, kernel_options)
    elif kernel_options is not None:
        raise TypeError("kernel_options is only valid with a Dataset")
    if isinstance(model_or_models, Mapping):
        models = dict(model_or_models)
        if extra_enrich_col is not None:
            raise ValueError("extra_enrich_col requires a single model")
        selected = (
            include_genes
            if isinstance(include_genes, Mapping)
            else {label: include_genes for label in models}
        )
        if set(selected) != set(models):
            raise ValueError("include_genes keys must match model labels")
        return {
            label: fit_fixed_effects(
                model,
                include_genes=selected[label],
                method=method,
                extra_enrich_col=extra_enrich_col,
                burden_1d_wgts=burden_1d_wgts,
                burden_indices=burden_indices,
            )
            for label, model in models.items()
        }

    model = model_or_models
    if isinstance(include_genes, Mapping):
        raise TypeError("include_genes mapping requires a model mapping")
    genes = tuple(dict.fromkeys(include_genes))
    if extra_enrich_col is not None:
        if not genes:
            raise ValueError("enrichment requires at least one fixed gene")
        if method != "burden":
            raise ValueError("extra_enrich_col requires method='burden'")
        if burden_1d_wgts is not None or burden_indices is not None:
            raise ValueError(
                "extra_enrich_col cannot be combined with burden_1d_wgts or "
                "burden_indices"
            )
    effective_burden_1d_wgts = (
        None
        if method != "burden_1d"
        else (
            None
            if burden_1d_wgts is None and model.burden_1d_wgts is None
            else dict(
                model.burden_1d_wgts if burden_1d_wgts is None else burden_1d_wgts
            )
        )
    )
    result = _fit_fixed_effects(
        model,
        include_genes=genes,
        method=method,
        burden_1d_wgts=effective_burden_1d_wgts,
        burden_indices=burden_indices,
        extra_enrich_col=extra_enrich_col,
    )
    if extra_enrich_col is not None:
        return result, _fixed_enrich_result(model, result, extra_enrich_col)
    return result


def combine_random_hsq(fits: Mapping[str, RandomFit]) -> tuple[float, float]:
    """Sum nonoverlapping random fits using their aligned delete replicates."""
    fits = dict(fits)
    if not fits or any(not isinstance(fit, RandomFit) for fit in fits.values()):
        raise TypeError("fits must be a nonempty mapping of RandomFit values")
    n_blocks = {fit.n_jk_blocks for fit in fits.values()}
    if len(n_blocks) != 1:
        raise ValueError("random fits must have aligned jackknife block counts")
    for fit in fits.values():
        if np.asarray(fit.hsq_jk).shape != (fit.n_jk_blocks,):
            raise ValueError("random fits must have aligned jackknife arrays")
    hsq = float(sum(fit.hsq for fit in fits.values()))
    deleted = sum(
        (np.asarray(fit.hsq_jk) for fit in fits.values()),
        start=np.zeros(n_blocks.pop()),
    )
    return (
        hsq,
        float(fitting.jackknife_se(deleted[:, None])[0])
        if len(deleted) > 1
        else np.nan,
    )


def combine_mixed_effects(
    *, random_fit: RandomFit, fixed_fit: FixedFit
) -> tuple[float, float]:
    """Combine complementary random and fixed gene sets without refitting."""
    if not isinstance(random_fit, RandomFit):
        raise TypeError("random_fit must be a RandomFit")
    if not isinstance(fixed_fit, FixedFit):
        raise TypeError("fixed_fit must be a FixedFit")
    if isinstance(random_fit.excluded_genes, dict):
        raise TypeError(
            "grouped random exclusions cannot be combined with one fixed fit"
        )
    excluded = set(random_fit.excluded_genes)
    fixed_genes = set(fixed_fit.per_gene["gene"])
    if fixed_genes != excluded:
        raise ValueError("fixed genes must exactly match the random exclusions")
    return (
        float(random_fit.hsq + fixed_fit.hsq),
        float(np.hypot(random_fit.hsq_se, fixed_fit.hsq_se)),
    )
