"""Random-effect estimators: hsq of the subthreshold gene set.

"Random" means the genes are treated as exchangeable draws, so uncertainty comes
from a block jackknife over genomic blocks. Every estimator here requires
``exclude_genes``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from functools import cached_property
from io import BytesIO
from typing import ClassVar

import numpy as np
import polars as pl

from kasm import fitting
from kasm._logging import logger


@dataclass(frozen=True)
class RandomFit:
    """One random-effect model's parameters, estimates, and diagnostics.

    This is what makes a fitted model reusable: the kernel coefficients and the
    gene coefficients are all that variant scoring needs, so a fit learned on
    one chromosome can score another without repeating the LD-dependent work.

    ``gene_coef`` has one column per kernel group and ``kernel_groups`` names
    each kernel's column, or ``-1`` for a kernel that carries no gene-level
    coefficient. An ungrouped fit therefore has a single column, and a grouped
    fit — one that lets, say, pLoF and missense kernels scale differently with
    gene constraint — is representable without a second result type.
    """

    kernel_coef: np.ndarray
    gene_coef: np.ndarray
    kernel_groups: np.ndarray
    residual: float
    sample_size: float
    kernel_labels: tuple[str, ...]
    gene_annot_cols: tuple[str, ...]
    heritable_labels: tuple[str, ...]
    kernel_options: dict
    hsq: float = np.nan
    hsq_se: float = np.nan
    variance: np.ndarray | None = None
    variance_se: np.ndarray | None = None
    kernel_coef_se: np.ndarray | None = None
    kernel_coef_jk: np.ndarray | None = None
    gene_coef_jk: np.ndarray | None = None
    variance_jk: np.ndarray | None = None
    hsq_jk: np.ndarray | None = None
    gene_annot_hsq: np.ndarray | None = None
    gene_annot_hsq_se: np.ndarray | None = None
    gene_annot_hsq_jk: np.ndarray | None = None
    kernel_group_names: tuple[str, ...] = ()
    joint: bool = False
    n_genes: int | None = 0
    n_variants: int | None = 0
    n_jk_blocks: int = 0
    n_iter: int = 0
    converged: bool = False
    n_iter_jk: np.ndarray | None = None
    converged_jk: np.ndarray | None = None
    minimum_eigenvalues: np.ndarray | None = None
    excluded_genes: tuple[str, ...] | dict[str, tuple[str, ...]] = ()
    gene_blocks: np.ndarray | None = None
    n_genes_per_annot: pl.DataFrame | None = None
    weight: float = 1.0
    kernel_group_counts: np.ndarray | None = None
    _maf_components: pl.DataFrame | None = None
    _gene_annot_components: pl.DataFrame | None = None

    _OPTIONAL_ARRAY_FIELDS: ClassVar[tuple[str, ...]] = (
        "variance",
        "variance_se",
        "kernel_coef_se",
        "kernel_coef_jk",
        "gene_coef_jk",
        "variance_jk",
        "hsq_jk",
        "gene_annot_hsq",
        "gene_annot_hsq_se",
        "gene_annot_hsq_jk",
        "n_iter_jk",
        "converged_jk",
        "minimum_eigenvalues",
        "gene_blocks",
        "kernel_group_counts",
    )

    @staticmethod
    def _frame_bytes(frame: pl.DataFrame) -> np.ndarray:
        buffer = BytesIO()
        frame.write_ipc(buffer)
        return np.frombuffer(buffer.getvalue(), dtype=np.uint8)

    @staticmethod
    def _frame_from_bytes(value: np.ndarray) -> pl.DataFrame:
        return pl.read_ipc(BytesIO(value.tobytes()))

    def __post_init__(self):
        gene_coef = np.asarray(self.gene_coef, dtype=float)
        if gene_coef.ndim != 2:
            raise ValueError(
                "gene_coef must be two-dimensional, (n_annotations, n_kernel_groups)"
            )
        object.__setattr__(self, "gene_coef", gene_coef)
        object.__setattr__(
            self, "kernel_groups", np.asarray(self.kernel_groups, dtype=np.intp)
        )
        if isinstance(self.excluded_genes, dict):
            object.__setattr__(
                self,
                "excluded_genes",
                {
                    name: tuple(sorted(set(genes)))
                    for name, genes in self.excluded_genes.items()
                },
            )
        else:
            object.__setattr__(
                self, "excluded_genes", tuple(sorted(set(self.excluded_genes)))
            )

    @property
    def n_kernel_groups(self) -> int:
        return self.gene_coef.shape[1]

    @property
    def point(self) -> pl.DataFrame:
        rows = [(None, self.hsq, self.hsq_se, self.n_genes, self.n_variants)]
        for position, name in enumerate(self.kernel_group_names):
            selected = (self.kernel_groups == position) & np.isin(
                self.kernel_labels, self.heritable_labels
            )
            deleted = self.variance_jk[:, selected].sum(axis=1)
            rows.append(
                (
                    name,
                    float(self.variance[selected].sum()),
                    float(fitting.jackknife_se(deleted[:, None])[0]),
                    *self.kernel_group_counts[position],
                )
            )
        return pl.DataFrame(
            rows,
            orient="row",
            schema={
                "kernel_group": pl.String,
                "hsq": pl.Float64,
                "hsq_se": pl.Float64,
                "n_genes": pl.Int64,
                "n_variants": pl.Int64,
            },
        )

    @property
    def jk(self) -> pl.DataFrame:
        values = [self.hsq_jk] + [
            self.variance_jk[
                :,
                (self.kernel_groups == position)
                & np.isin(self.kernel_labels, self.heritable_labels),
            ].sum(axis=1)
            for position in range(len(self.kernel_group_names))
        ]
        return pl.DataFrame(
            {
                "block": np.tile(np.arange(self.n_jk_blocks), len(values)),
                "kernel_group": np.repeat(
                    [None, *self.kernel_group_names], self.n_jk_blocks
                ).tolist(),
                "hsq": np.concatenate(values),
            },
            schema={"block": pl.Int64, "kernel_group": pl.String, "hsq": pl.Float64},
        )

    @cached_property
    def components(self) -> pl.DataFrame:
        from kasm.kernels import KernelId, KernelOptions

        layout = KernelOptions.from_recorded(self.kernel_options).layout
        kernels = {kernel.label: kernel for kernel in layout.kernel_ids}
        for label in self.kernel_labels:
            if label not in kernels and label.startswith(("C:", "I:")):
                kind, column = label.split(":", 1)
                kernels[label] = KernelId(kind, column)
        rows = []
        for position, label in enumerate(self.kernel_labels):
            kernel = kernels[label]
            owner = int(self.kernel_groups[position])
            rows.append(
                {
                    **(
                        {
                            "kernel_group": None
                            if owner < 0
                            else self.kernel_group_names[owner]
                        }
                        if self.kernel_group_names
                        else {}
                    ),
                    "kernel": kernel.label,
                    "kind": kernel.kind,
                    "annotation": kernel.col,
                    "annotation_2": kernel.col2,
                    "maf_bin": kernel.maf_label,
                    "maf_bin_2": kernel.maf_label2,
                    "heritable": bool(kernel.is_heritable),
                    "kernel_coef": float(self.kernel_coef[position]),
                    "kernel_coef_se": float(self.kernel_coef_se[position]),
                    "variance": float(self.variance[position]),
                    "variance_se": float(self.variance_se[position]),
                }
            )
        return pl.DataFrame(rows)

    @cached_property
    def overdispersion_components(self) -> pl.DataFrame:
        from kasm.api import _overdispersion_table
        from kasm.kernels import KernelOptions

        return _overdispersion_table(
            KernelOptions.from_recorded(self.kernel_options).layout,
            self.kernel_coef,
            (
                fitting.jackknife_se(self.kernel_coef_jk)
                if self.joint and self.n_jk_blocks > 1
                else self.kernel_coef_se
            ),
            self.variance,
            self.variance_se,
        )

    @property
    def maf_components(self) -> pl.DataFrame:
        if self._maf_components is None:
            raise TypeError("MAF components are unavailable for this fit configuration")
        return self._maf_components

    @property
    def gene_annot_components(self) -> pl.DataFrame:
        if self._gene_annot_components is None:
            raise TypeError(
                "gene annotation components require a single kernel group and exhaustive gene annotations"
            )
        return self._gene_annot_components

    @property
    def gene_coef_dict(self):
        names = self.gene_annot_cols or ("constant",)
        if not self.kernel_group_names:
            return dict(zip(names, self.gene_coef[:, 0], strict=True))
        return {
            name: dict(zip(names, self.gene_coef[:, position], strict=True))
            for position, name in enumerate(self.kernel_group_names)
        }

    def save(self, path) -> None:
        """Write to a compressed ``.npz``."""
        stored = dict(  # noqa: C408
            kernel_coef=self.kernel_coef,
            gene_coef=self.gene_coef,
            kernel_groups=self.kernel_groups,
            residual=np.array(self.residual),
            sample_size=np.array(self.sample_size),
            kernel_labels=np.array(self.kernel_labels, dtype=str),
            gene_annot_cols=np.array(list(self.gene_annot_cols), dtype=str),
            heritable_labels=np.array(self.heritable_labels, dtype=str),
            # Not sort_keys: the declared order of maf_bins determines kernel
            # order, so sorting it would misrepresent the configuration.
            kernel_options_json=np.array(json.dumps(self.kernel_options)),
            result_json=np.array(
                json.dumps(
                    {
                        "hsq": self.hsq,
                        "hsq_se": self.hsq_se,
                        "kernel_group_names": self.kernel_group_names,
                        "joint": self.joint,
                        "n_genes": self.n_genes,
                        "n_variants": self.n_variants,
                        "n_jk_blocks": self.n_jk_blocks,
                        "n_iter": self.n_iter,
                        "converged": self.converged,
                        "excluded_genes": self.excluded_genes,
                        "weight": self.weight,
                    }
                )
            ),
        )
        for name in self._OPTIONAL_ARRAY_FIELDS:
            value = getattr(self, name)
            if value is not None:
                stored[name] = value
        for name in ("n_genes_per_annot", "_maf_components", "_gene_annot_components"):
            value = getattr(self, name)
            if value is not None:
                stored[name] = self._frame_bytes(value)
        np.savez_compressed(path, **stored)

    @classmethod
    def load(cls, path) -> RandomFit:
        with np.load(path, allow_pickle=False) as archive:
            stored = {name: archive[name] for name in archive.files}
        kernel_labels = tuple(stored["kernel_labels"].tolist())
        result = json.loads(str(stored["result_json"]))
        result["kernel_group_names"] = tuple(result["kernel_group_names"])
        arrays = {
            name: stored[name] for name in cls._OPTIONAL_ARRAY_FIELDS if name in stored
        }
        for name in ("n_genes_per_annot", "_maf_components", "_gene_annot_components"):
            if name in stored:
                arrays[name] = cls._frame_from_bytes(stored[name])
        return cls(
            kernel_coef=stored["kernel_coef"],
            gene_coef=stored["gene_coef"],
            kernel_groups=stored["kernel_groups"],
            residual=float(stored["residual"]),
            sample_size=float(stored["sample_size"]),
            kernel_labels=kernel_labels,
            gene_annot_cols=tuple(stored["gene_annot_cols"].tolist()),
            heritable_labels=tuple(stored["heritable_labels"].tolist()),
            kernel_options=json.loads(str(stored["kernel_options_json"])),
            **result,
            **arrays,
        )

    def gene_scale(self, model, *, burden_col: str | None = None) -> np.ndarray:
        """Per-gene, per-group coefficient ``A eta``, shape ``(n_genes, n_kernel_groups)``.

        Aligned to *model*'s gene-annotation columns, so a fit can scale a gene
        set it was not fitted on. When ``burden_col`` is supplied, return the
        one-dimensional scale for that annotation's unique kernel group.
        """
        names = list(model.options.gene_annot_cols or ["constant"])
        recorded = list(self.gene_annot_cols or ("constant",))
        missing = set(names) - set(recorded)
        if missing:
            raise ValueError(f"this fit has no gene coefficient for {sorted(missing)}")
        coefficients = self.gene_coef[[recorded.index(name) for name in names]]
        annot = model.gene_annot_matrix()
        if not model.options.gene_annot_cols:
            scale = np.repeat(coefficients[:1], annot.shape[0], axis=0)
        else:
            scale = annot @ coefficients
        if burden_col is None:
            return scale

        matches = {
            int(self.kernel_groups[self.kernel_labels.index(kernel.label)])
            for kernel in model.layout.kernel_ids
            if kernel.kind == "B"
            and not kernel.is_cross
            and kernel.col == burden_col
            and kernel.label in self.kernel_labels
        }
        if not matches:
            raise ValueError(f"no burden kernels match annotation {burden_col!r}")
        if -1 in matches:
            raise ValueError(f"burden annotation {burden_col!r} is ungrouped")
        if len(matches) != 1:
            raise ValueError(
                f"burden annotation {burden_col!r} matches multiple gene-scale groups"
            )
        return scale[:, matches.pop()]

    def overdispersion_variance(self, dataset) -> np.ndarray:
        """Return the fitted residual-variance contribution per variant."""
        from kasm.kernels import KernelOptions
        from kasm.model import Model

        model = Model(dataset, KernelOptions.from_recorded(self.kernel_options))
        indices = list(model.layout.overdispersion_indices)
        if not indices:
            return np.zeros(dataset.n_variants, dtype=float)
        position_of = {label: i for i, label in enumerate(self.kernel_labels)}
        labels = [model.layout.kernel_ids[index].label for index in indices]
        missing = [label for label in labels if label not in position_of]
        if missing:
            raise ValueError(f"this fit has no coefficient for kernels {missing}")
        coefficients = self.kernel_coef[[position_of[label] for label in labels]]
        return model.overdispersion_weights() @ coefficients

    def score_variants(self, dataset) -> pl.DataFrame:
        """Within-gene scores for a single-group model, with unit gene coefficients.

        Returns ``variant``, ``gene``, ``effsq``, and ``hsq``. ``effsq`` is the
        diagonal of the burden, cross, and independent kernel sum with fitted
        kernel coefficients and unit gene coefficients.
        ``hsq = effsq * model.variant_scale() ** 2`` is the corresponding
        diagonal heritability score.

        Cross kernels contribute ``2 theta_c u_left u_right`` to the diagonal.
        Aligned by ``(variant, gene)``, so a fit can score another gene set.
        """
        if self.n_kernel_groups != 1:
            raise ValueError("score_variants requires a single kernel group")
        from kasm.kernels import KernelOptions
        from kasm.model import Model

        options = KernelOptions.from_recorded(self.kernel_options)
        model = Model(dataset, options)
        burden_wgts = model.burden_wgts()
        # Match coefficients by label to support reordered MAF bins.
        layout = model.layout
        scoring_indices = (
            layout.burden_indices + layout.cross_indices + layout.independent_indices
        )
        coefficients = dict(zip(self.kernel_labels, self.kernel_coef, strict=True))
        missing = [
            layout.kernel_ids[index].label
            for index in scoring_indices
            if layout.kernel_ids[index].label not in coefficients
        ]
        if missing:
            raise ValueError(f"this fit has no coefficient for kernels {missing}")

        effsq = np.zeros(burden_wgts.shape[0])

        def accumulate(kernel: int, values: np.ndarray) -> None:
            effsq[:] += coefficients[layout.kernel_ids[kernel].label] * values

        for column, kernel in enumerate(layout.burden_indices):
            accumulate(kernel, burden_wgts[:, column] ** 2)
        for kernel, (left, right) in zip(layout.cross_indices, layout.cross_pairs):
            accumulate(kernel, 2.0 * burden_wgts[:, left] * burden_wgts[:, right])
        diagonal_wgts = model.diag_kernel_weights()
        diagonal_position = {
            index: position for position, index in enumerate(layout.diagonal_indices)
        }
        for index in layout.independent_indices:
            accumulate(index, diagonal_wgts[:, diagonal_position[index]])

        return (
            dataset.variant_df()
            .select("variant", "gene")
            .with_columns(
                pl.Series("effsq", effsq),
                pl.Series("hsq", effsq * model.variant_scale() ** 2),
            )
        )


def make_kernel_groups(
    model, base_kernel_groups: Mapping[str, object]
) -> dict[str, list[int]]:
    """Expand annotation groups and exact kernel selectors.

    Cross kernels within one annotation group stay in that group. Cross kernels
    spanning two groups receive a deterministic ``<left>_x_<right>`` group.
    A structured :class:`KernelId` selector assigns one exact kernel. Every
    heritable kernel must be covered exactly; overdispersion is omitted.
    """
    from kasm.kernels import KernelId

    if not base_kernel_groups:
        raise ValueError("base_kernel_groups cannot be empty")
    annotation_kernel_group = {}
    explicit_kernel_group = {}
    for kernel_group, kernel_group_annotations in base_kernel_groups.items():
        kernel_group_annotations = tuple(kernel_group_annotations)
        if not kernel_group_annotations:
            raise ValueError(f"kernel group {kernel_group!r} has no annotations")
        for selector in kernel_group_annotations:
            if isinstance(selector, KernelId):
                if selector not in model.layout.kernel_ids:
                    raise ValueError(
                        f"kernel group {kernel_group!r} selects absent kernel {selector!r}"
                    )
                index = model.layout.index_of(selector)
                if not selector.is_heritable:
                    raise ValueError(
                        f"kernel group {kernel_group!r} selects non-heritable kernel "
                        f"{selector!r}"
                    )
                if index in explicit_kernel_group:
                    raise ValueError(f"kernel {selector!r} appears in multiple groups")
                explicit_kernel_group[index] = kernel_group
                continue
            if not isinstance(selector, str):
                raise TypeError("kernel group selectors must be strings or KernelId")
            if selector in annotation_kernel_group:
                raise ValueError(f"annotation {selector!r} appears in multiple groups")
            annotation_kernel_group[selector] = kernel_group

    kernel_groups = {kernel_group: [] for kernel_group in base_kernel_groups}
    unassigned = []
    for index, kernel in enumerate(model.layout.kernel_ids):
        if not kernel.is_heritable:
            continue
        if index in explicit_kernel_group:
            if kernel.kind == "B":
                left = annotation_kernel_group.get(kernel.col)
                right = (
                    left
                    if not kernel.is_cross
                    else annotation_kernel_group.get(kernel.col2)
                )
                if left is not None and right is not None:
                    raise ValueError(
                        f"kernel {kernel!r} is selected both explicitly and by annotation"
                    )
            kernel_groups[explicit_kernel_group[index]].append(index)
            continue
        if kernel.kind != "B":
            unassigned.append(index)
            continue
        left = annotation_kernel_group.get(kernel.col)
        right = (
            left if not kernel.is_cross else annotation_kernel_group.get(kernel.col2)
        )
        if left is None or right is None:
            unassigned.append(index)
            continue
        kernel_group = left if left == right else "_x_".join(sorted((left, right)))
        kernel_groups.setdefault(kernel_group, []).append(index)

    kernel_groups = {
        kernel_group: indices
        for kernel_group, indices in kernel_groups.items()
        if indices
    }
    covered = {index for indices in kernel_groups.values() for index in indices}
    heritable = set(model.layout.heritable_indices)
    if covered != heritable:
        raise ValueError(
            "kernel groups do not cover every heritable kernel; "
            f"unassigned={unassigned[:5]}"
        )
    return kernel_groups


def _burden_psd_constraint_basis(layout):
    basis = np.zeros((layout.n_kernels, layout.n_burden, layout.n_burden))
    for column, kernel in enumerate(layout.burden_indices):
        basis[kernel, column, column] = 1.0
    for kernel, (left, right) in zip(layout.cross_indices, layout.cross_pairs):
        basis[kernel, left, right] = 1.0
        basis[kernel, right, left] = 1.0
    return basis


def _fit_point_and_deletes(
    fit_once,
    delete_blocks,
    *,
    point_start,
    deleted_start,
    started,
):
    """Run one point fit and its delete-block fits with a shared start policy."""
    point = fit_once(None, point_start)
    blocks = tuple(delete_blocks)
    if len(blocks) == 1:
        return point, [point]
    start = deleted_start(point)
    replicates = []
    for position, block in enumerate(blocks):
        replicate = fit_once(block, start)
        replicates.append(replicate)
        if (position + 1) % 10 == 0:
            logger.info(
                "event=fit_jk_progress blocks=%d/%d elapsed_seconds=%.3f",
                position + 1,
                len(blocks),
                time.monotonic() - started,
            )
    return point, replicates


def _gene_annot_hsq(theta, gene_coef, trace_by_annot, heritable):
    """LD-weighted hsq attributed to each gene-annotation column."""
    return np.asarray(gene_coef) * (
        np.asarray(theta)[heritable] @ np.asarray(trace_by_annot)[heritable]
    )


def _require_gene_coef_keys(values, labels, context):
    if not isinstance(values, dict):
        raise TypeError(f"{context} must be a dictionary")
    missing, extra = set(labels) - set(values), set(values) - set(labels)
    if missing or extra:
        raise ValueError(
            f"{context} keys must match {list(labels)}; "
            f"missing labels: {sorted(missing, key=str)}; extra labels: {sorted(extra, key=str)}"
        )


def _prepare_gene_coef(values, annotations, kernel_group_names, context):
    """Align public dictionaries once to metadata-defined annotation x group arrays."""
    _require_gene_coef_keys(values, kernel_group_names or annotations, context)
    columns = []
    for kernel_group in kernel_group_names or (None,):
        supplied = values if kernel_group is None else values[kernel_group]
        label = context if kernel_group is None else f"{context}[{kernel_group!r}]"
        _require_gene_coef_keys(supplied, annotations, label)
        if any(
            isinstance(supplied[a], (str, bytes, bool)) or not np.isscalar(supplied[a])
            for a in annotations
        ):
            raise ValueError(f"{label} values must be finite numeric values")
        try:
            column = np.asarray([supplied[a] for a in annotations], dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{label} values must be finite numeric values") from error
        if not np.isfinite(column).all():
            raise ValueError(f"{label} values must be finite")
        if np.any(column < 0):
            raise ValueError(f"{label} values must be nonnegative")
        columns.append(column)
    return np.column_stack(columns)


def _fit_random_effects(
    stats_or_stats,
    *,
    weights: Mapping[str, float] | None = None,
    n_jk_blocks: int | None = None,
    kernel_groups: Mapping[str, object] | None = None,
    init_gene_coef=None,
    fix_gene_coef: bool = False,
    max_iter: int = 50,
    tol: float = 1e-6,
    psd_burden_coef: bool = False,
):
    """Numerically fit prepared JK statistics through one point/JK lifecycle."""
    from kasm.suffstats import JKSuffStats

    supplied_mapping = isinstance(stats_or_stats, Mapping)
    models = dict(stats_or_stats) if supplied_mapping else {"model": stats_or_stats}
    if not models or not all(
        isinstance(value, JKSuffStats) for value in models.values()
    ):
        raise TypeError(
            "_fit_random_effects requires JKSuffStats or a nonempty mapping of them"
        )
    # The numerical lifecycle receives only its determined baseline or selected
    # blocks. Parent metadata/arrays are shared; siblings are never copied.
    models = {
        name: dataclass_replace(value, burden_extensions={}, covar_extensions={})
        if isinstance(value, JKSuffStats)
        and (value.burden_extensions or value.covar_extensions)
        else value
        for name, value in models.items()
    }
    # Public options and input geometry are validated by api.fit_random_effects.

    if supplied_mapping and init_gene_coef is not None:
        _require_gene_coef_keys(
            init_gene_coef, tuple(models), "init_gene_coef model mapping"
        )
    counts = {value.n_jk_blocks for value in models.values()}
    if len(counts) != 1:
        raise ValueError("all JKSuffStats must use the same n_jk_blocks")
    stored_blocks = next(iter(counts))
    n_jk_blocks = stored_blocks if n_jk_blocks is None else n_jk_blocks
    if stored_blocks != n_jk_blocks:
        raise ValueError("all JKSuffStats must use the requested n_jk_blocks")
    stats = models

    for name, stat in stats.items():
        if not stat.live_blocks.size:
            raise ValueError(f"exclude_genes removes every gene from {name!r}")

    if weights is None:
        normalized_weights = {name: 1.0 for name in models}
    else:
        if set(weights) != set(models):
            raise ValueError("weights keys must match the model labels")
        values = np.asarray([weights[name] for name in models], float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("weights must be positive and finite")
        values /= values.mean()
        normalized_weights = dict(zip(models, values, strict=True))

    parameter_names = {}
    kernel_group_names_by_model = {}
    starts = None if init_gene_coef is None else []
    inputs = []
    maf_bounds = {}
    for name, source in models.items():
        stat = stats[name]
        layout = source.layout
        if supplied_mapping:
            kernel_group = np.zeros(layout.n_kernels, dtype=np.intp)
            kernel_group_names = ()
        elif kernel_groups is not None:
            kernel_group_names = tuple(kernel_groups)
            kernel_group = np.full(layout.n_kernels, -1, dtype=np.intp)
            for kernel_group_index, label in enumerate(kernel_group_names):
                indices = np.asarray(tuple(kernel_groups[label]), dtype=np.intp)
                if np.any(indices < 0) or np.any(indices >= len(kernel_group)):
                    raise ValueError(
                        f"kernel group {label!r} contains an invalid index"
                    )
                if np.any(kernel_group[indices] >= 0):
                    raise ValueError("kernel groups overlap")
                kernel_group[indices] = kernel_group_index
        else:
            kernel_group = np.asarray(layout.kernel_groups, dtype=np.intp)
            kernel_group_names = (
                tuple(layout.kernel_group_names)
                if source.options.kernel_groups is not None or psd_burden_coef
                else ()
            )
        if np.any(kernel_group[stat.heritable] < 0):
            raise ValueError("kernel groups must cover exactly the heritable kernels")
        if kernel_groups is not None:
            kernel_group[stat.overdispersion] = 0
        kernel_group_names_by_model[name] = kernel_group_names

        kernel_coef_loc = []
        for index, kernel in enumerate(layout.kernel_ids):
            if supplied_mapping and kernel.kind == "B":
                key = (
                    "burden",
                    kernel.kind,
                    kernel.col,
                    kernel.col2,
                    kernel.maf_label,
                    kernel.maf_label2,
                )
                bins = source.options.maf_bins or {}
                bounds = tuple(
                    None if label is None else tuple(bins[label])
                    for label in (kernel.maf_label, kernel.maf_label2)
                )
                if key in maf_bounds and maf_bounds[key] != bounds:
                    raise ValueError(
                        f"shared burden parameter {key} has inconsistent MAF bounds"
                    )
                maf_bounds[key] = bounds
            elif (
                source.options.overdispersion_shared
                and index in layout.overdispersion_indices
            ):
                key = (name, "overdispersion", kernel.maf_label)
            else:
                key = (name, index)
            kernel_coef_loc.append(
                parameter_names.setdefault(key, len(parameter_names))
            )

        if starts is not None:
            starts.append(
                _prepare_gene_coef(
                    init_gene_coef[name] if supplied_mapping else init_gene_coef,
                    tuple(stat.gene_annot_names),
                    kernel_group_names,
                    f"init_gene_coef[{name!r}]"
                    if supplied_mapping
                    else "init_gene_coef",
                )
            )
        basis = patterns = None
        if psd_burden_coef:
            basis = _burden_psd_constraint_basis(source.layout)
            patterns = np.ones((1, len(stat.gene_annot_names)))
        inputs.append(
            fitting.RandomEffectsInput(
                totals=stat.totals,
                kernel_group=kernel_group,
                overdispersion=np.asarray(stat.overdispersion, dtype=np.intp),
                sample_size=float(stat.sample_size),
                kernel_coef_loc=np.asarray(kernel_coef_loc, dtype=np.intp),
                weight=normalized_weights[name],
                psd_kernel_basis=basis,
                psd_annotation_patterns=patterns,
            )
        )

    live = [set(stats[name].live_blocks.tolist()) for name in models]
    missing = set(range(n_jk_blocks)) - set().union(*live)
    if (
        not supplied_mapping
        and kernel_group_names_by_model["model"]
        and not isinstance(stats["model"].excluded_genes, dict)
    ):
        # Grouped fits can have empty blocks on their retained support.
        missing = set()
    if missing:
        raise ValueError(
            f"every jackknife block must contain a retained gene; empty blocks: {sorted(missing)}"
        )

    def fit_once(block, start):
        fit_inputs = (
            inputs
            if block is None
            else [
                dataclass_replace(
                    item,
                    totals={
                        key: total - stats[name].blocks[key][block]
                        for key, total in item.totals.items()
                    },
                )
                for name, item in zip(stats, inputs, strict=True)
            ]
        )
        try:
            return fitting.alternating_fit(
                fit_inputs,
                init_gene_coef=start,
                fix_gene_coef=fix_gene_coef,
                max_iter=max_iter,
                tol=tol,
            )
        except np.linalg.LinAlgError as error:
            context = "point" if block is None else f"JK block {block}"
            raise np.linalg.LinAlgError(f"random fit {context}: {error}") from error

    started = time.monotonic()
    logger.info(
        "event=fit_start method=random models=%d jk_blocks=%d", len(stats), n_jk_blocks
    )
    point, replicates = _fit_point_and_deletes(
        fit_once,
        range(n_jk_blocks),
        point_start=starts,
        deleted_start=lambda fitted: [fit["gene_coef"] for fit in fitted],
        started=started,
    )
    logger.info(
        "event=fit_complete method=random iterations=%d converged=%s elapsed_seconds=%.3f",
        point[0]["n_iter"],
        point[0]["converged"],
        time.monotonic() - started,
    )

    output = {}
    for i, (name, source) in enumerate(models.items()):
        stat, fit = stats[name], point[i]
        kernel_coef_jk = np.asarray([value[i]["theta"] for value in replicates])
        gene_coef_jk = np.asarray([value[i]["gene_coef"] for value in replicates])
        variance = fit["theta"] * fit["trace"]
        variance_jk = np.asarray(
            [value[i]["theta"] * value[i]["trace"] for value in replicates]
        )
        heritable = np.array([kid.is_heritable for kid in source.layout.kernel_ids])
        hsq_jk = variance_jk[:, heritable].sum(axis=1)
        annotation_results = {}
        if not supplied_mapping and not kernel_group_names_by_model[name]:
            annot_hsq = _gene_annot_hsq(
                fit["theta"], fit["gene_coef"][:, 0], stat.totals["tr1"], heritable
            )
            annot_jk = np.asarray(
                [
                    _gene_annot_hsq(
                        value[i]["theta"],
                        value[i]["gene_coef"][:, 0],
                        stat.totals["tr1"]
                        if n_jk_blocks == 1
                        else stat.totals["tr1"] - stat.blocks["tr1"][block],
                        heritable,
                    )
                    for block, value in enumerate(replicates)
                ]
            )
            if not np.isclose(
                annot_hsq.sum(), variance[heritable].sum(), rtol=1e-9, atol=1e-11
            ) or not np.allclose(annot_jk.sum(axis=1), hsq_jk, rtol=1e-9, atol=1e-11):
                raise RuntimeError("gene-annotation components do not recover hsq")
            annotation_results = {
                "gene_annot_hsq": annot_hsq,
                "gene_annot_hsq_jk": annot_jk,
                "gene_annot_hsq_se": fitting.jackknife_se(annot_jk),
                "n_genes_per_annot": pl.DataFrame(
                    {
                        "annotation": stat.gene_annot_names,
                        "n_genes": stat.blocks["ga1"].sum(axis=0).astype(np.int64),
                    }
                ),
            }
        output[name] = RandomFit(
            kernel_coef=fit["theta"],
            gene_coef=fit["gene_coef"],
            kernel_groups=inputs[i].kernel_group,
            residual=fit["residual"],
            sample_size=float(stat.sample_size),
            kernel_labels=stat.kernel_labels,
            gene_annot_cols=stat.gene_annot_names,
            heritable_labels=tuple(np.asarray(stat.kernel_labels)[heritable]),
            kernel_options=source.kernel_options,
            variance=variance,
            kernel_coef_jk=kernel_coef_jk,
            gene_coef_jk=gene_coef_jk,
            variance_jk=variance_jk,
            hsq=float(variance[heritable].sum()),
            hsq_jk=hsq_jk,
            hsq_se=float(fitting.jackknife_se(hsq_jk[:, None])[0]),
            variance_se=np.full(len(variance), np.nan)
            if supplied_mapping
            else fitting.jackknife_se(variance_jk),
            kernel_coef_se=np.full(len(variance), np.nan)
            if supplied_mapping
            else fitting.jackknife_se(kernel_coef_jk),
            n_jk_blocks=len(variance_jk),
            kernel_group_names=kernel_group_names_by_model[name],
            joint=supplied_mapping,
            n_genes=None if isinstance(stat.excluded_genes, dict) else stat.n_genes,
            n_variants=None
            if isinstance(stat.excluded_genes, dict)
            else int(np.asarray(stat.blocks["nv"]).sum()),
            n_iter=fit["n_iter"],
            converged=fit["converged"],
            n_iter_jk=np.asarray([value[i]["n_iter"] for value in replicates]),
            converged_jk=np.asarray([value[i]["converged"] for value in replicates]),
            minimum_eigenvalues=fit["psd_min_eigenvalues"],
            excluded_genes=stat.excluded_genes,
            gene_blocks=stat.gene_blocks,
            weight=inputs[i].weight,
        )
        if annotation_results:
            output[name] = dataclass_replace(output[name], **annotation_results)
    return output


def _assemble_random_reports(result, model, stats=None, *, include_maf=True):
    """Finish geometry-dependent reports before releasing the training model."""
    from kasm.api import (
        _maf_trace_data,
        _random_gene_annot_components,
        _random_maf_components,
    )

    from kasm.suffstats import _get_gene_group_mask

    updates = {}
    if result.kernel_group_names:
        gene_group_mask = _get_gene_group_mask(
            model.genes,
            result.excluded_genes,
            result.kernel_group_names,
            result.kernel_options.get("covariance_between_kernel_groups"),
        )
        counts = np.zeros((len(result.kernel_group_names), 2), dtype=np.int64)
        result_position = {
            label: position for position, label in enumerate(result.kernel_labels)
        }
        columns = [
            {
                column
                for kernel in model.layout.kernel_ids
                if kernel.is_heritable
                and result.kernel_groups[result_position[kernel.label]] == position
                for column in (
                    kernel.col,
                    kernel.col2,
                )
                if column is not None
            }
            for position in range(len(result.kernel_group_names))
        ]
        for gene_index, gene in enumerate(model.genes):
            for position in range(len(result.kernel_group_names)):
                if not gene_group_mask[gene_index, position]:
                    continue
                selected = np.column_stack(
                    [
                        model.dataset.variant_values(column, gene) != 0
                        for column in columns[position]
                    ]
                ).any(axis=1)
                counts[position] += (int(selected.any()), int(selected.sum()))
        updates["kernel_group_counts"] = counts
    if include_maf and model.options.maf_col is not None and model.options.maf_bins:
        trace_data = _maf_trace_data(model, stats)
        if not result.kernel_group_names:
            updates["_maf_components"] = _random_maf_components(
                model,
                result,
                result.excluded_genes,
                result.hsq,
                expected_hsq_jk=result.hsq_jk,
                trace_data=trace_data,
                independent_trace=stats.maf_independent_trace,
            )
        else:
            updates["_maf_components"] = pl.concat(
                [
                    _random_maf_components(
                        model,
                        result,
                        tuple(np.asarray(model.genes)[~gene_group_mask[:, position]]),
                        float(
                            result.variance[
                                (result.kernel_groups == position)
                                & np.isin(result.kernel_labels, result.heritable_labels)
                            ].sum()
                        ),
                        target_kernel_group=position,
                        expected_hsq_jk=result.variance_jk[
                            :,
                            (result.kernel_groups == position)
                            & np.isin(result.kernel_labels, result.heritable_labels),
                        ].sum(axis=1),
                        trace_data=trace_data,
                        independent_trace=stats.maf_independent_trace,
                    ).with_columns(pl.lit(name).alias("kernel_group"))
                    for position, name in enumerate(result.kernel_group_names)
                ]
            ).select("kernel_group", "maf_bin", "hsq", "hsq_se", "hsq_jk")
    if result.gene_annot_hsq is not None:
        selected = [gene not in result.excluded_genes for gene in model.genes]
        annotations = model.gene_annot_matrix()[selected]
        if np.isin(annotations, (0.0, 1.0)).all() and np.all(
            annotations.sum(axis=1) == 1
        ):
            updates["_gene_annot_components"] = _random_gene_annot_components(
                model, result, result.excluded_genes
            )
    return dataclass_replace(result, **updates)
