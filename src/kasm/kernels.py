"""Kernel configuration and enumeration.

``KernelOptions`` is the declarative description a caller writes;
``KernelLayout`` is the ordered kernel list it expands to, plus the index groups
the numerical reducers need. Nothing here touches data — enumeration depends
only on the configuration, so two datasets sharing a configuration share a
kernel order.

Four kinds of kernel, identified by ``KernelId.kind``:

``B``
    Burden. Rank one per gene, ``K = u u'`` for an annotation column ``u``
    optionally masked to a MAF bin. A ``B`` with ``col2`` set is a *cross*
    kernel, ``u_i u_j' + u_j u_i'``, which carries the covariance between two
    burden components rather than a variance of its own.
``I``
    Independent. ``K = diag(w)``: unstructured per-variant effects, with
    ``w = 1`` for the global identity kernel or a named nonnegative variant
    column for a class-specific independent kernel. Heritable.
``O``
    Overdispersion. ``K = diag(w)`` for an annotation column ``w``. Not
    heritable — it absorbs residual per-variant variance.
``C``
    Covar extension. Its statistics are owned by SuffStats outside the base
    Model layout. Selecting a covar extension appends ``C`` to the fitting
    layout of the selected JKSuffStats.

``I`` and ``O`` are both diagonal and share one computational path, so the
layout groups them as ``diagonal_indices``; ``I`` is simply the case
``w = 1``. They differ only in whether they count toward hsq.
"""

from __future__ import annotations

import functools
import inspect
import itertools
from collections import Counter
from dataclasses import dataclass, field

HERITABLE_KINDS = frozenset({"B", "I", "C"})


@dataclass(frozen=True)
class KernelId:
    """Structured kernel identifier.

    Parameters
    ----------
    kind : {'B', 'I', 'O', 'C'}
    col, maf_label : str or None
        Annotation column and MAF bin name.
    col2, maf_label2 : str or None
        The second component of a cross-burden kernel.
    """

    kind: str
    col: str | None = None
    maf_label: str | None = None
    col2: str | None = None
    maf_label2: str | None = None

    def __post_init__(self):
        if self.kind == "I" and self.col is None:
            raise ValueError("independent kernels require an annotation column")
        # Canonicalize cross kernels so (a, b) and (b, a) are one identity.
        if self.kind == "B" and self.col2 is not None:
            first = (self.col or "", self.maf_label or "")
            second = (self.col2, self.maf_label2 or "")
            if second < first:
                object.__setattr__(self, "col", second[0])
                object.__setattr__(self, "maf_label", second[1] or None)
                object.__setattr__(self, "col2", first[0])
                object.__setattr__(self, "maf_label2", first[1] or None)

    @property
    def is_heritable(self) -> bool:
        return self.kind in HERITABLE_KINDS

    @property
    def is_cross(self) -> bool:
        return self.kind == "B" and self.col2 is not None

    @property
    def is_diagonal(self) -> bool:
        return self.kind in ("I", "O")

    @property
    def label(self) -> str:
        """Readable name, unique within a layout."""
        if self.kind == "I":
            return f"I:{self.col}"
        name = _stratified(self.col, self.maf_label)
        if self.is_cross:
            return f"{self.kind}:{name}*{_stratified(self.col2, self.maf_label2)}"
        return f"{self.kind}:{name}"

    def __repr__(self) -> str:
        return f"KernelId({self.label})"


def _stratified(col: str | None, maf_label: str | None) -> str:
    return col if maf_label is None else f"{col}[{maf_label}]"


@dataclass(frozen=True)
class KernelLayout:
    """Ordered kernels and the index groups the reducers index by.

    Base enumeration appends overdispersion kernels last; selecting a covar
    extension appends its heritable kernel after them. Use the explicit index
    groups to select kernels. Every index refers to a position in ``kernel_ids``
    except ``cross_pairs``, which refers to positions in ``burden_labels``.

    Two label vocabularies, deliberately: ``KernelId.label`` names a *kernel*
    and is prefixed by kind (``B:``, ``I:``, ``O:``, or ``C:``), while
    ``burden_labels`` name *design-matrix columns* and carry only the
    annotation and MAF bin. Do not derive one from the other by string surgery.
    """

    kernel_ids: tuple[KernelId, ...]
    burden_labels: tuple[str, ...]
    burden_indices: tuple[int, ...]
    cross_indices: tuple[int, ...]
    cross_pairs: tuple[tuple[int, int], ...]
    independent_indices: tuple[int, ...]
    overdispersion_indices: tuple[int, ...]
    kernel_group_names: tuple[str, ...]
    kernel_groups: tuple[int, ...]

    @property
    def diagonal_indices(self) -> tuple[int, ...]:
        """Every diagonal-matrix kernel: the shared reducer path.

        Heritable independent kernels followed by the overdispersion tail —
        the columns of ``diag_kernel_weights``.
        """
        return self.independent_indices + self.overdispersion_indices

    @property
    def n_kernels(self) -> int:
        return len(self.kernel_ids)

    @property
    def n_burden(self) -> int:
        """Expanded burden columns — the width of the burden design matrix."""
        return len(self.burden_labels)

    @property
    def n_diagonal(self) -> int:
        return len(self.diagonal_indices)

    @property
    def independent_index(self) -> int | None:
        """The sole independent kernel, or ``None`` when there is not one."""
        if len(self.independent_indices) > 1:
            raise ValueError("layout has multiple independent kernels")
        return self.independent_indices[0] if self.independent_indices else None

    @property
    def n_independent(self) -> int:
        return len(self.independent_indices)

    @property
    def n_overdispersion(self) -> int:
        return len(self.overdispersion_indices)

    @property
    def heritable_indices(self) -> tuple[int, ...]:
        return tuple(i for i, kid in enumerate(self.kernel_ids) if kid.is_heritable)

    def index_of(self, kernel_id: KernelId) -> int:
        return self.kernel_ids.index(kernel_id)


@dataclass(frozen=True)
class KernelOptions:
    """Declarative kernel configuration.

    Parameters
    ----------
    maf_col : str or None
        Variant column holding minor allele frequency.
    maf_bins : dict or None
        Named half-open bins ``{name: (low, high)}``, tested as
        ``low <= maf < high``. Must not overlap. Variants outside every bin
        contribute to no MAF-stratified kernel.
    gene_annot_cols : list[str] or None
        Binary, mutually exclusive gene-level annotation columns that scale
        each gene's contribution. A gene may belong to at most one annotation.
        An empty list means the same as ``None``: no gene-level annotations, so
        every gene is scaled alike.
    burden_cols : list[str] or None
        Annotation columns given rank-one burden kernels.
    kernel_groups : dict or None
        Named kernel groups. Each value contains ``burden_cols`` and optional
        ``independent_col``, ``overdispersion_cols``, and ``extra_covar_kernel_fun``. A
        group's extra covariance kernel names are qualified as ``group:name``.
        This form cannot be combined with top-level burden or independent
        options.
    covariance_between_kernel_groups : dict or None
        Named pairs of base groups whose expanded burden columns receive the
        complete cross-covariance basis. Independent kernels are never crossed.
    burden_by_maf : bool
        Cross each burden column with the MAF bins. Ignored without
        ``maf_bins``.
    within_gene_group_col : str or None
        Variant column partitioning each gene's burden into independent groups,
        typically exons. The public kernel layout is unchanged: each logical
        burden kernel sums its group-specific rank-one factors.
    independent_col : str or None
        Variant column weighting one heritable diagonal kernel. Use an explicit
        all-ones column for ordinary unstructured per-variant effects.
    overdispersion_cols : list[str] or None
        Explicit non-heritable diagonal columns. For grouped models, declare
        these inside each kernel group, independently of independent_col.
    overdispersion_shared : bool
        Share overdispersion coefficients across columns within each MAF bin.
    overdispersion_by_maf : bool
        Cross each overdispersion column with the MAF bins.
    variant_variance_from_maf : bool
        Scale variants by ``d = sqrt(2 p (1 - p))`` instead of
        ``d = 1 / (se sqrt(N))``.
    """

    maf_col: str | None = None
    maf_bins: dict[str, tuple[float, float]] | None = None
    gene_annot_cols: list[str] | None = None
    burden_cols: list[str] | None = None
    kernel_groups: dict[str, dict[str, object]] | None = None
    covariance_between_kernel_groups: dict[str, tuple[str, str]] | None = None
    burden_by_maf: bool = True
    within_gene_group_col: str | None = None
    independent_col: str | None = None
    overdispersion_cols: list[str] | None = None
    overdispersion_shared: bool = False
    overdispersion_by_maf: bool = False
    variant_variance_from_maf: bool = False
    layout: KernelLayout = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.overdispersion_shared) is not bool:
            raise TypeError("overdispersion_shared must be Boolean")
        if self.maf_bins is not None:
            if self.maf_col is None:
                raise ValueError("maf_bins requires maf_col")

        # Take our own copies. The layout is frozen here, so a caller who later
        # mutates the dict or list they passed would otherwise leave
        # `to_dict()` describing a configuration that was never used.
        if self.maf_bins is not None:
            object.__setattr__(
                self,
                "maf_bins",
                _prepare_maf_bins(self.maf_bins),
            )
        for name in (
            "gene_annot_cols",
            "burden_cols",
            "overdispersion_cols",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        if self.kernel_groups is not None:
            if self.burden_cols or self.independent_col is not None:
                raise ValueError(
                    "kernel_groups cannot be combined with top-level burden or "
                    "independent options"
                )
            if not self.kernel_groups:
                raise ValueError("kernel_groups must not be empty")
            kernel_groups = {}
            burden_cols = []
            group_independent_columns = []
            for kernel_group, raw in self.kernel_groups.items():
                if not isinstance(kernel_group, str) or not kernel_group:
                    raise ValueError("kernel group names must be nonempty strings")
                if not isinstance(raw, dict):
                    raise TypeError(f"kernel group {kernel_group!r} must be a dict")
                unexpected = set(raw) - {
                    "burden_cols",
                    "independent_col",
                    "overdispersion_cols",
                    "extra_covar_kernel_fun",
                    "extra_covar_with_indep",
                }
                if unexpected:
                    raise ValueError(
                        f"kernel group {kernel_group!r} has unknown options "
                        f"{sorted(unexpected)}"
                    )
                columns = tuple(raw.get("burden_cols") or ())
                independent_col = raw.get("independent_col")
                extra_covar_kernel_fun = raw.get("extra_covar_kernel_fun")
                with_indep = raw.get("extra_covar_with_indep", False)
                if type(with_indep) is not bool:
                    raise TypeError("extra_covar_with_indep must be Boolean")
                if with_indep and extra_covar_kernel_fun is None:
                    raise ValueError(
                        "extra_covar_with_indep requires extra_covar_kernel_fun"
                    )
                if independent_col is not None and not isinstance(independent_col, str):
                    raise TypeError(
                        f"kernel group {kernel_group!r} independent_col must be a string"
                    )
                if extra_covar_kernel_fun is not None:
                    if independent_col is None:
                        raise ValueError(
                            f"kernel group {kernel_group!r} requires independent_col when "
                            "extra_covar_kernel_fun is configured"
                        )
                    _validate_extra_covar_kernel_fun(extra_covar_kernel_fun)
                if not columns:
                    raise ValueError(
                        f"kernel group {kernel_group!r} requires at least one burden column"
                    )
                kernel_groups[kernel_group] = {
                    "burden_cols": columns,
                    "independent_col": independent_col,
                    "overdispersion_cols": tuple(raw.get("overdispersion_cols") or ()),
                    "extra_covar_kernel_fun": extra_covar_kernel_fun,
                    "extra_covar_with_indep": with_indep,
                }
                burden_cols.extend(columns)
                if independent_col is not None:
                    group_independent_columns.append(independent_col)
            if len(set(burden_cols)) != len(burden_cols):
                raise ValueError("burden columns must be unique across kernel groups")
            if len(set(group_independent_columns)) != len(group_independent_columns):
                raise ValueError(
                    "independent columns must be unique across kernel groups"
                )
            if self.overdispersion_cols:
                raise ValueError("put overdispersion_cols inside each kernel group")
            covariance = dict(self.covariance_between_kernel_groups or {})
            seen_pairs = set()
            for derived, parents in covariance.items():
                if not isinstance(derived, str) or not derived:
                    raise ValueError("covariance group names must be nonempty strings")
                if derived in kernel_groups:
                    raise ValueError(
                        f"covariance group {derived!r} duplicates a base group"
                    )
                if not isinstance(parents, (tuple, list)) or len(parents) != 2:
                    raise ValueError(
                        f"covariance group {derived!r} must name exactly two parents"
                    )
                left, right = parents
                if (
                    left == right
                    or left not in kernel_groups
                    or right not in kernel_groups
                ):
                    raise ValueError(
                        f"covariance group {derived!r} has invalid parents {parents!r}"
                    )
                pair = frozenset((left, right))
                if pair in seen_pairs:
                    raise ValueError("a pair of kernel groups may be crossed only once")
                seen_pairs.add(pair)
                covariance[derived] = (left, right)
            object.__setattr__(self, "kernel_groups", kernel_groups)
            object.__setattr__(self, "covariance_between_kernel_groups", covariance)
            object.__setattr__(self, "burden_cols", tuple(burden_cols))
        elif self.covariance_between_kernel_groups:
            raise ValueError("covariance_between_kernel_groups requires kernel_groups")
        elif self.independent_col is not None and not isinstance(
            self.independent_col, str
        ):
            raise TypeError("independent_col must be a string")
        if self.variant_variance_from_maf and self.maf_col is None:
            raise ValueError("variant_variance_from_maf requires maf_col")
        if not self.burden_cols and self.independent_col is None:
            raise ValueError("no heritable kernels: set burden_cols or independent_col")
        object.__setattr__(self, "layout", self._build_layout())

    @property
    def maf_labels(self) -> tuple[str, ...]:
        return tuple(self.maf_bins) if self.maf_bins else ()

    @property
    def independent_cols(self) -> tuple[str, ...]:
        """Independent columns from the top-level option or each kernel group."""
        if self.kernel_groups is None:
            return () if self.independent_col is None else (self.independent_col,)
        return tuple(
            options["independent_col"]
            for options in self.kernel_groups.values()
            if options["independent_col"] is not None
        )

    def extra_covar_kernel_functions(self) -> tuple[tuple[str, object], ...]:
        """Return configured kernel-group covariance builders."""
        return tuple(
            (kernel_group, options["extra_covar_kernel_fun"])
            for kernel_group, options in (self.kernel_groups or {}).items()
            if options["extra_covar_kernel_fun"] is not None
        )

    def _build_layout(self) -> KernelLayout:
        kernel_ids: list[KernelId] = []
        burden_labels: list[str] = []
        burden_indices: list[int] = []
        if self.kernel_groups is None:
            kernel_group_names = ("all",)
            burden_kernel_group = {col: "all" for col in self.burden_cols or ()}
            independent_group = {col: "all" for col in self.independent_cols}
            covariance_groups = {}
        else:
            kernel_group_names = (
                *self.kernel_groups,
                *(self.covariance_between_kernel_groups or {}),
            )
            burden_kernel_group = {
                col: kernel_group
                for kernel_group, options in self.kernel_groups.items()
                for col in options["burden_cols"]
            }
            independent_group = {
                options["independent_col"]: kernel_group
                for kernel_group, options in self.kernel_groups.items()
                if options["independent_col"] is not None
            }
            covariance_groups = {
                frozenset(parents): derived
                for derived, parents in (
                    self.covariance_between_kernel_groups or {}
                ).items()
            }
        kernel_group_index = {
            name: index for index, name in enumerate(kernel_group_names)
        }
        kernel_groups: list[int] = []
        burden_kernel_groups: list[str] = []

        burden_strata = (
            self.maf_labels if self.burden_by_maf and self.maf_bins else (None,)
        )
        for col in self.burden_cols or ():
            for maf_label in burden_strata:
                burden_labels.append(_stratified(col, maf_label))
                burden_kernel_groups.append(burden_kernel_group[col])
                burden_indices.append(len(kernel_ids))
                kernel_ids.append(KernelId("B", col, maf_label))
                kernel_groups.append(kernel_group_index[burden_kernel_group[col]])

        # Within-group burden covariance is automatic. Between-group covariance
        # is present only when the pair is explicitly named.
        cross_indices: list[int] = []
        cross_pairs: list[tuple[int, int]] = []
        for i, j in itertools.combinations(range(len(burden_labels)), 2):
            left_kernel_group, right_kernel_group = (
                burden_kernel_groups[i],
                burden_kernel_groups[j],
            )
            cross_group = (
                left_kernel_group
                if left_kernel_group == right_kernel_group
                else covariance_groups.get(
                    frozenset((left_kernel_group, right_kernel_group))
                )
            )
            if cross_group is None:
                continue
            left, right = kernel_ids[burden_indices[i]], kernel_ids[burden_indices[j]]
            cross_indices.append(len(kernel_ids))
            cross_pairs.append((i, j))
            kernel_ids.append(
                KernelId("B", left.col, left.maf_label, right.col, right.maf_label)
            )
            kernel_groups.append(kernel_group_index[cross_group])

        independent_indices: list[int] = []
        for col in self.independent_cols:
            independent_indices.append(len(kernel_ids))
            kernel_ids.append(KernelId("I", col))
            kernel_groups.append(kernel_group_index[independent_group[col]])

        # Overdispersion last, so a hsq fit can truncate the tail.
        overdispersion_indices: list[int] = []
        ovd_strata = (
            self.maf_labels if self.overdispersion_by_maf and self.maf_bins else (None,)
        )
        overdispersion_groups = (
            self.kernel_groups
            if self.kernel_groups is not None
            else {"all": {"overdispersion_cols": self.overdispersion_cols or ()}}
        )
        for name, options in overdispersion_groups.items():
            for col in options["overdispersion_cols"]:
                for maf_label in ovd_strata:
                    overdispersion_indices.append(len(kernel_ids))
                    kernel_ids.append(KernelId("O", col, maf_label))
                    kernel_groups.append(kernel_group_index[name])

        duplicates = {kid for kid, count in Counter(kernel_ids).items() if count > 1}
        if duplicates:
            raise ValueError(
                f"duplicate kernels: {sorted(k.label for k in duplicates)}"
            )

        return KernelLayout(
            kernel_ids=tuple(kernel_ids),
            burden_labels=tuple(burden_labels),
            burden_indices=tuple(burden_indices),
            cross_indices=tuple(cross_indices),
            cross_pairs=tuple(cross_pairs),
            independent_indices=tuple(independent_indices),
            overdispersion_indices=tuple(overdispersion_indices),
            kernel_group_names=tuple(kernel_group_names),
            kernel_groups=tuple(kernel_groups),
        )

    @classmethod
    def from_recorded(cls, options):
        """Reconstruct the base design without restoring covar callbacks."""
        recorded = dict(options)
        if recorded.get("kernel_groups"):
            recorded["kernel_groups"] = {
                name: {
                    key: value
                    for key, value in group.items()
                    if key not in ("extra_covar_kernel_fun", "extra_covar_with_indep")
                }
                for name, group in recorded["kernel_groups"].items()
            }
        return cls(**recorded)

    def to_dict(self) -> dict:
        """JSON-serializable kernel configuration.

        A callable is recorded by module and qualified name, never pickled.
        """
        return {
            "maf_col": self.maf_col,
            "maf_bins": (
                None
                if self.maf_bins is None
                else {name: list(bounds) for name, bounds in self.maf_bins.items()}
            ),
            "gene_annot_cols": list(self.gene_annot_cols or ()),
            "burden_cols": (
                None if self.kernel_groups is not None else list(self.burden_cols or ())
            ),
            "kernel_groups": (
                None
                if self.kernel_groups is None
                else {
                    kernel_group: {
                        "burden_cols": list(options["burden_cols"]),
                        "independent_col": options["independent_col"],
                        "overdispersion_cols": list(options["overdispersion_cols"]),
                        "extra_covar_with_indep": options["extra_covar_with_indep"],
                        "extra_covar_kernel_fun": _function_name(
                            options["extra_covar_kernel_fun"]
                        ),
                    }
                    for kernel_group, options in self.kernel_groups.items()
                }
            ),
            "covariance_between_kernel_groups": (
                None
                if self.covariance_between_kernel_groups is None
                else {
                    kernel_group: list(parents)
                    for kernel_group, parents in self.covariance_between_kernel_groups.items()
                }
            ),
            "burden_by_maf": self.burden_by_maf,
            "within_gene_group_col": self.within_gene_group_col,
            "independent_col": (
                None if self.kernel_groups is not None else self.independent_col
            ),
            "overdispersion_cols": list(self.overdispersion_cols or ()),
            "overdispersion_by_maf": self.overdispersion_by_maf,
            "overdispersion_shared": self.overdispersion_shared,
            "variant_variance_from_maf": self.variant_variance_from_maf,
        }


def _validate_extra_covar_kernel_fun(function) -> None:
    if isinstance(function, functools.partial):
        if function.args:
            raise ValueError(
                "an extra_covar_kernel_fun partial must bind keyword arguments only"
            )
        for keyword, value in function.keywords.items():
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise TypeError(
                    f"extra_covar_kernel_fun partial keyword {keyword!r} must be a "
                    "JSON scalar so it can be recorded"
                )
        _validate_extra_covar_kernel_fun(function.func)
        return
    if not callable(function):
        raise TypeError("extra_covar_kernel_fun must be callable")
    if (
        not inspect.isfunction(function)
        or function.__name__ == "<lambda>"
        or "<locals>" in function.__qualname__
    ):
        raise ValueError(
            "extra_covar_kernel_fun must be a named module-level function, or a "
            "functools.partial of one, so it can be recorded"
        )


def _function_name(function):
    if function is None:
        return None
    if isinstance(function, functools.partial):
        return {
            "function": _function_name(function.func),
            "kwargs": dict(function.keywords),
        }
    return f"{function.__module__}.{function.__qualname__}"


def _prepare_maf_bins(maf_bins: dict[str, tuple[float, float]]) -> dict:
    if not maf_bins:
        raise ValueError("maf_bins must not be empty")
    normalized = {}
    bounds = []
    for name, bound in maf_bins.items():
        if len(bound) != 2:
            raise ValueError(f"MAF bin {name!r} must be a (low, high) pair")
        low, high = float(bound[0]), float(bound[1])
        if not low < high:
            raise ValueError(f"MAF bin {name!r} has low >= high: {(low, high)}")
        normalized[name] = (low, high)
        bounds.append((low, high, name))
    bounds.sort()
    for lower, upper in itertools.pairwise(bounds):
        if upper[0] < lower[1]:
            raise ValueError(
                f"MAF bins {lower[2]!r} and {upper[2]!r} overlap: "
                f"{lower[:2]} and {upper[:2]}"
            )

    return normalized
