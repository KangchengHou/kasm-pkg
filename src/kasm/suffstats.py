"""Base sufficient statistics: the LD-dependent work, done once.

Every estimator in the package is a linear-in-moments fit against the same five
per-gene quantities, so they are computed once, stored, and reused. For kernels
``K_i`` and the scaled LD ``R_s = D R D``:

``trace[g, i]``
    ``tr(K_i R_s)``
``quadform[g, i]``
    ``b' K_i b`` for ``b = beta / (se^2 N)``
``pair_trace[g, i, j]``
    ``tr(K_i R_s K_j R_s)`` — the Gram matrix of the moment system
``burden_cov[g, p, q]``
    ``u_p' R_s u_q``, from which every burden and cross entry above follows
``burden_beta[g, p]``
    ``u_p' b``

Symmetric statistics are packed upper triangles, and every array is gene-major
so one gene's block is contiguous.

The reducer at the bottom of this module takes an ``LDM`` and NumPy arrays.
``compute_suffstats`` is the orchestration around it that knows about
``Model``; the arithmetic does not.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from kasm._logging import logger
from kasm.kernels import KernelLayout
from kasm.ldm import LDM

_LOG_EVERY = 1000


# ---------------------------------------------------------------------------
# Packed symmetric storage
# ---------------------------------------------------------------------------


def packed_size(n: int) -> int:
    """Length of a packed upper triangle, diagonal included."""
    return n * (n + 1) // 2


def pack_symmetric(matrix: np.ndarray) -> np.ndarray:
    """Upper triangle of a symmetric matrix, row-major."""
    rows, cols = np.triu_indices(matrix.shape[0])
    return matrix[rows, cols]


def unpack_symmetric(packed: np.ndarray, n: int) -> np.ndarray:
    """Rebuild a symmetric matrix from its packed upper triangle.

    Accepts a single packed vector or a stack of them, in which case the
    leading axis is preserved and the matrix axes come last.
    """
    rows, cols = np.triu_indices(n)
    out = np.zeros(packed.shape[:-1] + (n, n))
    out[..., rows, cols] = packed
    out[..., cols, rows] = packed
    return out


def _packed_positions(rows: np.ndarray, cols: np.ndarray, n: int) -> np.ndarray:
    """Positions of upper-triangle coordinates in row-major packed storage."""
    rows, cols = np.minimum(rows, cols), np.maximum(rows, cols)
    return rows * n - rows * (rows - 1) // 2 + cols - rows


def _extend_packed(
    base: np.ndarray,
    extension_rows: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    n: int,
) -> np.ndarray:
    """Scatter a packed old/old block and dense new rows into packed arrays."""
    result = np.zeros(
        (base.shape[0], packed_size(n)), dtype=np.result_type(base, extension_rows)
    )
    base_rows, base_cols = np.triu_indices(len(old))
    result[:, _packed_positions(old[base_rows], old[base_cols], n)] = base

    # Every new/old pair is unique. For new/new pairs retain only the row whose
    # extended index is the upper-triangle row, avoiding duplicate assignment.
    columns = np.arange(n)
    is_new = np.zeros(n, dtype=bool)
    is_new[new] = True
    rows = np.repeat(new, n)
    row_positions = np.repeat(np.arange(len(new)), n)
    columns = np.tile(columns, len(new))
    keep = ~is_new[columns] | (columns >= rows)
    result[:, _packed_positions(rows[keep], columns[keep], n)] = extension_rows[
        :, row_positions[keep], columns[keep]
    ]
    return result


# ---------------------------------------------------------------------------
# Kernel index groups, as plain integers for the reducer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelIndex:
    """Where each kernel writes in the output, as arrays rather than objects.

    This is the whole of what the reducer needs to know about a kernel
    configuration, which keeps ``Model`` out of the arithmetic.
    """

    n_kernels: int
    burden_indices: np.ndarray
    cross_indices: np.ndarray
    cross_left: np.ndarray
    cross_right: np.ndarray
    diag_indices: np.ndarray

    @classmethod
    def from_layout(cls, layout) -> KernelIndex:
        pairs = np.asarray(layout.cross_pairs, dtype=np.intp).reshape(-1, 2)
        return cls(
            n_kernels=layout.n_kernels,
            burden_indices=np.asarray(layout.burden_indices, dtype=np.intp),
            cross_indices=np.asarray(layout.cross_indices, dtype=np.intp),
            cross_left=pairs[:, 0],
            cross_right=pairs[:, 1],
            diag_indices=np.asarray(layout.diagonal_indices, dtype=np.intp),
        )


# ---------------------------------------------------------------------------
# The statistics
# ---------------------------------------------------------------------------


COVAR_PROJECTION_KEYS = (
    "numerator_weights",
    "denominator_weights",
)


def _select_covar_extension(extensions, name):
    if not isinstance(name, str):
        raise TypeError("extra_covar_kernel must be a string")
    if name not in extensions:
        raise KeyError(f"unknown covar kernel: {name!r}")
    return extensions[name]


@dataclass(frozen=True)
class ExtraEnrichCols:
    total_hsq_weights: np.ndarray
    annot_hsq_weights: np.ndarray
    annot_het: np.ndarray
    column: str
    burden_cov: np.ndarray | None = None
    burden_beta: np.ndarray | None = None


@dataclass(frozen=True)
class CovarPayload:
    kernel_group: str
    numerator_weights: np.ndarray
    denominator_weights: np.ndarray
    n_genes: np.ndarray
    n_pairs: np.ndarray


@dataclass(frozen=True)
class ExtSuffStats:
    """One named extension's added moments, actual layout, and typed readout.

    Per-gene moments have gene-leading axes. JK moments have block-leading
    axes and a final channel axis containing unweighted then annotation sums.
    Only rectangular added-kernel rows are retained, never sibling products.
    """

    trace: np.ndarray
    quadform: np.ndarray
    pair_trace: np.ndarray
    layout: KernelLayout
    base_to_extended: np.ndarray
    payload: ExtraEnrichCols | CovarPayload

    @property
    def new_indices(self):
        return np.setdiff1d(np.arange(self.layout.n_kernels), self.base_to_extended)

    @property
    def kernel_labels(self):
        return tuple(kernel.label for kernel in self.layout.kernel_ids)

    @property
    def burden_labels(self):
        return self.layout.burden_labels

    @property
    def blocks(self):
        return {
            key + suffix: values[..., 0] if suffix == "0" else values[..., 1:]
            for key, values in (
                ("pt", self.pair_trace),
                ("tr", self.trace),
                ("qf", self.quadform),
            )
            for suffix in ("0", "1")
        }


def _layout_from_dict(value):
    from kasm.kernels import KernelId

    return KernelLayout(
        **{
            key: tuple(KernelId(**item) for item in items)
            if key == "kernel_ids"
            else tuple(tuple(item) for item in items)
            if key == "cross_pairs"
            else tuple(items)
            for key, items in value.items()
        }
    )


def _validate_extensions(parent, *, aggregated):
    n_rows = parent.n_jk_blocks if aggregated else parent.n_genes
    n_base = parent.layout.n_kernels if aggregated else parent.n_kernels
    channels = (len(parent.gene_annot_names) + 1,) if aggregated else ()
    for namespace, payload_type in (
        ("burden", ExtraEnrichCols),
        ("covar", CovarPayload),
    ):
        for name, ext in getattr(parent, namespace + "_extensions").items():
            if not isinstance(ext, ExtSuffStats) or not isinstance(
                ext.payload, payload_type
            ):
                raise TypeError(
                    f"{namespace} extension {name!r} requires typed ExtSuffStats"
                )
            if not isinstance(ext.layout, KernelLayout):
                raise TypeError("extension requires an actual KernelLayout")
            if (
                len(ext.layout.kernel_groups) != ext.layout.n_kernels
                or len(set(ext.kernel_labels)) != ext.layout.n_kernels
            ):
                raise ValueError("invalid extension layout kernel identities/groups")
            old = np.asarray(ext.base_to_extended)
            if (
                old.shape != (n_base,)
                or old.dtype.kind not in "iu"
                or len(np.unique(old)) != n_base
                or np.any(old < 0)
                or np.any(old >= ext.layout.n_kernels)
            ):
                raise ValueError("invalid extension base-kernel mapping")
            if tuple(ext.kernel_labels[i] for i in old) != parent.kernel_labels:
                raise ValueError("extension layout does not match parent kernels")
            if aggregated and (
                ext.layout.kernel_group_names != parent.layout.kernel_group_names
                or tuple(ext.layout.kernel_groups[i] for i in old)
                != parent.layout.kernel_groups
            ):
                raise ValueError("extension kernel groups do not match parent layout")
            n_new = len(ext.new_indices)
            if not n_new:
                raise ValueError("extension must add at least one kernel")
            if namespace == "covar":
                kinds = tuple(ext.layout.kernel_ids[i].kind for i in ext.new_indices)
                if kinds not in (("C",), ("I", "C")) or not np.array_equal(
                    old, np.arange(n_base)
                ):
                    raise ValueError("covar extension layout must append [K] or [D,K]")
                group = ext.layout.kernel_group_names.index(ext.payload.kernel_group)
                if any(ext.layout.kernel_groups[i] != group for i in ext.new_indices):
                    raise ValueError(
                        "covar extension added kernels must share their owning group"
                    )
                if (
                    kinds == ("I", "C")
                    and ext.new_indices[0] not in ext.layout.independent_indices
                ):
                    raise ValueError(
                        "covar independent kernel must belong to the independent indices"
                    )
            expected = {
                "trace": (n_rows, n_new, *channels),
                "quadform": (n_rows, n_new, *channels),
                "pair_trace": (n_rows, n_new, ext.layout.n_kernels, *channels),
            }
            for key, shape in expected.items():
                value = getattr(ext, key)
                if np.shape(value) != shape or not np.isfinite(value).all():
                    raise ValueError(
                        f"invalid extension {name!r} {key}; expected finite {shape}"
                    )
            weights_shape = (n_rows, ext.layout.n_kernels) + (
                (len(parent.gene_annot_names),) if aggregated else ()
            )
            keys = (
                ("total_hsq_weights", "annot_hsq_weights")
                if namespace == "burden"
                else COVAR_PROJECTION_KEYS
            )
            for key in keys:
                value = getattr(ext.payload, key)
                if np.shape(value) != weights_shape or not np.isfinite(value).all():
                    raise ValueError(
                        f"invalid extension {name!r} {key}; expected finite {weights_shape}"
                    )
            counts = ("annot_het",) if namespace == "burden" else ("n_genes", "n_pairs")
            for key in counts:
                value = getattr(ext.payload, key)
                if (
                    np.shape(value) != (n_rows,)
                    or not np.isfinite(value).all()
                    or (namespace == "covar" and np.any(value < 0))
                ):
                    raise ValueError(f"invalid extension {name!r} {key}")
            if namespace == "burden":
                if ext.payload.column != name:
                    raise ValueError(
                        "burden extension column does not match its namespace key"
                    )
                if aggregated and (
                    ext.payload.burden_cov is not None
                    or ext.payload.burden_beta is not None
                ):
                    raise ValueError(
                        "JK extensions must not retain per-gene burden data"
                    )
                if not aggregated and (
                    ext.payload.burden_cov is None or ext.payload.burden_beta is None
                ):
                    raise ValueError("per-gene burden extensions require burden data")
            elif not aggregated and not np.isin(ext.payload.n_genes, (0, 1)).all():
                raise ValueError("per-gene covar eligibility must be Boolean")
            elif (
                not name.startswith(ext.payload.kernel_group + ":")
                or ext.payload.kernel_group not in ext.layout.kernel_group_names
            ):
                raise ValueError(
                    "covar extension names must be qualified by their kernel group"
                )


def _validate_maf(source, n_rows, *, n_annotations=None):
    """Check retained reporting tensors at the owning parent stage."""
    for name, ndim in (("maf_trace", 2), ("maf_independent_trace", 3)):
        value = getattr(source, name, None)
        if value is None:
            continue
        expected_ndim = ndim + (n_annotations is not None)
        if (
            value.ndim != expected_ndim
            or value.shape[0] != n_rows
            or (n_annotations is not None and value.shape[-1] != n_annotations)
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"invalid {name} reporting geometry or nonfinite moments")


def _save_maf(stored, source):
    for name in ("maf_trace", "maf_independent_trace"):
        value = getattr(source, name, None)
        stored["has_" + name] = np.array(value is not None)
        if value is not None:
            stored[name] = value


def _load_maf(stored):
    return {
        name: stored[name] if bool(stored["has_" + name]) else None
        for name in ("maf_trace", "maf_independent_trace")
    }


def _aggregate_maf(
    trace, layout, options, gene_kernel_mask, keep, annot, blocks, n_blocks
):
    from kasm.fitting import annotation_weighted_sums
    from kasm.kernels import KernelOptions

    if trace is None:
        return None

    def identity(kernel):
        return (
            kernel.kind,
            kernel.col,
            None if kernel.col2 == kernel.col else kernel.col2,
        )

    identities = [identity(kernel) for kernel in layout.kernel_ids]
    mapping = [
        identities.index(identity(kernel))
        for kernel in KernelOptions.from_recorded(
            {**options, "burden_by_maf": True}
        ).layout.kernel_ids
    ]
    return annotation_weighted_sums(
        (trace * gene_kernel_mask[:, mapping])[keep], annot, blocks, n_blocks
    )[1]


def _save_extensions(stored, parent):
    for namespace in ("burden", "covar"):
        extensions = getattr(parent, namespace + "_extensions")
        stored[namespace + "_extension_names"] = np.asarray(
            tuple(extensions), dtype=str
        )
        for position, ext in enumerate(extensions.values()):
            prefix = f"{namespace}_extension_{position}_"
            for key in ("trace", "quadform", "pair_trace", "base_to_extended"):
                stored[prefix + key] = getattr(ext, key)
            stored[prefix + "layout_json"] = np.array(json.dumps(asdict(ext.layout)))
            for key, value in vars(ext.payload).items():
                if value is not None:
                    stored[prefix + key] = np.asarray(value)


def _load_extensions(stored, namespace):
    extensions = {}
    for position, name in enumerate(stored[namespace + "_extension_names"].tolist()):
        prefix = f"{namespace}_extension_{position}_"
        if namespace == "burden":
            payload = ExtraEnrichCols(
                **{
                    key: stored[prefix + key]
                    for key in ("total_hsq_weights", "annot_hsq_weights", "annot_het")
                },
                column=str(stored[prefix + "column"]),
                burden_cov=stored.get(prefix + "burden_cov"),
                burden_beta=stored.get(prefix + "burden_beta"),
            )
        else:
            payload = CovarPayload(
                kernel_group=str(stored[prefix + "kernel_group"]),
                **{
                    key: stored[prefix + key]
                    for key in (*COVAR_PROJECTION_KEYS, "n_genes", "n_pairs")
                },
            )
        extensions[name] = ExtSuffStats(
            **{
                key: stored[prefix + key]
                for key in ("trace", "quadform", "pair_trace", "base_to_extended")
            },
            layout=_layout_from_dict(json.loads(str(stored[prefix + "layout_json"]))),
            payload=payload,
        )
    return extensions


@dataclass(frozen=True)
class SuffStats:
    """Per-gene base sufficient statistics, gene-major and packed.

    Attributes
    ----------
    pair_trace_upper : ndarray, shape (n_genes, packed_size(n_kernels))
    trace, quadform : ndarray, shape (n_genes, n_kernels)
    burden_cov_upper : ndarray, shape (n_genes, packed_size(n_burden))
    burden_beta : ndarray, shape (n_genes, n_burden)
    n_variants : ndarray, shape (n_genes,)
    genes, kernel_labels, burden_labels : tuple[str, ...]
    burden_extensions, covar_extensions : dict[str, ExtSuffStats]
        Separate optional burden and covar extensions computed with the shared base.
    """

    pair_trace_upper: np.ndarray
    trace: np.ndarray
    quadform: np.ndarray
    burden_cov_upper: np.ndarray
    burden_beta: np.ndarray
    n_variants: np.ndarray
    genes: tuple[str, ...]
    kernel_labels: tuple[str, ...]
    burden_labels: tuple[str, ...]
    total_het: np.ndarray
    burden_extensions: dict[str, ExtSuffStats] = field(default_factory=dict)
    covar_extensions: dict[str, ExtSuffStats] = field(default_factory=dict)
    maf_trace: np.ndarray | None = None
    maf_independent_trace: np.ndarray | None = None

    def __post_init__(self):
        if np.shape(self.total_het) != (self.n_genes,):
            raise ValueError("total heterozygosity must have one value per gene")
        _validate_maf(self, self.n_genes)
        _validate_extensions(self, aggregated=False)
        for annotation in self.burden_extensions:
            self._extension_indices(annotation)

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    @property
    def n_kernels(self) -> int:
        return len(self.kernel_labels)

    @property
    def n_burden(self) -> int:
        return len(self.burden_labels)

    def pair_trace(self, gene: str | None = None, rows=None) -> np.ndarray:
        """Dense ``tr(K_i R_s K_j R_s)``, for one gene or stacked over genes."""
        if gene is not None and rows is not None:
            raise ValueError("pass either gene or rows, not both")
        packed = self.pair_trace_upper
        if gene is not None:
            packed = packed[self.genes.index(gene)]
        elif rows is not None:
            packed = packed[rows]
        return unpack_symmetric(packed, self.n_kernels)

    def burden_cov(self, gene: str | None = None, rows=None) -> np.ndarray:
        """Dense ``u_p' R_s u_q``, for one gene or stacked over genes."""
        if gene is not None and rows is not None:
            raise ValueError("pass either gene or rows, not both")
        packed = self.burden_cov_upper
        if gene is not None:
            packed = packed[self.genes.index(gene)]
        elif rows is not None:
            packed = packed[rows]
        return unpack_symmetric(packed, self.n_burden)

    def _extension_indices(base, annotation):
        extension = base.burden_extensions[annotation]
        old = np.asarray(extension.base_to_extended, dtype=np.intp)
        new = extension.new_indices
        try:
            old_burden = np.asarray(
                [extension.burden_labels.index(label) for label in base.burden_labels],
                dtype=np.intp,
            )
        except ValueError as error:
            raise ValueError("invalid extension base-burden mapping") from error
        if len(np.unique(old_burden)) != len(old_burden):
            raise ValueError("invalid extension base-burden mapping")
        new_burden = np.setdiff1d(np.arange(len(extension.burden_labels)), old_burden)
        expected = {
            "burden_cov": (
                base.n_genes,
                len(new_burden),
                len(extension.burden_labels),
            ),
            "burden_beta": (base.n_genes, len(new_burden)),
        }
        for name, shape in expected.items():
            value = getattr(extension.payload, name)
            if np.shape(value) != shape:
                raise ValueError(f"invalid extension {name} shape: expected {shape}")
        return old, new, old_burden, new_burden

    def with_burden_extension(base, annotation: str) -> SuffStats:
        extension = base.burden_extensions[annotation]
        old, new, old_burden, new_burden = base._extension_indices(annotation)
        n_kernels = len(extension.kernel_labels)
        trace = np.zeros((base.n_genes, n_kernels))
        quadform = np.zeros_like(trace)
        trace[:, old], quadform[:, old] = base.trace, base.quadform
        trace[:, new], quadform[:, new] = extension.trace, extension.quadform

        n_burden = len(extension.burden_labels)
        burden_beta = np.zeros((base.n_genes, n_burden))
        burden_beta[:, old_burden] = base.burden_beta
        burden_beta[:, new_burden] = extension.payload.burden_beta
        return SuffStats(
            pair_trace_upper=_extend_packed(
                base.pair_trace_upper, extension.pair_trace, old, new, n_kernels
            ),
            trace=trace,
            quadform=quadform,
            burden_cov_upper=_extend_packed(
                base.burden_cov_upper,
                extension.payload.burden_cov,
                old_burden,
                new_burden,
                n_burden,
            ),
            burden_beta=burden_beta,
            total_het=base.total_het,
            n_variants=base.n_variants,
            genes=base.genes,
            kernel_labels=extension.kernel_labels,
            burden_labels=extension.burden_labels,
        )

    def save(self, path) -> None:
        """Write the base and all compact extensions to one compressed ``.npz``."""
        payload = dict(
            pair_trace_upper=self.pair_trace_upper,
            trace=self.trace,
            quadform=self.quadform,
            burden_cov_upper=self.burden_cov_upper,
            burden_beta=self.burden_beta,
            total_het=self.total_het,
            n_variants=self.n_variants,
            genes=np.array(self.genes, dtype=str),
            kernel_labels=np.array(self.kernel_labels, dtype=str),
            burden_labels=np.array(self.burden_labels, dtype=str),
        )
        _save_maf(payload, self)
        _save_extensions(payload, self)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path) -> SuffStats:
        """Read the base and all compact extensions from one ``.npz``."""
        with np.load(path, allow_pickle=False) as stored:
            return cls(
                pair_trace_upper=stored["pair_trace_upper"],
                trace=stored["trace"],
                quadform=stored["quadform"],
                burden_cov_upper=stored["burden_cov_upper"],
                burden_beta=stored["burden_beta"],
                total_het=stored["total_het"],
                n_variants=stored["n_variants"],
                genes=tuple(stored["genes"].tolist()),
                kernel_labels=tuple(stored["kernel_labels"].tolist()),
                burden_labels=tuple(stored["burden_labels"].tolist()),
                burden_extensions=_load_extensions(stored, "burden"),
                covar_extensions=_load_extensions(stored, "covar"),
                **_load_maf(stored),
            )


@dataclass(frozen=True)
class JKSuffStats:
    """Baseline and all prepared extensions, aggregated over aligned gene blocks.

    The actual layout owns kernel identities and index groups. Generation
    configuration is retained for scoring, not for reconstructing the layout.
    Extension moments and readout payloads are owned by this parent.
    """

    blocks: dict[str, np.ndarray]
    genes: tuple[str, ...]
    layout: KernelLayout
    excluded_genes: tuple[str, ...] | dict[str, tuple[str, ...]]
    n_jk_blocks: int
    gene_blocks: np.ndarray
    total_het: np.ndarray
    sample_size: float
    gene_annot_names: tuple[str, ...]
    kernel_options: dict
    burden_extensions: dict[str, ExtSuffStats] = field(default_factory=dict)
    covar_extensions: dict[str, ExtSuffStats] = field(default_factory=dict)
    maf_trace: np.ndarray | None = None
    maf_independent_trace: np.ndarray | None = None

    def __post_init__(self):
        from kasm import fitting
        from kasm.kernels import KernelLayout

        if not isinstance(self.layout, KernelLayout):
            raise TypeError("jackknife statistics require an actual KernelLayout")
        if self.n_jk_blocks < 1:
            raise ValueError("n_jk_blocks must be at least 1")
        if set(self.blocks) != set(fitting.ANNOTATION_BLOCK_KEYS):
            raise ValueError("jackknife block keys do not match the current format")
        expected = fitting.empty_annotation_blocks(
            0, self.layout.n_kernels, len(self.gene_annot_names)
        )
        if any(
            value.shape != (self.n_jk_blocks, *expected[key].shape[1:])
            for key, value in self.blocks.items()
        ):
            raise ValueError(
                "jackknife block arrays do not match layout and annotations"
            )
        gene_blocks = np.asarray(self.gene_blocks, dtype=np.intp)
        if gene_blocks.shape != (len(self.genes),):
            raise ValueError("gene_blocks must contain one label per gene")
        if (gene_blocks < 0).any() or (gene_blocks >= self.n_jk_blocks).any():
            raise ValueError("gene_blocks labels are outside the requested block range")
        object.__setattr__(self, "gene_blocks", gene_blocks)
        if isinstance(self.excluded_genes, dict):
            object.__setattr__(
                self,
                "excluded_genes",
                {name: tuple(values) for name, values in self.excluded_genes.items()},
            )
        else:
            object.__setattr__(self, "excluded_genes", tuple(self.excluded_genes))
        if np.shape(self.total_het) != (self.n_jk_blocks,):
            raise ValueError("total heterozygosity must have one value per block")
        _validate_maf(self, self.n_jk_blocks, n_annotations=len(self.gene_annot_names))
        _validate_extensions(self, aggregated=True)

    def validate_jk_shape(
        self, *, exclude_genes=None, n_jk_blocks=None, gene_blocks=None
    ):
        if n_jk_blocks is not None and n_jk_blocks != self.n_jk_blocks:
            raise ValueError("JK cache n_jk_blocks does not match request")
        if gene_blocks is not None and not np.array_equal(
            np.asarray(gene_blocks), self.gene_blocks
        ):
            raise ValueError("JK cache gene_blocks do not match request")
        if exclude_genes is None:
            return
        requested = _normalize_excluded_genes(
            exclude_genes,
            self.genes,
            tuple((self.kernel_options.get("kernel_groups") or {}).keys()),
        )
        if requested != self.excluded_genes:
            raise ValueError("JK cache exclusions do not match request")

    @property
    def kernel_labels(self):
        return tuple(kernel.label for kernel in self.layout.kernel_ids)

    @property
    def burden_labels(self):
        return self.layout.burden_labels

    @property
    def heritable(self):
        return np.asarray(self.layout.heritable_indices, dtype=np.intp)

    @property
    def overdispersion(self):
        return np.asarray(self.layout.overdispersion_indices, dtype=np.intp)

    @property
    def n_genes(self):
        if isinstance(self.excluded_genes, dict):
            return None
        return int(np.asarray(self.blocks["ng"]).sum())

    @property
    def live_blocks(self):
        return np.flatnonzero(np.asarray(self.blocks["ng"]) > 0)

    @property
    def totals(self):
        return {key: value.sum(axis=0) for key, value in self.blocks.items()}

    @property
    def options(self):
        from kasm.kernels import KernelOptions

        return KernelOptions.from_recorded(self.kernel_options)

    def save(self, path):
        arrays = {f"block_{key}": value for key, value in self.blocks.items()}
        arrays.update(
            layout_json=np.array(json.dumps(asdict(self.layout))),
            genes=np.array(self.genes, dtype=str),
            excluded_genes_json=np.array(json.dumps(self.excluded_genes)),
            n_jk_blocks=np.array(self.n_jk_blocks),
            gene_blocks=self.gene_blocks,
            sample_size=np.array(self.sample_size),
            gene_annot_names=np.array(self.gene_annot_names, dtype=str),
            kernel_options_json=np.array(json.dumps(self.kernel_options)),
            total_het=self.total_het,
        )
        _save_maf(arrays, self)
        _save_extensions(arrays, self)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path, *, exclude_genes=None, n_jk_blocks=None):
        from kasm import fitting

        with np.load(path, allow_pickle=False) as stored:
            loaded = cls(
                blocks={
                    key: stored[f"block_{key}"] for key in fitting.ANNOTATION_BLOCK_KEYS
                },
                layout=_layout_from_dict(json.loads(str(stored["layout_json"]))),
                genes=tuple(stored["genes"].tolist()),
                excluded_genes=json.loads(str(stored["excluded_genes_json"])),
                n_jk_blocks=int(stored["n_jk_blocks"]),
                gene_blocks=stored["gene_blocks"],
                sample_size=float(stored["sample_size"]),
                gene_annot_names=tuple(stored["gene_annot_names"].tolist()),
                kernel_options=json.loads(str(stored["kernel_options_json"])),
                total_het=stored["total_het"],
                burden_extensions=_load_extensions(stored, "burden"),
                covar_extensions=_load_extensions(stored, "covar"),
                **_load_maf(stored),
            )
        loaded.validate_jk_shape(exclude_genes=exclude_genes, n_jk_blocks=n_jk_blocks)
        return loaded


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def compute_suffstats(model, extra_enrich_cols=None) -> SuffStats:
    """Compute a shared base and compact annotation extensions in one LD pass."""
    from kasm.enrichment import _kernel_diagonal_weights_from_arrays

    annotations = tuple(dict.fromkeys(extra_enrich_cols or ()))
    extended = {a: model._with_burden_column(a) for a in annotations}
    dataset, layout = model.dataset, model.layout
    base_arrays = _empty_suffstats_arrays(dataset.n_genes, layout)
    base_labels = tuple(kernel.label for kernel in layout.kernel_ids)
    mappings, new_indices, new_burden_indices, enrich_exts = {}, {}, {}, {}
    old_burden_indices = {}
    for annotation, value in extended.items():
        labels = tuple(kernel.label for kernel in value.layout.kernel_ids)
        mapping = np.asarray(
            [labels.index(label) for label in base_labels], dtype=np.intp
        )
        new = np.setdiff1d(np.arange(value.layout.n_kernels), mapping)
        old_burden = np.asarray(
            [value.layout.burden_labels.index(label) for label in layout.burden_labels],
            dtype=np.intp,
        )
        new_burden = np.setdiff1d(np.arange(value.layout.n_burden), old_burden)
        mappings[annotation] = mapping
        new_indices[annotation] = new
        new_burden_indices[annotation] = new_burden
        old_burden_indices[annotation] = old_burden
        enrich_exts[annotation] = {
            "pair_trace": np.zeros((dataset.n_genes, len(new), value.layout.n_kernels)),
            "trace": np.zeros((dataset.n_genes, len(new))),
            "quadform": np.zeros((dataset.n_genes, len(new))),
            "burden_cov": np.zeros(
                (dataset.n_genes, len(new_burden), value.layout.n_burden)
            ),
            "burden_beta": np.zeros((dataset.n_genes, len(new_burden))),
            "total_hsq_weights": np.zeros((dataset.n_genes, value.layout.n_kernels)),
            "annot_hsq_weights": np.zeros((dataset.n_genes, value.layout.n_kernels)),
            "annot_het": np.zeros(dataset.n_genes),
        }
    maf_model = maf_trace = None
    independent_traces = None
    if model.options.maf_col is not None and model.options.maf_bins:
        from kasm.model import Model

        independent_traces = np.zeros(
            (
                dataset.n_genes,
                len(layout.independent_indices),
                len(model.options.maf_bins),
            )
        )
        if not model.options.burden_by_maf:
            options = replace(
                model.options,
                burden_by_maf=True,
                burden_cols=None
                if model.options.kernel_groups
                else model.options.burden_cols,
            )
            maf_model = Model(dataset, options)
            maf_trace = np.empty((dataset.n_genes, options.layout.n_kernels))
    base_index = KernelIndex.from_layout(layout)
    base_scratch = np.zeros((layout.n_kernels, layout.n_kernels))
    indices = {
        annotation: KernelIndex.from_layout(value.layout)
        for annotation, value in extended.items()
    }
    beta_tilde = dataset.beta / (dataset.se**2 * dataset.sample_size)
    covar_arrays, covar_groups = {}, {}
    covar_names = None
    diagonal_positions = [
        layout.diagonal_indices.index(i) for i in layout.independent_indices
    ]
    maf_index = None if maf_model is None else KernelIndex.from_layout(maf_model.layout)
    from kasm.covar import _covar_extension_inputs

    started = time.monotonic()
    logger.info(
        "event=suffstats_start variants=%d genes=%d kernels=%d",
        dataset.n_variants,
        dataset.n_genes,
        layout.n_kernels,
    )
    for position, gene in enumerate(dataset.genes):
        rows = dataset.gene_slice(gene)
        scale = model.variant_scale(gene)
        d2 = scale**2
        base_arrays["total_het"][position] = d2.sum()
        ld = dataset.load_ldm(gene).scale(scale)
        if independent_traces is not None:
            membership = np.column_stack(
                [mask[rows] for mask in model._maf_masks.values()]
            )
            independent_traces[position] = np.einsum(
                "v,vi,vm->im",
                ld.diag(),
                model.diag_kernel_weights(gene)[:, diagonal_positions],
                membership,
            )
            if maf_model is not None:
                maf_trace[position] = reduce_gene_trace(
                    ldm=ld,
                    burden_wgts=maf_model.burden_wgts(gene),
                    burden_partition=maf_model.burden_partition_codes(gene),
                    diag_weights=maf_model.diag_kernel_weights(gene),
                    index=maf_index,
                )

        def burden_inputs(gene=gene, d2=d2):
            for annotation, value in extended.items():
                burden = value.burden_wgts(gene)
                diagonal = value.diag_kernel_weights(gene)
                annotation_values = dataset.variant_values(annotation, gene)
                total, annotated = _kernel_diagonal_weights_from_arrays(
                    value.layout, burden, d2, annotation_values, diagonal
                )
                yield (
                    annotation,
                    {
                        "burden_wgts": burden,
                        "diag_weights": diagonal,
                        "index": indices[annotation],
                        "new": new_indices[annotation],
                        "new_burden": new_burden_indices[annotation],
                        "base_burden_positions": old_burden_indices[annotation],
                        "attribution": {
                            "total_hsq_weights": total,
                            "annot_hsq_weights": annotated,
                            "annot_het": annotation_values @ d2,
                        },
                    },
                )

        emitted = []
        for kind, name, stats in compute_gene_suffstats(
            ldm=ld,
            beta_tilde=beta_tilde[rows],
            burden_wgts=model.burden_wgts(gene),
            burden_partition=model.burden_partition_codes(gene),
            diag_weights=model.diag_kernel_weights(gene),
            index=base_index,
            scratch=base_scratch,
            burden_extensions=burden_inputs(),
            covar_extensions=_covar_extension_inputs(model, gene)
            if model.options.extra_covar_kernel_functions()
            else (),
        ):
            if kind == "base":
                _store_gene(base_arrays, position, stats, rows.stop - rows.start)
            elif kind == "burden":
                for key, array in enrich_exts[name].items():
                    array[position] = stats[key]
            else:
                emitted.append(name)
                group = stats.pop("kernel_group")
                if position == 0:
                    covar_groups[name] = group
                    covar_arrays[name] = {
                        key: np.empty(
                            (dataset.n_genes, *np.shape(value)),
                            dtype=np.asarray(value).dtype,
                        )
                        for key, value in stats.items()
                    }
                if name not in covar_arrays or covar_groups[name] != group:
                    raise ValueError(
                        "covar extension names or groups changed between genes"
                    )
                for key, array in covar_arrays[name].items():
                    if np.shape(stats[key]) != array.shape[1:]:
                        raise ValueError(
                            f"covar extension {name!r} {key} shape changed between genes"
                        )
                    array[position] = stats[key]
        if covar_names is None:
            covar_names = tuple(emitted)
        elif tuple(emitted) != covar_names:
            raise ValueError(
                "extra_covar_kernel_fun must return the same kernel names in the same order for every gene"
            )
        if (position + 1) % _LOG_EVERY == 0:
            logger.info(
                "event=suffstats_progress genes=%d/%d elapsed_seconds=%.3f",
                position + 1,
                dataset.n_genes,
                time.monotonic() - started,
            )
    logger.info(
        "event=suffstats_complete elapsed_seconds=%.3f variants=%d genes=%d",
        time.monotonic() - started,
        dataset.n_variants,
        dataset.n_genes,
    )
    base = SuffStats(
        **base_arrays,
        genes=tuple(dataset.genes),
        kernel_labels=base_labels,
        burden_labels=layout.burden_labels,
        maf_trace=maf_trace,
        maf_independent_trace=independent_traces,
    )
    result = {}
    for annotation, emodel in extended.items():
        values = enrich_exts[annotation]
        result[annotation] = ExtSuffStats(
            **{key: values[key] for key in ("trace", "quadform", "pair_trace")},
            layout=emodel.layout,
            base_to_extended=mappings[annotation],
            payload=ExtraEnrichCols(
                **{
                    key: values[key]
                    for key in ("total_hsq_weights", "annot_hsq_weights", "annot_het")
                },
                column=annotation,
                burden_cov=values["burden_cov"],
                burden_beta=values["burden_beta"],
            ),
        )
    covar_extensions = {}
    from kasm.kernels import KernelId

    for name, values in covar_arrays.items():
        group = covar_groups[name]
        with_indep = model.options.kernel_groups[group]["extra_covar_with_indep"]
        added = (
            (KernelId("I", name + ":covar_indep"), KernelId("C", name))
            if with_indep
            else (KernelId("C", name),)
        )
        augmented = replace(
            layout,
            kernel_ids=(*layout.kernel_ids, *added),
            independent_indices=(*layout.independent_indices, layout.n_kernels)
            if with_indep
            else layout.independent_indices,
            kernel_groups=(
                *layout.kernel_groups,
                *((layout.kernel_group_names.index(group),) * len(added)),
            ),
        )
        covar_extensions[name] = ExtSuffStats(
            **{key: values[key] for key in ("trace", "quadform", "pair_trace")},
            layout=augmented,
            base_to_extended=np.arange(layout.n_kernels),
            payload=CovarPayload(
                kernel_group=group,
                **{key: values[key] for key in COVAR_PROJECTION_KEYS},
                n_genes=values["eligible"],
                n_pairs=values["pair_counts"],
            ),
        )
    return replace(base, burden_extensions=result, covar_extensions=covar_extensions)


def _empty_suffstats_arrays(n_genes, layout):
    return dict(
        pair_trace_upper=np.zeros((n_genes, packed_size(layout.n_kernels))),
        trace=np.zeros((n_genes, layout.n_kernels)),
        quadform=np.zeros((n_genes, layout.n_kernels)),
        burden_cov_upper=np.zeros((n_genes, packed_size(layout.n_burden))),
        burden_beta=np.zeros((n_genes, layout.n_burden)),
        n_variants=np.zeros(n_genes, dtype=np.int64),
        total_het=np.zeros(n_genes),
    )


def _store_gene(a, pos, stats, n):
    a["pair_trace_upper"][pos] = pack_symmetric(stats["pair_trace"])
    a["trace"][pos] = stats["trace"]
    a["quadform"][pos] = stats["quadform"]
    a["burden_cov_upper"][pos] = pack_symmetric(stats["burden_cov"])
    a["burden_beta"][pos] = stats["burden_beta"]
    a["n_variants"][pos] = n


def _normalize_excluded_genes(exclude_genes, genes, kernel_group_names):
    """Canonicalize one flat selection or one selection per configured group."""
    from collections.abc import Mapping

    genes = tuple(genes)
    groups = tuple(kernel_group_names)
    if isinstance(exclude_genes, Mapping):
        if not groups or set(exclude_genes) != set(groups):
            raise ValueError("exclude_genes keys must match kernel groups")
        result = {name: tuple(sorted(set(exclude_genes[name]))) for name in groups}
        if len(set(result.values())) == 1:
            result = next(iter(result.values()))
    else:
        selected = tuple(sorted(set(exclude_genes)))
        result = selected
    values = result.values() if isinstance(result, dict) else (result,)
    unknown = set().union(*map(set, values)) - set(genes)
    if unknown:
        raise KeyError(f"excluded units are not in this model: {sorted(unknown)[:5]}")
    return result


def _get_gene_group_mask(genes, excluded_genes, group_names, covariance_groups):
    """Gene × group mask; True retains a gene, cross groups require both parents.

    Columns follow group_names. Shared exclusions are broadcast to every group.
    """
    if not isinstance(excluded_genes, dict):
        return np.broadcast_to(
            ~np.isin(genes, excluded_genes)[:, None],
            (len(genes), len(group_names)),
        )
    columns = {name: ~np.isin(genes, excluded_genes[name]) for name in excluded_genes}
    for name, (left, right) in (covariance_groups or {}).items():
        columns[name] = columns[left] & columns[right]
    return np.column_stack([columns[name] for name in group_names])


def compute_jk_suffstats(
    suffstats: SuffStats,
    *,
    exclude_genes,
    gene_annot: np.ndarray,
    sample_size: float,
    gene_annot_names,
    kernel_options: dict,
    layout,
    n_jk_blocks=100,
    gene_blocks=None,
) -> JKSuffStats:
    """Aggregate retained per-gene moments, without LD or design reconstruction."""
    from kasm import fitting

    if not isinstance(suffstats, SuffStats):
        raise TypeError("compute_jk_suffstats requires retained per-gene SuffStats")
    if n_jk_blocks < 1:
        raise ValueError("n_jk_blocks must be at least 1")
    excluded_genes = _normalize_excluded_genes(
        exclude_genes,
        suffstats.genes,
        tuple((kernel_options.get("kernel_groups") or {}).keys()),
    )
    block_of, _, _ = fitting.jackknife_blocks(
        suffstats.genes,
        fitting.assign_jackknife_blocks(suffstats.n_genes, n_jk_blocks)
        if gene_blocks is None
        else gene_blocks,
        n_jk_blocks,
    )
    gene_annot = fitting.validate_gene_annotations(gene_annot)
    if gene_annot.shape != (suffstats.n_genes, len(gene_annot_names)):
        raise ValueError("gene annotations must align with retained genes and names")
    grouped = isinstance(excluded_genes, dict)
    gene_group_mask = _get_gene_group_mask(
        suffstats.genes,
        excluded_genes,
        layout.kernel_group_names,
        kernel_options.get("covariance_between_kernel_groups"),
    )
    if grouped:
        for position, name in enumerate(layout.kernel_group_names):
            counts = np.bincount(
                block_of,
                weights=gene_group_mask[:, position],
                minlength=n_jk_blocks,
            )
            if not counts.sum() or (
                n_jk_blocks > 1 and np.any(counts.sum() - counts == 0)
            ):
                raise ValueError(
                    f"kernel group {name!r} has no retained gene in a fit geometry"
                )
    keep = np.ones(suffstats.n_genes, dtype=bool) if grouped else gene_group_mask[:, 0]
    selected_blocks, annot = block_of[keep], gene_annot[keep]
    gene_kernel_mask = gene_group_mask[:, layout.kernel_groups]
    upper = np.triu_indices(suffstats.n_kernels)
    blocks = fitting.annotation_block_sums(
        (
            suffstats.pair_trace_upper
            * gene_kernel_mask[:, upper[0]]
            * gene_kernel_mask[:, upper[1]]
        )[keep],
        (suffstats.trace * gene_kernel_mask)[keep],
        (suffstats.quadform * gene_kernel_mask)[keep],
        annot,
        selected_blocks,
        n_blocks=n_jk_blocks,
    )
    # Only block-level base triangles are expanded for the numerical solver.
    blocks["pt0"] = unpack_symmetric(blocks["pt0"], suffstats.n_kernels)
    blocks["pt1"] = np.moveaxis(
        unpack_symmetric(np.moveaxis(blocks["pt1"], -1, 1), suffstats.n_kernels),
        1,
        -1,
    )
    blocks["ga1"] = np.zeros((n_jk_blocks, annot.shape[1]))
    np.add.at(blocks["ga1"], selected_blocks, annot)
    # For different group supports, ng tracks retained group-gene contributions
    # for block occupancy, not a deduplicated gene count or a reported total.
    blocks["ng"] = np.bincount(
        selected_blocks,
        weights=(gene_group_mask.sum(axis=1) if grouped else gene_group_mask[:, 0])[
            keep
        ],
        minlength=n_jk_blocks,
    )
    blocks["nv"] = np.bincount(
        selected_blocks,
        weights=suffstats.n_variants[keep],
        minlength=n_jk_blocks,
    )
    jk_stats = JKSuffStats(
        blocks=blocks,
        total_het=np.bincount(
            selected_blocks, weights=suffstats.total_het[keep], minlength=n_jk_blocks
        ),
        genes=suffstats.genes,
        layout=layout,
        excluded_genes=excluded_genes,
        n_jk_blocks=n_jk_blocks,
        gene_blocks=block_of,
        sample_size=sample_size,
        gene_annot_names=tuple(gene_annot_names),
        kernel_options=dict(kernel_options),
        maf_trace=_aggregate_maf(
            suffstats.maf_trace,
            layout,
            kernel_options,
            gene_kernel_mask,
            keep,
            annot,
            selected_blocks,
            n_jk_blocks,
        ),
        maf_independent_trace=None
        if suffstats.maf_independent_trace is None
        else fitting.annotation_weighted_sums(
            (
                suffstats.maf_independent_trace
                * gene_kernel_mask[:, layout.independent_indices, None]
            )[keep],
            annot,
            selected_blocks,
            n_jk_blocks,
        )[1],
    )
    return replace(
        jk_stats,
        **_aggregate_extensions(
            suffstats,
            jk_stats,
            gene_annot=gene_annot,
            gene_group_mask=gene_group_mask,
        ),
    )


def _aggregate_extensions(suffstats, base, *, gene_annot, gene_group_mask):
    """Aggregate every compact extension with the parent's exclusions and blocks."""
    from kasm import fitting

    keep = (
        np.ones(suffstats.n_genes, dtype=bool)
        if isinstance(base.excluded_genes, dict)
        else gene_group_mask[:, 0]
    )
    selected_blocks, annot = base.gene_blocks[keep], gene_annot[keep]
    result = {"burden_extensions": {}, "covar_extensions": {}}
    for namespace in ("burden", "covar"):
        for name, ext in getattr(suffstats, namespace + "_extensions").items():
            gene_kernel_mask = gene_group_mask[:, ext.layout.kernel_groups]
            new_gene_kernel_mask = gene_kernel_mask[:, ext.new_indices]
            enrich_ext = fitting.annotation_block_sums(
                (
                    ext.pair_trace
                    * new_gene_kernel_mask[:, :, None]
                    * gene_kernel_mask[:, None, :]
                )[keep],
                (ext.trace * new_gene_kernel_mask)[keep],
                (ext.quadform * new_gene_kernel_mask)[keep],
                annot,
                selected_blocks,
                base.n_jk_blocks,
            )
            keys = (
                ("total_hsq_weights", "annot_hsq_weights")
                if namespace == "burden"
                else COVAR_PROJECTION_KEYS
            )
            weights = {
                key: fitting.annotation_weighted_sums(
                    (getattr(ext.payload, key) * gene_kernel_mask)[keep],
                    annot,
                    selected_blocks,
                    base.n_jk_blocks,
                )[1]
                for key in keys
            }

            def counts(values):
                return np.bincount(
                    selected_blocks,
                    weights=(values * new_gene_kernel_mask.any(axis=1))[keep],
                    minlength=base.n_jk_blocks,
                )

            payload = (
                ExtraEnrichCols(
                    **weights,
                    annot_het=counts(ext.payload.annot_het),
                    column=ext.payload.column,
                )
                if namespace == "burden"
                else CovarPayload(
                    **weights,
                    kernel_group=ext.payload.kernel_group,
                    n_genes=counts(ext.payload.n_genes),
                    n_pairs=counts(ext.payload.n_pairs),
                )
            )
            result[namespace + "_extensions"][name] = ExtSuffStats(
                **{
                    field: np.concatenate(
                        (enrich_ext[key + "0"][..., None], enrich_ext[key + "1"]),
                        axis=-1,
                    )
                    for field, key in (
                        ("trace", "tr"),
                        ("quadform", "qf"),
                        ("pair_trace", "pt"),
                    )
                },
                layout=ext.layout,
                base_to_extended=ext.base_to_extended.copy(),
                payload=payload,
            )
    return result


def _extended_options(base_options, extension):
    if not isinstance(extension.payload, ExtraEnrichCols):
        return base_options
    options = dict(base_options)
    column = extension.payload.column
    if options.get("kernel_groups"):
        options["kernel_groups"] = {
            name: {**config, "burden_cols": [*config["burden_cols"], column]}
            for name, config in options["kernel_groups"].items()
        }
    else:
        options["burden_cols"] = [*(options.get("burden_cols") or ()), column]
    return options


def _select_jk_extension(base, extension):
    """Assemble one solver input, sharing parent metadata and retaining no siblings."""
    blocks = _assemble_jk_blocks(
        base.blocks,
        extension.blocks,
        extension.base_to_extended,
        extension.layout.n_kernels,
    )
    layout = extension.layout
    selected = replace(
        base,
        blocks=blocks,
        layout=layout,
        kernel_options=_extended_options(base.kernel_options, extension),
        maf_trace=None,
        maf_independent_trace=None,
        burden_extensions={},
        covar_extensions={},
    )
    return selected


def _assemble_jk_blocks(base, rows, old, n_kernels):
    """Scatter aggregated old/old moments and new rows into one fitting system."""
    from kasm import fitting

    old = np.asarray(old, dtype=np.intp)
    new = np.setdiff1d(np.arange(n_kernels), old)
    blocks = fitting.empty_annotation_blocks(
        len(base["ng"]), n_kernels, base["ga1"].shape[1]
    )
    for key in ("pt0", "pt1"):
        blocks[key][:, old[:, None], old] = base[key]
        blocks[key][:, new, :] = rows[key]
        blocks[key][:, :, new] = rows[key].swapaxes(1, 2)
    for key in ("tr0", "tr1", "qf0", "qf1"):
        blocks[key][:, old] = base[key]
        blocks[key][:, new] = rows[key]
    for key in ("ga1", "ng", "nv"):
        blocks[key] = base[key]
    return blocks


# ---------------------------------------------------------------------------
# The reducer: an LDM and arrays only
# ---------------------------------------------------------------------------


def _kernel_quadforms(scores, diagonal_quad, index):
    quadform = np.zeros(index.n_kernels)
    quadform[index.burden_indices] = np.einsum("pe,pe->p", scores, scores)
    quadform[index.cross_indices] = 2.0 * np.einsum(
        "ce,ce->c", scores[index.cross_left], scores[index.cross_right]
    )
    quadform[index.diag_indices] = diagonal_quad
    return quadform


def reduce_gene_rhs(
    *, beta_tilde, burden_wgts, burden_partition=None, diag_weights, index
):
    """Common RHS array math for refreshing retained gene statistics."""
    burden_beta = burden_wgts.T @ beta_tilde
    scores = (
        burden_beta[:, None]
        if burden_partition is None
        else np.einsum(
            "vpe,v->pe",
            _partitioned_burden_wgts_factors(burden_wgts, burden_partition),
            beta_tilde,
        )
    )
    return dict(
        quadform=_kernel_quadforms(scores, diag_weights.T @ (beta_tilde**2), index),
        burden_beta=burden_beta,
    )


def _diagonal_moments(ldm, beta_tilde, weights, propagated, index):
    """Diagonal traces, quadratic forms, and cross-moments with burden kernels.

    Propagated burden factors have shape (variants, burdens, partitions).
    A single partition represents the ordinary unpartitioned burden design.
    """
    pair = np.zeros((weights.shape[1], index.n_kernels))
    pair[:, index.burden_indices] = np.einsum(
        "vd,vpe,vpe->dp", weights, propagated, propagated
    )
    pair[:, index.cross_indices] = 2 * np.einsum(
        "vd,vce,vce->dc",
        weights,
        propagated[:, index.cross_left, :],
        propagated[:, index.cross_right, :],
    )
    return weights.T @ ldm.diag(), weights.T @ (beta_tilde**2), pair


def compute_gene_suffstats(
    *,
    ldm,
    beta_tilde,
    burden_wgts,
    diag_weights,
    index,
    burden_partition=None,
    scratch=None,
    burden_extensions=(),
    covar_extensions=(),
):
    """Yield the baseline and one extension at a time for a single gene.

    Inputs are arrays and sparse kernels. The baseline LD-propagated burden
    design is reused by covar extensions and by the baseline columns of each
    burden extension. No pair matrix crosses two different extensions.
    """
    # Shared intermediates belong to this gene call, before either output family.
    propagated = ldm @ burden_wgts
    burden_cov = burden_wgts.T @ propagated
    burden_beta = burden_wgts.T @ beta_tilde
    n_burden = burden_wgts.shape[1]
    if burden_partition is None:
        base_factor_beta = burden_beta[:, None]
    else:
        base_factors = _partitioned_burden_wgts_factors(burden_wgts, burden_partition)
        n_partitions = base_factors.shape[-1]
        flat = base_factors.reshape(len(burden_wgts), -1)
        base_projected_factors = (ldm @ flat).reshape(base_factors.shape)
        base_factor_cov = (
            flat.T @ base_projected_factors.reshape(len(burden_wgts), -1)
        ).reshape(n_burden, n_partitions, n_burden, n_partitions)
        base_factor_beta = np.einsum("vpe,v->pe", base_factors, beta_tilde)
    quadform = _kernel_quadforms(
        base_factor_beta, np.zeros(len(index.diag_indices)), index
    )
    n_kernels = index.n_kernels
    pair_trace = np.zeros((n_kernels, n_kernels)) if scratch is None else scratch
    pair_trace.fill(0.0)
    trace = np.zeros(n_kernels)
    burden_index = index.burden_indices
    cross_index = index.cross_indices
    left, right = index.cross_left, index.cross_right

    # -- burden and cross kernels, all from G = U' R_s U ---------------------
    # K_p = u_p u_p' is rank one, so tr(K_p R_s) = G_pp,
    # tr(K_p R_s K_q R_s) = G_pq^2, and b' K_p b = (u_p' b)^2.
    if burden_partition is None:
        if burden_index.size:
            pair_trace[np.ix_(burden_index, burden_index)] = burden_cov**2
            trace[burden_index] = np.diag(burden_cov)

        if cross_index.size:
            # K_c = u_p u_q' + u_q u_p'.
            burden_cross_pair_trace = 2.0 * burden_cov[:, left] * burden_cov[:, right]
            pair_trace[np.ix_(burden_index, cross_index)] = burden_cross_pair_trace
            pair_trace[np.ix_(cross_index, burden_index)] = burden_cross_pair_trace.T
            pair_trace[np.ix_(cross_index, cross_index)] = 2.0 * (
                burden_cov[np.ix_(left, left)] * burden_cov[np.ix_(right, right)]
                + burden_cov[np.ix_(left, right)] * burden_cov[np.ix_(right, left)]
            )
            trace[cross_index] = 2.0 * burden_cov[left, right]
    elif burden_index.size:
        # Contract G[burden, group, burden, group] directly. The public logical
        # kernels are sums over same-group factors; no m-by-m kernel is formed.
        trace[burden_index] = np.einsum("pepe->p", base_factor_cov)
        pair_trace[np.ix_(burden_index, burden_index)] = np.einsum(
            "peqf,peqf->pq", base_factor_cov, base_factor_cov
        )
        if cross_index.size:
            trace[cross_index] = 2.0 * np.trace(
                base_factor_cov[left, :, right, :], axis1=1, axis2=2
            )
            factors_flat = base_factor_cov.transpose(0, 2, 1, 3).reshape(
                n_burden * n_burden, n_partitions * n_partitions
            )
            factor_gram = (factors_flat @ factors_flat.T).reshape(
                n_burden, n_burden, n_burden, n_burden
            )
            positions = np.arange(n_burden)
            burden_cross_pair_trace = (
                2.0
                * factor_gram[
                    positions[:, None],
                    left[None, :],
                    positions[:, None],
                    right[None, :],
                ]
            )
            pair_trace[np.ix_(burden_index, cross_index)] = burden_cross_pair_trace
            pair_trace[np.ix_(cross_index, burden_index)] = burden_cross_pair_trace.T
            pair_trace[np.ix_(cross_index, cross_index)] = 2.0 * (
                factor_gram[
                    left[:, None],
                    left[None, :],
                    right[:, None],
                    right[None, :],
                ]
                + factor_gram[
                    left[:, None],
                    right[None, :],
                    right[:, None],
                    left[None, :],
                ]
            )

    # -- diagonal kernels: independent and overdispersion --------------------
    diagonal_index = index.diag_indices
    if diagonal_index.size:
        diagonal_trace, diagonal_quadform, diagonal_pair = _diagonal_moments(
            ldm,
            beta_tilde,
            diag_weights,
            propagated[:, :, None]
            if burden_partition is None
            else base_projected_factors,
            index,
        )
        trace[diagonal_index] = diagonal_trace
        quadform[diagonal_index] = diagonal_quadform
        diagonal_pair[:, diagonal_index] = ldm.diag_pair_trace(diag_weights)
        pair_trace[diagonal_index, :] = diagonal_pair
        pair_trace[:, diagonal_index] = diagonal_pair.T

    yield (
        "base",
        None,
        dict(
            pair_trace=pair_trace,
            trace=trace,
            quadform=quadform,
            burden_cov=burden_cov,
            burden_beta=burden_beta,
        ),
    )
    for name, burden_ext in burden_extensions:
        extension_index = burden_ext["index"]
        old, new = burden_ext["base_burden_positions"], burden_ext["new_burden"]
        weights = burden_ext["burden_wgts"]
        added = weights[:, new]
        projected_new = ldm @ added
        n_burden = weights.shape[1]
        gram = np.empty((n_burden, n_burden))
        gram[np.ix_(old, old)] = burden_cov
        gram[np.ix_(old, new)] = burden_wgts.T @ projected_new
        gram[np.ix_(new, old)] = gram[np.ix_(old, new)].T
        gram[np.ix_(new, new)] = added.T @ projected_new
        beta = np.empty(n_burden)
        beta[old], beta[new] = burden_beta, added.T @ beta_tilde
        if burden_partition is None:
            covariance = gram[:, :, None, None]
            scores = beta[:, None]
            projected = np.empty((len(weights), n_burden, 1))
            projected[:, old, 0], projected[:, new, 0] = propagated, projected_new
        else:
            factors = _partitioned_burden_wgts_factors(added, burden_partition)
            n_parts = factors.shape[-1]
            new_projected = (ldm @ factors.reshape(len(weights), -1)).reshape(
                factors.shape
            )
            old_factors = (
                base_factors if len(old) else np.empty((len(weights), 0, n_parts))
            )
            covariance = np.empty((n_burden, n_burden, n_parts, n_parts))
            if len(old):
                covariance[np.ix_(old, old)] = base_factor_cov.transpose(0, 2, 1, 3)
            cross = np.einsum("vpe,vqf->pqef", old_factors, new_projected)
            covariance[np.ix_(old, new)] = cross
            covariance[np.ix_(new, old)] = cross.transpose(1, 0, 3, 2)
            covariance[np.ix_(new, new)] = np.einsum(
                "vpe,vqf->pqef", factors, new_projected
            )
            scores = np.empty((n_burden, n_parts))
            if len(old):
                scores[old] = base_factor_beta
            scores[new] = np.einsum("vpe,v->pe", factors, beta_tilde)
            projected = np.empty((len(weights), n_burden, n_parts))
            if len(old):
                projected[:, old] = base_projected_factors
            projected[:, new] = new_projected
        # Each low-rank kernel is sym(u_p u_q') times one (burden) or two (cross).
        low_rank = np.r_[extension_index.burden_indices, extension_index.cross_indices]
        left = np.r_[np.arange(n_burden), extension_index.cross_left]
        right = np.r_[np.arange(n_burden), extension_index.cross_right]
        multiplier = np.r_[
            np.ones(n_burden), np.full(len(extension_index.cross_indices), 2.0)
        ]
        rows = burden_ext["new"]
        pair = np.zeros((len(rows), extension_index.n_kernels))
        trace, quad = np.empty(len(rows)), np.empty(len(rows))
        for row, kernel in enumerate(rows):
            position = np.flatnonzero(low_rank == kernel)
            if len(position) != 1:
                raise ValueError(
                    "a burden extension must add only burden/cross kernels"
                )
            p, q, c = left[position[0]], right[position[0]], multiplier[position[0]]
            pair[row, low_rank] = (
                c
                * multiplier
                / 2
                * (
                    np.einsum("kef,kef->k", covariance[p, left], covariance[q, right])
                    + np.einsum("kef,kef->k", covariance[p, right], covariance[q, left])
                )
            )
            pair[row, extension_index.diag_indices] = c * np.einsum(
                "vd,ve,ve->d",
                burden_ext["diag_weights"],
                projected[:, p],
                projected[:, q],
            )
            trace[row] = c * np.trace(covariance[p, q])
            quad[row] = c * (scores[p] @ scores[q])
        yield (
            "burden",
            name,
            dict(
                pair_trace=pair,
                trace=trace,
                quadform=quad,
                burden_cov=gram[new],
                burden_beta=beta[new],
                **burden_ext["attribution"],
            ),
        )

    from kasm.covar import _compute_covar_ext

    for name, covar_ext in covar_extensions:
        yield (
            "covar",
            name,
            {
                "kernel_group": covar_ext["kernel_group"],
                **_compute_covar_ext(
                    kernel=covar_ext["kernel"],
                    with_indep=covar_ext["with_indep"],
                    independent_indices=covar_ext["independent_indices"],
                    ldm=ldm,
                    beta_tilde=beta_tilde,
                    burden_wgts=burden_wgts,
                    diag_weights=diag_weights,
                    index=index,
                    propagated_burden_wgts=propagated,
                ),
            },
        )


def reduce_gene_trace(
    *,
    ldm: LDM,
    burden_wgts: np.ndarray,
    burden_partition: np.ndarray | None,
    diag_weights: np.ndarray,
    index: KernelIndex,
) -> np.ndarray:
    """LD-weighted kernel traces only, without RHS or kernel-pair moments."""
    factors = (
        burden_wgts[:, :, None]
        if burden_partition is None
        else _partitioned_burden_wgts_factors(burden_wgts, burden_partition)
    )
    projected = (ldm @ factors.reshape(len(factors), -1)).reshape(factors.shape)
    trace = np.zeros(index.n_kernels)
    trace[index.burden_indices] = np.einsum("vpe,vpe->p", factors, projected)
    trace[index.cross_indices] = 2.0 * np.einsum(
        "vce,vce->c", factors[:, index.cross_left], projected[:, index.cross_right]
    )
    trace[index.diag_indices] = diag_weights.T @ ldm.diag()
    return trace


def _partitioned_burden_wgts_factors(
    burden_wgts: np.ndarray, partition: np.ndarray
) -> np.ndarray:
    """One masked burden weight per logical column and local partition."""
    partition = np.asarray(partition)
    if partition.shape != (burden_wgts.shape[0],):
        raise ValueError("burden_partition must have one value per variant")
    if not np.issubdtype(partition.dtype, np.integer):
        raise TypeError("burden_partition must contain integer codes")
    codes = partition.astype(np.intp, copy=False)
    if (codes < 0).any():
        raise ValueError("burden_partition codes must be nonnegative")
    membership = codes[:, None] == np.arange(codes.max() + 1)
    return burden_wgts[:, :, None] * membership[:, None, :]
