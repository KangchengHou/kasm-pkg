"""Compact, reusable LD panels with alignment performed at analysis time."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import scipy.sparse as sps

from kasm.dataset import parse_variant_ids

_CHUNK = struct.Struct("<8sQQ")
_CHUNK_MAGIC = b"KASMLD1\0"
_I4 = np.dtype("<i4")
_F8 = np.dtype("<f8")


@dataclass(frozen=True)
class KernelTraceStats:
    """Trace quantities for a sparse kernel and an LDM."""

    trace: float
    self_pair_trace: float
    diag_pair_weights: np.ndarray


def _variant_id_array(variant_ids) -> np.ndarray:
    canonical = parse_variant_ids(variant_ids)["variant"]
    return np.asarray(canonical.to_list(), dtype=object)


class LDM:
    """A gene-level LD matrix with sparse, matrix-like operations.

    The sparse-plus-rank-one storage is deliberately private.  Callers can
    multiply, scale, subset, inspect the diagonal, or materialize the matrix
    without depending on that representation.
    """

    __slots__ = ("_mean", "_sparse", "n_covered")

    def __init__(self, S: sps.csr_matrix, v: np.ndarray, n_covered: int | None = None):
        self._sparse = S
        self._mean = v
        self.n_covered = S.shape[0] if n_covered is None else int(n_covered)

    @staticmethod
    def identity(m: int) -> LDM:
        return LDM(sps.eye(m, format="csr"), np.zeros(m), n_covered=0)

    def scale(self, d: np.ndarray) -> LDM:
        coo = self._sparse.tocoo(copy=True)
        coo.data = coo.data * (d[coo.row] * d[coo.col])
        return LDM(
            coo.tocsr(),
            d * self._mean,
            self.n_covered,
        )

    def subset(self, indices) -> LDM:
        """Return the principal submatrix at *indices*, in the given order."""
        indices = np.asarray(indices, dtype=np.intp)
        return LDM(
            self._sparse[indices, :][:, indices].tocsr(),
            self._mean[indices],
            min(self.n_covered, len(indices)),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self._sparse.shape

    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        """Multiply by a vector or a matrix whose columns are vectors."""
        other = np.asarray(other)
        product = self._sparse @ other
        mean_product = self._mean @ other
        if other.ndim == 1:
            product -= self._mean * mean_product
        else:
            product -= self._mean[:, None] * mean_product
        return product

    def diag(self) -> np.ndarray:
        """Return the matrix diagonal without materializing the matrix."""
        return self._sparse.diagonal() - self._mean**2

    def quadform(self, design: np.ndarray) -> np.ndarray:
        """Return ``design.T @ self @ design`` with one sparse multiply."""
        design = np.asarray(design)
        return design.T @ (self @ design)

    def diag_pair_trace(
        self, weights: np.ndarray, right_weights: np.ndarray | None = None
    ) -> np.ndarray:
        """Return square or rectangular diagonal-kernel pair traces without densifying."""
        weights = np.asarray(weights)
        right_weights = weights if right_weights is None else np.asarray(right_weights)
        mean_squared = self._mean**2
        weighted_mean = weights * self._mean[:, None]
        result = (
            weights.T @ self._sparse.multiply(self._sparse) @ right_weights
            - 2.0
            * (weighted_mean.T @ (self._sparse @ (right_weights * self._mean[:, None])))
            + np.outer(weights.T @ mean_squared, right_weights.T @ mean_squared)
        )
        return np.asarray(result)

    def kernel_trace_stats(self, kernel: sps.spmatrix) -> KernelTraceStats:
        """Return a kernel's trace and pair-trace inputs without densifying."""
        kernel_mean = kernel @ self._mean
        mean_kernel_mean = float(self._mean @ kernel_mean)
        sparse_kernel_mean = self._sparse @ kernel_mean
        trace = float(kernel.multiply(self._sparse.T).sum() - mean_kernel_mean)

        pair_trace = 0.0
        diag_pair_weights = np.zeros(self.shape[0])
        for start in range(0, self.shape[0], 256):
            stop = min(start + 256, self.shape[0])
            kernel_sparse_rows = kernel[start:stop] @ self._sparse
            pair_trace += float(
                kernel_sparse_rows.multiply(self._sparse[start:stop] @ kernel).sum()
            )
            diag_pair_weights += np.asarray(
                kernel_sparse_rows.multiply(self._sparse[start:stop]).sum(axis=0)
            ).ravel()
        pair_trace -= 2.0 * float(sparse_kernel_mean @ kernel_mean)
        pair_trace += mean_kernel_mean**2
        diag_pair_weights -= 2.0 * self._mean * sparse_kernel_mean
        diag_pair_weights += self._mean**2 * mean_kernel_mean
        return KernelTraceStats(trace, pair_trace, diag_pair_weights)

    def stored_upper_pairs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return stored upper-triangle pair coordinates and LD values.

        This supports panel-compression diagnostics without exposing the
        sparse-plus-rank-one representation to the caller.
        """
        stored = self._sparse.tocoo()
        upper = stored.row < stored.col
        left = stored.row[upper]
        right = stored.col[upper]
        values = stored.data[upper] - self._mean[left] * self._mean[right]
        return left, right, values

    def toarray(self) -> np.ndarray:
        return self._sparse.toarray() - np.outer(self._mean, self._mean)

    def __repr__(self) -> str:
        return (
            f"LDM(m={self._sparse.shape[0]}, stored.nnz={self._sparse.nnz}, "
            f"rank_one.nz={int(np.count_nonzero(self._mean))})"
        )


def _pack_chunk(matrix: sps.spmatrix, vector: np.ndarray) -> bytes:
    matrix = matrix.tocsr()
    matrix.sort_indices()
    n, m = matrix.shape
    if n != m:
        raise ValueError(f"LD matrix is not square: {matrix.shape}")
    if n > np.iinfo(np.int32).max or matrix.nnz > np.iinfo(np.int32).max:
        raise ValueError("LD matrix exceeds the format's int32 CSR limits")
    return b"".join(
        (
            _CHUNK.pack(_CHUNK_MAGIC, n, matrix.nnz),
            np.asarray(matrix.indptr, dtype=_I4).tobytes(),
            np.asarray(matrix.indices, dtype=_I4).tobytes(),
            np.asarray(matrix.data, dtype=_F8).tobytes(),
            np.asarray(vector, dtype=_F8).tobytes(),
        )
    )


def _unpack_chunk(payload: bytes) -> tuple[sps.csr_matrix, np.ndarray]:
    if len(payload) < _CHUNK.size:
        raise ValueError("truncated LDM matrix chunk")
    magic, n, nnz = _CHUNK.unpack_from(payload)
    if magic != _CHUNK_MAGIC:
        raise ValueError("invalid LDM matrix chunk")
    expected = _CHUNK.size + (n + 1 + nnz) * 4 + (nnz + n) * 8
    if len(payload) != expected:
        raise ValueError(
            f"invalid LDM matrix chunk length: expected {expected}, got {len(payload)}"
        )
    cursor = _CHUNK.size
    indptr = np.frombuffer(payload, _I4, n + 1, cursor)
    cursor += (n + 1) * 4
    indices = np.frombuffer(payload, _I4, nnz, cursor)
    cursor += nnz * 4
    data = np.frombuffer(payload, _F8, nnz, cursor)
    cursor += nnz * 8
    vector = np.frombuffer(payload, _F8, n, cursor)
    if indptr[0] != 0 or indptr[-1] != nnz or np.any(indptr[1:] < indptr[:-1]):
        raise ValueError("invalid CSR row offsets in LDM matrix chunk")
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("invalid CSR column index in LDM matrix chunk")
    return sps.csr_matrix((data, indices, indptr), shape=(n, n)), vector


class LDMLoader:
    """Load a compact correlation panel and align each gene to a Dataset."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.suffix != ".ldm" or not self.path.is_file():
            raise ValueError(f"LD panel must be a .ldm file, got {self.path}")
        self._zfile = ZipFile(self.path)
        try:
            self._gene_names = self._zfile.read("meta/genes.txt").decode().splitlines()
            if len(set(self._gene_names)) != len(self._gene_names):
                raise ValueError("LDM metadata contains duplicate gene names")
            self._gene_position = {
                gene: position for position, gene in enumerate(self._gene_names)
            }
            self._byte_offsets = np.frombuffer(
                self._zfile.read("meta/gene_byte_offsets.i8"), dtype="<i8"
            )
            self._variant_bytes = self._zfile.read("meta/variant_ids.txt")
            expected_offsets = len(self._gene_names) + 1
            if len(self._byte_offsets) != expected_offsets:
                raise ValueError("LDM metadata offset counts do not match its genes")
            if (
                self._byte_offsets[0] != 0
                or np.any(self._byte_offsets[1:] < self._byte_offsets[:-1])
                or self._byte_offsets[-1] != len(self._variant_bytes)
            ):
                raise ValueError("LDM metadata contains invalid offsets")
        except Exception:
            self._zfile.close()
            raise

    @property
    def genes(self) -> set[str]:
        return set(self._gene_names)

    def _ids(self, gene: str) -> list[str]:
        try:
            position = self._gene_position[gene]
        except KeyError:
            raise KeyError(
                f"gene {gene!r} is not in the LD panel at {self.path}"
            ) from None
        start, stop = map(int, self._byte_offsets[position : position + 2])
        if start == stop:
            return []
        return self._variant_bytes[start:stop].decode().splitlines()

    def load_gene_ldm(self, gene: str, variant_ids, *, _canonical: bool = False) -> LDM:
        requested = (
            np.asarray(variant_ids, dtype=object)
            if _canonical
            else _variant_id_array(variant_ids)
        )
        if len(set(requested)) != len(requested):
            raise ValueError(
                f"duplicate variant IDs requested for {gene!r}; LD would be structurally singular"
            )
        panel_ids = self._ids(gene)
        matrix, vector = _unpack_chunk(self._zfile.read(f"matrix/{gene}.bin"))
        if matrix.shape[0] != len(panel_ids):
            raise ValueError(f"LDM metadata and matrix disagree for {gene!r}")
        if not np.isfinite(matrix.data).all() or not np.isfinite(vector).all():
            raise ValueError(f"LDM matrix contains non-finite values for {gene!r}")
        positions = {variant: i for i, variant in enumerate(panel_ids)}
        located = np.fromiter(
            (positions.get(variant, -1) for variant in requested),
            dtype=np.intp,
            count=len(requested),
        )
        usable = located >= 0
        rows = located[usable]
        if rows.size == 0:
            return LDM.identity(len(requested))
        out_positions = np.flatnonzero(usable)
        selected = matrix[rows, :][:, rows].tocoo()
        off_panel = np.flatnonzero(~usable)
        full = sps.csr_matrix(
            (
                np.concatenate((np.ones(off_panel.size), selected.data)),
                (
                    np.concatenate((off_panel, out_positions[selected.row])),
                    np.concatenate((off_panel, out_positions[selected.col])),
                ),
            ),
            shape=(len(requested), len(requested)),
        )
        full_vector = np.zeros(len(requested))
        full_vector[out_positions] = vector[rows]
        return LDM(full, full_vector, n_covered=rows.size)

    def close(self) -> None:
        self._zfile.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __contains__(self, gene: str) -> bool:
        return gene in self._gene_position
