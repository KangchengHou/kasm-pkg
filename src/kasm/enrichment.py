"""Additive burden-attribution moments for binary, continuous and signed annotations.

The raw annotation weights readout linearly. Binary annotations have enrichment
interpretation; continuous annotations define weighted moments. Random diagonal
attribution and fixed nested projections are distinct estimands.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import ClassVar

import numpy as np
import polars as pl

from kasm import fitting


def _annotation_heterozygosity(model, enrichment_col: str, genes=None):
    """Return additive annotated and total heterozygosity moments."""
    selected = None if genes is None else set(genes)
    numerator = total = 0.0
    for gene in model.dataset.genes:
        if selected is not None and gene not in selected:
            continue
        scale_squared = model.variant_scale(gene) ** 2
        annotation = model.dataset.variant_values(enrichment_col, gene)
        numerator += float((annotation * scale_squared).sum())
        total += float(scale_squared.sum())
    return numerator, total


def _kernel_diagonal_weights_from_arrays(
    layout, burden_wgts, d2, annotation, diag_wgts=None
):
    """Diagonal-attribution contractions from one gene's plain arrays.

    A within-gene partition changes off-diagonal pairs only: each variant
    belongs to exactly one partition, so burden and cross-kernel diagonals
    remain U**2 and 2*U_left*U_right.
    """
    d2 = np.asarray(d2)
    weighted = d2 * np.asarray(annotation)
    total = np.zeros(layout.n_kernels, dtype=float)
    annotated = np.zeros(layout.n_kernels, dtype=float)
    squared = burden_wgts**2
    burden_indices = np.asarray(layout.burden_indices, dtype=np.intp)
    total[burden_indices] = d2 @ squared
    annotated[burden_indices] = weighted @ squared
    cross_indices = np.asarray(layout.cross_indices, dtype=np.intp)
    if len(cross_indices):
        pairs = np.asarray(layout.cross_pairs, dtype=np.intp)
        product = 2.0 * burden_wgts[:, pairs[:, 0]] * burden_wgts[:, pairs[:, 1]]
        total[cross_indices] = d2 @ product
        annotated[cross_indices] = weighted @ product
    if layout.independent_indices:
        if diag_wgts is None:
            if len(layout.independent_indices) != 1:
                raise ValueError(
                    "multiple independent kernels require diagonal weights"
                )
            independent_wgts = np.ones((len(d2), 1))
        else:
            positions = {
                index: position
                for position, index in enumerate(layout.diagonal_indices)
            }
            independent_wgts = np.asarray(diag_wgts)[
                :, [positions[index] for index in layout.independent_indices]
            ]
        total[list(layout.independent_indices)] = d2 @ independent_wgts
        annotated[list(layout.independent_indices)] = weighted @ independent_wgts
    # Overdispersion positions intentionally remain zero: they are residual
    # variance, not genetic hsq.
    return total, annotated


@dataclass(frozen=True)
class EnrichResult:
    """Additive hsq and denominator moments for enrichment readouts."""

    annot_hsq: float
    total_hsq: float
    annot_het: float
    total_het: float
    annotation: str
    n_genes: int
    n_variants: int
    annot_hsq_variance: float | None = None
    total_hsq_variance: float | None = None
    annot_total_hsq_covariance: float | None = None
    annot_hsq_jk: np.ndarray | None = None
    total_hsq_jk: np.ndarray | None = None
    annot_het_jk: np.ndarray | None = None
    total_het_jk: np.ndarray | None = None
    annot_count: float | None = None
    total_count: float | None = None
    annot_count_jk: np.ndarray | None = None
    total_count_jk: np.ndarray | None = None
    _ARRAY_FIELDS: ClassVar[tuple[str, ...]] = (
        "annot_hsq_jk",
        "total_hsq_jk",
        "annot_het_jk",
        "total_het_jk",
        "annot_count_jk",
        "total_count_jk",
    )

    def save(self, path) -> None:
        """Write moments and metadata to a compressed ``.npz``."""
        stored = {
            name: getattr(self, name)
            for name in self._ARRAY_FIELDS
            if getattr(self, name) is not None
        }
        stored["result_json"] = np.array(
            json.dumps(
                {
                    field.name: getattr(self, field.name)
                    for field in fields(self)
                    if field.name not in self._ARRAY_FIELDS
                }
            )
        )
        np.savez_compressed(path, **stored)

    @classmethod
    def load(cls, path) -> EnrichResult:
        """Read an independently stored enrichment result without a fitted model."""
        with np.load(path, allow_pickle=False) as archive:
            values = json.loads(str(archive["result_json"]))
            values.update(
                {name: archive[name] for name in cls._ARRAY_FIELDS if name in archive}
            )
        return cls(**values)

    def __post_init__(self):
        if not np.isfinite(
            [self.annot_hsq, self.total_hsq, self.annot_het, self.total_het]
        ).all():
            raise ValueError("enrichment raw moments must be finite")
        counts = (self.annot_count, self.total_count)
        if (counts[0] is None) != (counts[1] is None):
            raise ValueError("annotation and total counts must be supplied together")
        if counts[0] is not None and not np.isfinite(counts).all():
            raise ValueError("enrichment count moments must be finite")
        count_deletes = (self.annot_count_jk, self.total_count_jk)
        if (count_deletes[0] is None) != (count_deletes[1] is None):
            raise ValueError("delete-block count moments must be supplied together")
        if count_deletes[0] is not None:
            if counts[0] is None or self.annot_hsq_jk is None:
                raise ValueError(
                    "delete-block counts require point counts and hsq deletes"
                )
            for name in ("annot_count_jk", "total_count_jk"):
                array = np.asarray(getattr(self, name), dtype=float)
                if (
                    array.ndim != 1
                    or array.shape != np.shape(self.annot_hsq_jk)
                    or not np.isfinite(array).all()
                ):
                    raise ValueError("delete-block counts must align with hsq deletes")
                object.__setattr__(self, name, array)
        delete = (
            self.annot_hsq_jk,
            self.total_hsq_jk,
            self.annot_het_jk,
            self.total_het_jk,
        )
        if any(value is None for value in delete) and not all(
            value is None for value in delete
        ):
            raise ValueError("delete-block enrichment moments are incomplete")
        if delete[0] is None:
            return
        arrays = tuple(np.asarray(value, dtype=float) for value in delete)
        if (
            any(value.ndim != 1 for value in arrays)
            or len({value.shape for value in arrays}) != 1
        ):
            raise ValueError("delete-block enrichment moments must be aligned vectors")
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("enrichment delete-block raw moments must be finite")
        for name, value in zip(
            ("annot_hsq_jk", "total_hsq_jk", "annot_het_jk", "total_het_jk"),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, name, value)

    def _denominators(self, normalization: str):
        if normalization not in ("het", "count"):
            raise ValueError("normalization must be 'het' or 'count'")
        annotated = getattr(self, f"annot_{normalization}")
        total = getattr(self, f"total_{normalization}")
        if annotated is None or total is None:
            raise ValueError(
                f"{normalization} normalization requires denominator moments"
            )
        return annotated, total

    def enrichment(self, normalization: str = "het") -> float:
        """Form enrichment using heterozygosity or annotation-weighted counts."""
        annotated, total = self._denominators(normalization)
        if annotated == 0 or self.total_hsq == 0 or total == 0:
            return np.nan
        return (self.annot_hsq / annotated) / (self.total_hsq / total)

    @property
    def n_jk_blocks(self) -> int:
        return 0 if self.annot_hsq_jk is None else len(self.annot_hsq_jk)

    def enrichment_jk(self, normalization: str = "het") -> np.ndarray | None:
        """Form each delete-block ratio with its corresponding denominator."""
        self._denominators(normalization)
        if self.annot_hsq_jk is None:
            return None
        annotated = getattr(self, f"annot_{normalization}_jk")
        total = getattr(self, f"total_{normalization}_jk")
        if annotated is None or total is None:
            raise ValueError(
                f"{normalization} normalization requires delete-block denominators"
            )
        valid = (annotated != 0) & (self.total_hsq_jk != 0) & (total != 0)
        values = np.full_like(self.annot_hsq_jk, np.nan)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            values[valid] = (self.annot_hsq_jk[valid] / annotated[valid]) / (
                self.total_hsq_jk[valid] / total[valid]
            )
        values[~np.isfinite(values)] = np.nan
        return values

    @property
    def annot_hsq_se(self) -> float:
        if self.annot_hsq_jk is not None:
            return float(fitting.jackknife_se(self.annot_hsq_jk[:, None])[0])
        if self.annot_hsq_variance is None:
            return np.nan
        return float(np.sqrt(max(0.0, self.annot_hsq_variance)))

    @property
    def total_hsq_se(self) -> float:
        if self.total_hsq_jk is not None:
            return float(fitting.jackknife_se(self.total_hsq_jk[:, None])[0])
        if self.total_hsq_variance is None:
            return np.nan
        return float(np.sqrt(max(0.0, self.total_hsq_variance)))

    def enrichment_se(self, normalization: str = "het") -> float:
        if not np.isfinite(self.enrichment(normalization)):
            return np.nan
        deletes = self.enrichment_jk(normalization)
        if deletes is not None:
            return float(fitting.jackknife_se(deletes[:, None])[0])
        if (
            self.annot_hsq_variance is None
            or self.total_hsq_variance is None
            or self.annot_total_hsq_covariance is None
            or self.total_hsq == 0
        ):
            return np.nan
        annotated, total = self._denominators(normalization)
        scale = total / annotated
        variance = scale**2 * (
            self.annot_hsq_variance / self.total_hsq**2
            + self.annot_hsq**2 * self.total_hsq_variance / self.total_hsq**4
            - 2.0 * self.annot_hsq * self.annot_total_hsq_covariance / self.total_hsq**3
        )
        tolerance = 1e-12 * max(1.0, abs(variance))
        if variance < -tolerance:
            raise ValueError(f"materially negative enrichment variance: {variance}")
        return float(np.sqrt(max(0.0, variance)))


def _random_enrich_result(stats, result) -> EnrichResult:
    """Contract retained attribution weights with the completed point/JK fit."""
    n_blocks = result.n_jk_blocks
    for key in ("annot_hsq_weights", "total_hsq_weights", "annot_het", "total_het"):
        expected = (
            (n_blocks, len(result.kernel_coef), result.gene_coef.shape[0])
            if key.endswith("weights")
            else (n_blocks,)
        )
        if np.shape(stats[key]) != expected or not np.isfinite(stats[key]).all():
            raise ValueError(
                f"enrichment attribution {key} must be finite with shape {expected}"
            )
    delete_one = n_blocks > 1
    attributed = {}
    for name, weights in (
        ("annot", stats["annot_hsq_weights"]),
        ("total", stats["total_hsq_weights"]),
    ):
        total = weights.sum(axis=0)
        attributed[name] = float(
            np.einsum("k,ka,a->", result.kernel_coef, total, result.gene_coef[:, 0])
        )
        delete = total[None] - weights if delete_one else total[None]
        attributed[name + "_jk"] = np.einsum(
            "bk,bka,ba->b",
            result.kernel_coef_jk,
            delete,
            result.gene_coef_jk[:, :, 0],
        )
    annot_hsq, total_hsq = attributed["annot"], attributed["total"]
    annot_hsq_jk, total_hsq_jk = attributed["annot_jk"], attributed["total_jk"]
    annot_het, total_het = (
        float(stats["annot_het"].sum()),
        float(stats["total_het"].sum()),
    )
    annot_het_jk = (
        annot_het - stats["annot_het"] if delete_one else np.array([annot_het])
    )
    total_het_jk = (
        total_het - stats["total_het"] if delete_one else np.array([total_het])
    )

    return EnrichResult(
        annot_hsq=annot_hsq,
        total_hsq=total_hsq,
        annot_het=annot_het,
        total_het=total_het,
        annotation=stats["annotation"],
        n_genes=result.n_genes,
        n_variants=result.n_variants,
        annot_hsq_jk=annot_hsq_jk,
        total_hsq_jk=total_hsq_jk,
        annot_het_jk=annot_het_jk,
        total_het_jk=total_het_jk,
    )


def _fixed_enrich_result(model, result, enrichment_col):
    """Assemble fixed enrichment from moments retained by the fixed fit."""
    per_gene = result.burden_per_gene
    if per_gene is None:
        raise ValueError("fixed fit did not retain enrichment moments")
    genes = result.per_gene.filter(pl.col("rank") < result.sample_size)[
        "gene"
    ].to_list()
    if not genes:
        raise ValueError("enrichment requires at least one estimable fixed gene")
    per_gene = per_gene.filter(pl.col("gene").is_in(genes))
    annotated, annotated_se = fitting.fixed_hsq_from_quadforms(
        per_gene["quadform"].to_numpy(),
        per_gene["rank"].to_numpy(),
        result.sample_size,
    )
    total = result.hsq
    annot_het, total_het = _annotation_heterozygosity(model, enrichment_col, genes)
    raw_annotated_variance = float(per_gene["variance"].sum())
    raw_total_variance = float(per_gene["total_variance"].sum())
    annotated_variance = annotated_se**2
    total_variance = result.hsq_se**2
    annotated_total_covariance = float(per_gene["total_covariance"].sum())
    if raw_annotated_variance > 0.0 and raw_total_variance > 0.0:
        annotated_total_covariance *= np.sqrt(
            annotated_variance
            * total_variance
            / (raw_annotated_variance * raw_total_variance)
        )

    return EnrichResult(
        annot_hsq=annotated,
        total_hsq=total,
        annot_het=annot_het,
        total_het=total_het,
        annotation=enrichment_col,
        n_genes=len(genes),
        n_variants=sum(model.dataset.variants_per_gene[gene] for gene in genes),
        annot_hsq_variance=annotated_variance,
        total_hsq_variance=total_variance,
        annot_total_hsq_covariance=annotated_total_covariance,
    )


def _validate_burden_annotation(model, enrichment_col):
    """Validate finite raw annotation values and the base burden design."""
    if len(model.layout.kernel_group_names) != 1:
        raise ValueError("enrichment requires a single kernel group")
    annotation = model.dataset.variant_values(enrichment_col)
    if not np.isfinite(annotation).all():
        raise ValueError("burden annotation must be finite")
    if not model.layout.n_burden:
        raise ValueError("burden attribution needs at least one burden column")


def _nested_burden_designs(model, gene, enrichment_col):
    """Return the total and annotated fixed-enrichment designs.

    For a partitioned burden, ``X`` has one independently fitted column per
    logical burden column and within-gene group. The full gene LDM still enters
    the quadratic form, so LD between groups is retained.
    """
    burden_wgts = model.burden_wgts(gene)
    partition = model.burden_partition_codes(gene)
    if partition is not None:
        membership = partition[:, None] == np.arange(partition.max() + 1)
        burden_wgts = (burden_wgts[:, :, None] * membership[:, None, :]).reshape(
            len(partition), -1
        )
    annotation = model.dataset.variant_values(enrichment_col, gene)[:, None]
    annotated = burden_wgts * annotation
    return np.column_stack((burden_wgts, annotated)), annotated


def pool_enrichment(results, *, weights=None, independent=False):
    """Pool additive enrichment moments, then form the enrichment ratio."""
    if isinstance(results, Mapping):
        labels = tuple(results)
        values = tuple(results.values())
    else:
        values = tuple(results)
        labels = tuple(map(str, range(len(values))))
    if not values:
        raise ValueError("results must contain at least one enrichment result")
    if len({result.annotation for result in values}) != 1:
        raise ValueError("all enrichment results must use the same annotation")
    if weights is None:
        point_weights = np.ones(len(values))
    elif isinstance(weights, Mapping):
        if set(weights) != set(labels):
            raise ValueError("weight keys must match result labels")
        point_weights = np.asarray([weights[label] for label in labels], dtype=float)
    else:
        point_weights = np.asarray(weights, dtype=float)
    if (
        point_weights.shape != (len(values),)
        or not np.isfinite(point_weights).all()
        or np.any(point_weights < 0)
        or point_weights.sum() <= 0
    ):
        raise ValueError("weights must be finite, nonnegative, and have positive sum")
    point_weights = point_weights / point_weights.sum()

    analytic_names = (
        "annot_hsq_variance",
        "total_hsq_variance",
        "annot_total_hsq_covariance",
    )
    analytic = dict.fromkeys(analytic_names)
    if independent and all(
        getattr(result, name) is not None
        for result in values
        for name in analytic_names
    ):
        squared = point_weights**2
        analytic = {
            name: float(
                squared
                @ np.asarray([getattr(result, name) for result in values], dtype=float)
            )
            for name in analytic_names
        }

    delete_names = (
        "annot_hsq_jk",
        "total_hsq_jk",
        "annot_het_jk",
        "total_het_jk",
    )
    block_counts = {
        result.n_jk_blocks for result in values if result.annot_hsq_jk is not None
    }
    if len(block_counts) > 1:
        raise ValueError("delete-block enrichment moments must be aligned")

    count_names = ("annot_count", "total_count")
    counts = {}
    if all(result.annot_count is not None for result in values):
        counts = {
            name: float(
                point_weights @ np.asarray([getattr(result, name) for result in values])
            )
            for name in count_names
        }
        if all(result.annot_count_jk is not None for result in values):
            delete_names += ("annot_count_jk", "total_count_jk")

    deleted = dict.fromkeys(delete_names)
    if all(result.annot_hsq_jk is not None for result in values):
        deleted = {
            name: point_weights @ np.stack([getattr(result, name) for result in values])
            for name in delete_names
        }

    return EnrichResult(
        **{
            name: float(
                point_weights @ np.asarray([getattr(result, name) for result in values])
            )
            for name in ("annot_hsq", "total_hsq", "annot_het", "total_het")
        },
        annotation=values[0].annotation,
        n_genes=sum(result.n_genes for result in values),
        n_variants=sum(result.n_variants for result in values),
        **counts,
        **analytic,
        **deleted,
    )
