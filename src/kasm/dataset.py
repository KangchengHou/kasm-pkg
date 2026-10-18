"""One trait's gene-grouped variant table with aligned per-gene LD."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from kasm._logging import logger

if TYPE_CHECKING:
    from kasm.ldm import LDM, LDMLoader

INDEX_NAMES = ("variant", "gene")
_CHROM_RANK = {str(i): i for i in range(1, 23)} | {
    "X": 23,
    "Y": 24,
    "MT": 25,
}


def _variant_series(variant_ids) -> pl.Series:
    if isinstance(variant_ids, pl.Series):
        ids = variant_ids.rename("variant")
    else:
        try:
            ids = pl.Series("variant", variant_ids)
        except (TypeError, ValueError) as error:
            raise ValueError("variant IDs must be strings") from error
    if ids.is_empty():
        return ids.cast(pl.String)
    if ids.dtype != pl.String:
        raise ValueError("variant IDs must be strings")
    return ids


def parse_variant_ids(variant_ids) -> pl.DataFrame:
    """Validate and split IDs, including their canonical string spelling."""
    ids = _variant_series(variant_ids)
    fields = ids.str.split_exact(":", 3).struct.unnest()
    chrom = fields["field_0"]
    parsed = fields.with_columns(
        chrom.replace_strict(_CHROM_RANK, default=None, return_dtype=pl.Int64).alias(
            "chrom_rank"
        ),
        pl.col("field_1").cast(pl.Int64, strict=False).alias("pos"),
        pl.col("field_2").alias("ref"),
        pl.col("field_3").alias("alt"),
    )
    bad = parsed.select(
        pl.any_horizontal(
            ids.is_null(),
            ids.str.count_matches(":") != 3,
            pl.col("chrom_rank").is_null(),
            pl.col("pos").is_null(),
            ~pl.col("field_1").str.contains(r"^[1-9][0-9]*$"),
            pl.col("ref").is_null() | (pl.col("ref") == ""),
            pl.col("alt").is_null() | (pl.col("alt") == ""),
        ).fill_null(True)
    ).to_series()
    if bad.any():
        examples = ids.filter(bad).head(5).to_list()
        raise ValueError(
            f"{int(bad.sum())} variant IDs are not canonical "
            f"[chr]:[pos]:[ref]:[alt]: {examples}"
        )
    return parsed.select(
        "chrom_rank",
        "pos",
        "ref",
        "alt",
        pl.concat_str([chrom, "pos", "ref", "alt"], separator=":").alias("variant"),
    )


def sort_variant_df(
    frame: pl.DataFrame, *by: str, gene_col: str | None = None
) -> pl.DataFrame:
    """Sort variants canonically, optionally grouping genes by genomic position."""
    if not isinstance(frame, pl.DataFrame) or "variant" not in frame.columns:
        raise ValueError("frame must be a Polars DataFrame with a variant column")
    if gene_col is not None:
        if by:
            raise ValueError("gene_col cannot be combined with leading sort keys")
        if gene_col not in frame.columns:
            raise ValueError(f"frame must contain {gene_col!r}")
        canonical = sort_variant_df(frame.sort(gene_col, maintain_order=True))
        rank = "__kasm_gene_rank"
        while rank in frame.columns:
            rank = f"_{rank}"
        first = canonical.select(gene_col, "variant").unique(
            subset=gene_col, keep="first", maintain_order=True
        )
        genes = first.with_row_index(rank).select(gene_col, rank)
        return sort_variant_df(
            frame.join(genes, on=gene_col, how="left", validate="m:1"), rank
        ).drop(rank)
    parsed = parse_variant_ids(frame["variant"])
    temporary = []
    for suffix in ("chrom_rank", "pos", "ref", "alt"):
        name = f"__kasm_sort_{suffix}"
        while name in frame.columns or name in temporary:
            name = f"_{name}"
        temporary.append(name)
    return (
        frame.with_columns(
            *(
                parsed[column].alias(name)
                for column, name in zip(
                    ("chrom_rank", "pos", "ref", "alt"), temporary, strict=True
                )
            ),
        )
        .sort(*by, *temporary, maintain_order=True)
        .drop(*temporary)
    )


class Dataset:
    """One trait's variant-level summary statistics, annotations, and LD.

    Parameters
    ----------
    variant_df : polars.DataFrame
        Contains explicit ``variant`` and ``gene`` columns. Variant IDs are
        unique and canonical ``[chr]:[pos]:[ref]:[alt]``. Rows are retained in
        their supplied order and each gene must occupy exactly one contiguous
        block. Gene order is the order of first appearance.
    gene_df : polars.DataFrame or None
        Gene-level covariates keyed by ``gene`` and aligned by name rather than
        by row order. Genes absent from this table are removed from
        ``variant_df`` without reordering the retained rows, and extra rows are
        ignored. ``None`` means no gene-level covariates.
    beta_col, se_col : str
        Marginal effect estimate and its standard error.
    sample_size : float
        GWAS sample size N.
    ldm : str, Path, or None
        Per-gene LD panel. ``None`` means LD-free (identity correlation),
        which exists so that estimators can be checked against hand
        computations.
    """

    def __init__(
        self,
        variant_df: pl.DataFrame,
        *,
        gene_df: pl.DataFrame | None = None,
        beta_col: str,
        se_col: str,
        sample_size: float,
        ldm: str | Path | None = None,
    ):
        variant_df = _prepare_variant_df(variant_df)
        for col in (beta_col, se_col):
            if col not in variant_df.columns:
                raise KeyError(f"variant_df has no column {col!r}")
        if not np.isfinite(sample_size) or sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {sample_size!r}")

        variant_df, aligned_gene_df = _align_gene_df(variant_df, gene_df)

        self.beta_col = beta_col
        self.se_col = se_col
        self._sample_size = float(sample_size)
        self._ldm_source = None if ldm is None else str(ldm)
        self._ldm_loader: LDMLoader | None = None

        for col in (beta_col, se_col):
            if variant_df.schema[col] != pl.Float64:
                raise ValueError(f"{col!r} must be numeric")
        se = variant_df[se_col]
        invalid_numeric = [
            column
            for column, dtype in variant_df.schema.items()
            if dtype == pl.Float64 and _invalid_numeric(variant_df[column])
        ]
        if se_col in invalid_numeric or (se <= 0).any():
            raise ValueError(f"{se_col!r} must be positive and finite")
        if beta_col in invalid_numeric:
            raise ValueError(f"{beta_col!r} must be finite")
        if invalid_numeric:
            raise ValueError(
                "variant_df numeric columns must contain only finite values; "
                f"invalid: {invalid_numeric}"
            )

        self.n_variants = len(variant_df)
        gene_labels = np.asarray(variant_df["gene"].to_list(), dtype=object)
        starts = np.concatenate(
            [[0], np.flatnonzero(gene_labels[:-1] != gene_labels[1:]) + 1]
        )
        ends = np.concatenate([starts[1:], [self.n_variants]])
        genes = tuple(gene_labels[start] for start in starts)
        if len(set(genes)) != len(genes):
            raise ValueError("variant_df rows for each gene must be contiguous")
        gene_slices = {
            gene: slice(int(start), int(end))
            for gene, start, end in zip(genes, starts, ends)
        }

        self._variant_df = variant_df
        self._gene_df = aligned_gene_df
        self._genes = genes
        self._gene_slices = MappingProxyType(gene_slices)
        self._variants_per_gene = MappingProxyType(
            {gene: rows.stop - rows.start for gene, rows in gene_slices.items()}
        )

    # -- column access -------------------------------------------------------

    def variant_df(self, gene: str | None = None) -> pl.DataFrame:
        """Independent Polars snapshot, optionally restricted to one gene."""
        if gene is None:
            return self._variant_df.clone()
        rows = self._gene_slices[gene]
        return self._variant_df.slice(rows.start, rows.stop - rows.start)

    def gene_df(self) -> pl.DataFrame:
        """Independent Polars gene table in Dataset order."""
        return self._gene_df.clone()

    def variant_values(self, columns, gene: str | None = None):
        """Float64 variant column values, for one gene or for all variants.

        A single column name gives a 1-d array; a list gives a 2-d array with
        one column per name, in the order requested. Results are read-only
        views on the dataset's buffers, so a reducer that needs to write must
        copy. This keeps the contract independent of the input dtypes.
        """
        rows = slice(None) if gene is None else self._gene_slices[gene]
        return _numeric_values(self._variant_df, columns, rows, "variant_df")

    def gene_values(self, columns):
        """Float64 gene covariate values in dataset gene order. Read-only."""
        return _numeric_values(self._gene_df, columns, slice(None), "gene_df")

    @property
    def beta(self) -> np.ndarray:
        return self.variant_values(self.beta_col)

    @property
    def se(self) -> np.ndarray:
        return self.variant_values(self.se_col)

    @property
    def sample_size(self) -> float:
        return self._sample_size

    @property
    def ldm_source(self) -> str | None:
        return self._ldm_source

    def gene_slice(self, gene: str) -> slice:
        """Row slice of *gene* in the supplied gene-major ordering."""
        return self._gene_slices[gene]

    def _gene_variant_ids(self, gene: str) -> pl.Series:
        rows = self._gene_slices[gene]
        return self._variant_df["variant"].slice(rows.start, rows.stop - rows.start)

    @property
    def genes(self) -> list[str]:
        return list(self._genes)

    @property
    def n_genes(self) -> int:
        return len(self._genes)

    @property
    def variants_per_gene(self) -> Mapping[str, int]:
        return self._variants_per_gene

    # -- LD ------------------------------------------------------------------

    def _get_ldm_loader(self) -> LDMLoader:
        if self._ldm_loader is None:
            from kasm.ldm import LDMLoader

            self._ldm_loader = LDMLoader(self.ldm_source)
        return self._ldm_loader

    def load_ldm(self, gene: str) -> LDM:
        """Correlation-scale LD for *gene*, aligned to its variant order."""
        from kasm.ldm import LDM

        if gene not in self._gene_slices:
            raise KeyError(f"gene {gene!r} is not in this dataset")
        if self.ldm_source is None:
            return LDM.identity(self.variants_per_gene[gene])
        loader = self._get_ldm_loader()
        if gene not in loader:
            logger.warning(
                "event=ldm_gene_missing gene=%s source=%s fallback=identity",
                gene,
                self.ldm_source,
            )
            return LDM.identity(self.variants_per_gene[gene])
        return loader.load_gene_ldm(gene, self._gene_variant_ids(gene), _canonical=True)

    def close(self) -> None:
        """Release the LD panel handle."""
        if self._ldm_loader is not None:
            self._ldm_loader.close()
            self._ldm_loader = None

    def __enter__(self) -> "Dataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Dataset(n_variants={self.n_variants}, n_genes={self.n_genes}, "
            f"sample_size={self.sample_size:g}, ldm={self.ldm_source!r})"
        )


def _numeric_values(frame, columns, rows, table_name):
    """Expose validated Float64 buffers; only multi-column access allocates."""
    names = [columns] if isinstance(columns, str) else columns
    schema = frame.schema
    for column in names:
        if schema.get(column) != pl.Float64:
            available = sorted(
                name for name, dtype in schema.items() if dtype == pl.Float64
            )
            raise KeyError(
                f"{table_name} has no numeric column {column!r}; available: {available}"
            )
    if isinstance(columns, str):
        return frame[columns].to_numpy(writable=False, allow_copy=False)[rows]
    values = np.column_stack(
        [
            frame[column].to_numpy(writable=False, allow_copy=False)[rows]
            for column in names
        ]
    )
    values.flags.writeable = False
    return values


def _numeric_dtype(dtype: pl.DataType) -> bool:
    return dtype.is_numeric() or dtype == pl.Boolean


def _invalid_numeric(values: pl.Series) -> bool:
    return bool(values.null_count() or not values.is_finite().all())


def _float_series(values: pl.Series) -> pl.Series:
    """Own one contiguous Float64 buffer independent of the caller's input."""
    return pl.Series(
        values.name,
        np.array(values, dtype=np.float64, copy=True, order="C"),
    )


def _prepare_variant_df(variant_df: pl.DataFrame) -> pl.DataFrame:
    """Validate and canonicalize IDs without changing variant-table row order."""
    if isinstance(variant_df, pl.DataFrame):
        frame = variant_df
        missing = set(INDEX_NAMES) - set(frame.columns)
        if missing:
            raise ValueError(
                "Polars variant_df must contain columns "
                f"{INDEX_NAMES}, missing {sorted(missing)}"
            )
        if frame.height == 0:
            raise ValueError("variant_df is empty")
        if frame["variant"].n_unique() != frame.height:
            raise ValueError("variant_df has duplicate variant IDs")
    else:
        raise TypeError("variant_df must be a Polars DataFrame")

    if frame["variant"].dtype != pl.String or frame["gene"].dtype != pl.String:
        raise ValueError("variant and gene labels must be strings")

    # Parsing rejects noncanonical spellings; accepted IDs are unchanged.
    parse_variant_ids(frame["variant"])

    numeric_columns = [
        column for column, dtype in frame.schema.items() if _numeric_dtype(dtype)
    ]
    frame = frame.with_columns(
        [_float_series(frame[column]) for column in numeric_columns]
    ).rechunk()
    return frame


def _align_gene_df(
    variant_df: pl.DataFrame, gene_df: pl.DataFrame | None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply the gene-table inner overlap and align its numeric covariates."""
    genes = variant_df["gene"].unique(maintain_order=True).to_list()
    if gene_df is None:
        logger.info(
            "event=gene_df_alignment gene_df_provided=false "
            "input_genes=%d retained_genes=%d dropped_genes=0 "
            "input_variants=%d retained_variants=%d dropped_variants=0 "
            "extra_gene_df_genes=0",
            len(genes),
            len(genes),
            variant_df.height,
            variant_df.height,
        )
        return variant_df, pl.DataFrame({"gene": genes}).rechunk()
    if isinstance(gene_df, pl.DataFrame):
        frame = gene_df
        if "gene" not in frame.columns:
            raise ValueError("Polars gene_df must contain a 'gene' column")
        if frame["gene"].dtype != pl.String or frame["gene"].null_count():
            raise ValueError("gene_df gene labels must be strings")
        if frame["gene"].n_unique() != frame.height:
            raise ValueError("gene_df has duplicate gene names")
    else:
        raise TypeError("gene_df must be a Polars DataFrame")
    non_numeric = [
        column
        for column, dtype in frame.schema.items()
        if column != "gene" and not _numeric_dtype(dtype)
    ]
    if non_numeric:
        raise ValueError(f"gene_df columns must be numeric; got {non_numeric}")
    order = pl.DataFrame({"gene": genes}).with_row_index("_gene_rank")
    aligned = order.join(frame, on="gene", how="inner", validate="1:1").sort(
        "_gene_rank"
    )
    if aligned.is_empty():
        raise ValueError("variant_df and gene_df have no overlapping genes")
    input_variants = variant_df.height
    variant_df = variant_df.filter(
        pl.col("gene").is_in(aligned["gene"].implode())
    ).rechunk()
    aligned = aligned.drop("_gene_rank")
    aligned = aligned.with_columns(
        [
            _float_series(aligned[column])
            for column in aligned.columns
            if column != "gene"
        ]
    )
    non_finite = [
        column
        for column in aligned.columns
        if column != "gene" and _invalid_numeric(aligned[column])
    ]
    if non_finite:
        raise ValueError(
            "gene_df numeric columns must contain only finite values; "
            f"invalid: {non_finite}"
        )
    logger.info(
        "event=gene_df_alignment gene_df_provided=true "
        "input_genes=%d retained_genes=%d dropped_genes=%d "
        "input_variants=%d retained_variants=%d dropped_variants=%d "
        "extra_gene_df_genes=%d",
        len(genes),
        aligned.height,
        len(genes) - aligned.height,
        input_variants,
        variant_df.height,
        input_variants - variant_df.height,
        frame.height - aligned.height,
    )
    return variant_df, aligned.rechunk()
