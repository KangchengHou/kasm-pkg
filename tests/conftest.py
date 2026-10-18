"""Shared fixtures with two distinct jobs.

**Test dataset** (``test_*``) — the primary substrate. A gene subset of a real
height exome analysis in ``tests/data/n{n}/``, built by ``test_dataset.py`` in
the sibling ``kasm-data`` repository and copied here. The committed 50-gene
subset backs the default run; ``pytest --full`` switches to the local 1000-gene
subset.

**Synthetic** (``synthetic_*``) — only what real data cannot express: a variant
in two genes, LD-free hand computations, a gene small enough for a readable
dense oracle, and panels we can deliberately corrupt. Its LD panel is built
from an explicit diploid genotype matrix, which is also kept, so tests can
check the loader against the genotypes rather than against a restatement of the
loader's own algebra.
"""

from __future__ import annotations

import os

# One BLAS/OpenMP thread per process, set before NumPy loads, keeps every
# run, including each xdist worker, deterministic.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sps

from kasm.dataset import Dataset
from kasm.ldm import _pack_chunk

DATA_DIR = Path(__file__).resolve().parent / "data"
SMALL_FIXTURE_NAME = "n50"
FULL_FIXTURE_NAME = "n1000"

# The real MAF bins used by the KASM analyses.
MAF_BINS = {"AF_ULTRARARE": (0, 1e-5), "AF_RARE": (1e-5, 1e-4), "AF_LF": (1e-4, 1)}
SCORE_COLS = ["am.pathogenicity", "cadd.EsmScoreMissense", "vep.polyphen"]
GENE_ANNOT_COLS = [f"oebin{i}" for i in range(1, 6)]


# ---------------------------------------------------------------------------
# Test dataset
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--full",
        action="store_true",
        help=f"use the {FULL_FIXTURE_NAME} fixture",
    )


def _require_fixture(name: str) -> Path:
    """A real fixture directory; missing real data is a hard failure, not a skip."""
    path = DATA_DIR / name
    if not (path / "variant_df.parquet").exists():
        pytest.fail(
            f"missing {path}; copy it from the sibling kasm-data repository: "
            f"`cp -r ../kasm-data/DATA/test_dataset/{name} tests/data/`",
            pytrace=False,
        )
    return path


@pytest.fixture(scope="session")
def test_data_dir(pytestconfig) -> Path:
    full = pytestconfig.getoption("--full")
    return _require_fixture(FULL_FIXTURE_NAME if full else SMALL_FIXTURE_NAME)


@pytest.fixture(scope="session")
def test_variant_df(test_data_dir) -> pl.DataFrame:
    """Real variant table with explicit variant and gene keys."""
    return pl.read_parquet(test_data_dir / "variant_df.parquet")


@pytest.fixture(scope="session")
def test_signif_genes(test_data_dir) -> list[str]:
    """Reference significant genes present in the subset."""
    return (test_data_dir / "signif_genes.txt").read_text().split()


@pytest.fixture(scope="session")
def test_dataset(test_data_dir) -> Dataset:
    variant_df = pl.read_parquet(test_data_dir / "variant_df.parquet")
    return Dataset(
        variant_df,
        gene_df=pl.read_csv(test_data_dir / "gene_df.tsv", separator="\t"),
        beta_col="BETA",
        se_col="SE",
        sample_size=variant_df["N"].mean(),
        ldm=str(test_data_dir / "exome.ldm"),
    )


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------

# The synthetic panel has 2000 individuals, so its finest resolvable frequency
# is 1/4000. Its MAF bins are scaled to that spectrum; the real bins above
# reach 1e-5 and only the real fixture can populate them.
SYNTHETIC_MAF_BINS = {"rare": (0, 5e-4), "common": (5e-4, 1.0)}
SYNTHETIC_N_INDIVIDUALS = 2000
SYNTHETIC_SAMPLE_SIZE = 50_000

# gene -> unique variant IDs. GENE_B has a single variant, and IDs span three
# chromosomes with an indel among them.
SYNTHETIC_GENE_VARIANTS = {
    "GENE_A": ["1:1000:A:G", "1:2000:C:T", "1:3000:G:A"],
    "GENE_B": ["1:3500:G:A"],
    "GENE_C": ["2:5000:A:T", "2:6000:AT:A", "2:7000:G:C", "2:8000:T:G"],
    "GENE_D": ["X:9000:A:C", "X:9500:C:G"],
    # GENE_E's middle variant is absent from its panel entirely while its
    # flanking variants are strongly correlated, so a misplaced scatter index
    # in the loader's subset/reorder step cannot hide.
    "GENE_E": ["3:1000:A:G", "3:1500:C:T", "3:2000:G:C"],
}
# Panel-only variant, present in GENE_C's LD panel but not in variant_df.
SYNTHETIC_PANEL_EXTRA = "2:6500:C:A"
# Monomorphic in the panel (AF = 0) but polymorphic in the cohort, so the
# variant table still carries a positive frequency for it.
SYNTHETIC_MONOMORPHIC = "X:9500:C:G"
# In variant_df but written to no panel at all.
SYNTHETIC_OFF_PANEL = "3:1500:C:T"

# variant -> (carriers, homozygotes, LD pool). Counts put variants in both
# synthetic MAF bins; the homozygotes are the only feature that distinguishes
# G'G/(2n) from a per-haplotype second moment. Variants sharing a pool are
# correlated.
SYNTHETIC_CARRIERS = {
    "1:1000:A:G": (40, 3, 0),
    "1:2000:C:T": (12, 0, 1),
    "1:3000:G:A": (1, 0, 0),
    "1:3500:G:A": (1, 0, 0),
    "2:5000:A:T": (30, 2, 1),
    "2:6000:AT:A": (18, 0, 0),
    "2:6500:C:A": (25, 0, 1),
    "2:7000:G:C": (1, 0, 0),
    "2:8000:T:G": (9, 0, 1),
    "X:9000:A:C": (14, 0, 0),
    "X:9500:C:G": (0, 0, 0),
    "3:1000:A:G": (60, 4, 2),
    "3:1500:C:T": (20, 0, 2),
    "3:2000:G:C": (55, 1, 2),
}


@dataclass(frozen=True)
class Synthetic:
    """Small synthetic dataset plus the genotypes its LD panel came from."""

    variant_df: pl.DataFrame
    gene_df: pl.DataFrame
    ldm_path: Path
    variant_order: list[str]
    genotypes: np.ndarray  # (n_individuals, len(variant_order)), dosages 0/1/2
    panel_variants: dict[str, list[str]]  # gene -> panel row order

    def allele_frequency(self) -> np.ndarray:
        return self.genotypes.sum(axis=0) / (2 * self.genotypes.shape[0])

    def genotype_correlation(self, variant_ids) -> np.ndarray:
        """Correlation of the given variants computed straight from genotypes."""
        columns = [self.variant_order.index(vid) for vid in variant_ids]
        dosages = self.genotypes[:, columns].astype(np.float64)
        centered = dosages - dosages.mean(axis=0)
        covariance = centered.T @ centered / dosages.shape[0]
        sd = np.sqrt(np.diag(covariance))
        return covariance / np.outer(sd, sd)


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def write_ldm(path: Path, panels: dict[str, tuple[list[str], np.ndarray]]) -> Path:
    """Write a current-format test LDM from explicit genotype matrices."""
    path.parent.mkdir(parents=True, exist_ok=True)
    genes = sorted(panels)
    byte_offsets = [0]
    encoded_ids = []
    with ZipFile(path, "w") as archive:
        for gene in genes:
            ids, genotypes = panels[gene]
            genotypes = np.asarray(genotypes, dtype=np.float64)
            if genotypes.ndim != 2 or genotypes.shape[1] != len(ids):
                raise ValueError(f"invalid genotype matrix for {gene!r}")
            second = genotypes.T @ genotypes / genotypes.shape[0]
            mean = genotypes.mean(axis=0)
            variance = np.diag(second) - mean**2
            usable = variance > 0
            ids = [variant for variant, keep in zip(ids, usable) if keep]
            second = second[np.ix_(usable, usable)]
            mean = mean[usable]
            scaling = 1.0 / np.sqrt(variance[usable])
            matrix = sps.csr_matrix(second * np.outer(scaling, scaling))
            vector = scaling * mean
            archive.writestr(
                _zip_info(f"matrix/{gene}.bin"),
                _pack_chunk(matrix, vector),
                compresslevel=1,
            )
            encoded = "\n".join(ids).encode()
            encoded_ids.append(encoded)
            byte_offsets.append(byte_offsets[-1] + len(encoded))
        metadata = {
            "meta/genes.txt": "\n".join(genes).encode(),
            "meta/gene_byte_offsets.i8": np.asarray(
                byte_offsets, dtype="<i8"
            ).tobytes(),
            "meta/variant_ids.txt": b"".join(encoded_ids),
        }
        for name, payload in metadata.items():
            archive.writestr(_zip_info(name), payload, compresslevel=1)
    return path


def _simulate_genotypes(rng: np.random.Generator) -> tuple[list[str], np.ndarray]:
    """Diploid dosages with block LD.

    Carriers are drawn from two overlapping individual pools, so variants
    sharing a pool are genuinely correlated and variants in different pools are
    nearly independent — sparse, non-diagonal, and PSD because it is a real
    cross-product of real genotypes.
    """
    order = list(SYNTHETIC_CARRIERS)
    n = SYNTHETIC_N_INDIVIDUALS
    genotypes = np.zeros((n, len(order)), dtype=np.int64)
    n_pools = 1 + max(pool for _, _, pool in SYNTHETIC_CARRIERS.values())
    pools = [rng.choice(n, size=120, replace=False) for _ in range(n_pools)]
    for column, variant in enumerate(order):
        n_carriers, n_homozygous, pool = SYNTHETIC_CARRIERS[variant]
        if n_carriers == 0:
            continue
        carriers = rng.choice(pools[pool], size=n_carriers, replace=False)
        genotypes[carriers, column] = 1
        genotypes[carriers[:n_homozygous], column] = 2
    return order, genotypes


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory) -> Synthetic:
    rng = np.random.default_rng(0)
    order, genotypes = _simulate_genotypes(rng)
    af = genotypes.sum(axis=0) / (2 * SYNTHETIC_N_INDIVIDUALS)
    frequency = dict(zip(order, af))

    rows = [
        (variant, gene)
        for gene, variants in SYNTHETIC_GENE_VARIANTS.items()
        for variant in variants
    ]
    n_rows = len(rows)

    # The monomorphic-in-panel variant keeps a positive cohort frequency, as a
    # variant absent from an LD reference panel would in a real analysis.
    variant_af = np.array(
        [
            frequency[variant] if variant != SYNTHETIC_MONOMORPHIC else 4e-4
            for variant, _ in rows
        ]
    )
    score = rng.uniform(0, 1, n_rows)
    variance = 2 * variant_af * (1 - variant_af)
    se = np.exp(rng.normal(scale=0.05, size=n_rows)) / np.sqrt(
        SYNTHETIC_SAMPLE_SIZE * variance
    )
    variant_df = pl.DataFrame(
        {
            "variant": [variant for variant, _ in rows],
            "gene": [gene for _, gene in rows],
            "af": variant_af,
            "constant": np.ones(n_rows),
            "score": score,
            "buried": (rng.random(n_rows) < 0.4).astype(np.float64),
            "variant_class": np.where(score > 0.5, "missense", "synonymous"),
            "exon": np.where(np.arange(n_rows) % 2 == 0, "exon_1", "exon_2"),
            "se": se,
            "beta": 0.01 * score + rng.normal(scale=se),
            # Deliberately mixed dtypes: real sumstats parquets carry float32
            # and integer columns, and the float64 contract must not depend on
            # which the caller supplied.
            "score32": score.astype(np.float32),
            "carriers": np.array(
                [SYNTHETIC_CARRIERS[variant][0] for variant, _ in rows], dtype=np.int64
            ),
        },
    )
    # The caller owns Dataset order: prepare one contiguous genomic gene-major
    # frame while retaining every generated value by its (gene, variant) key.
    prepared_order = ["GENE_A", "GENE_B", "GENE_C", "GENE_E", "GENE_D"]
    variant_df = (
        pl.DataFrame(
            {
                "gene": [
                    gene
                    for gene in prepared_order
                    for _ in SYNTHETIC_GENE_VARIANTS[gene]
                ],
                "variant": [
                    variant
                    for gene in prepared_order
                    for variant in SYNTHETIC_GENE_VARIANTS[gene]
                ],
            }
        )
        .with_row_index("_row")
        .join(variant_df, on=["gene", "variant"], how="left", validate="1:1")
        .sort("_row")
        .drop("_row")
    )

    genes = list(SYNTHETIC_GENE_VARIANTS)
    alternating = np.arange(len(genes)) % 2
    gene_df = pl.DataFrame(
        {
            "gene": genes[::-1],  # row order differs on purpose
            "constraint_low": (alternating == 0).astype(np.float64),
            "constraint_high": (alternating == 1).astype(np.float64),
        },
    )

    # Panels: GENE_C carries an extra variant and shuffled rows, GENE_E omits
    # its middle variant, and every other gene is written in variant_df order.
    panel_variants = {}
    for gene, variants in SYNTHETIC_GENE_VARIANTS.items():
        panel = list(variants)
        if gene == "GENE_C":
            panel = list(rng.permutation(panel + [SYNTHETIC_PANEL_EXTRA]))
        if gene == "GENE_E":
            panel = [vid for vid in panel if vid != SYNTHETIC_OFF_PANEL]
        panel_variants[gene] = panel

    panels = {}
    for gene, panel in panel_variants.items():
        columns = [order.index(vid) for vid in panel]
        panels[gene] = (panel, genotypes[:, columns])

    ldm_path = tmp_path_factory.mktemp("synthetic_ldm") / "exome.ldm"
    write_ldm(ldm_path, panels)
    return Synthetic(
        variant_df=variant_df,
        gene_df=gene_df,
        ldm_path=ldm_path,
        variant_order=order,
        genotypes=genotypes,
        panel_variants=panel_variants,
    )


@pytest.fixture(scope="session")
def synthetic_ldm_path(synthetic) -> Path:
    return synthetic.ldm_path


@pytest.fixture(scope="session")
def synthetic_dataset(synthetic, synthetic_ldm_path) -> Dataset:
    return Dataset(
        synthetic.variant_df,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=SYNTHETIC_SAMPLE_SIZE,
        ldm=str(synthetic_ldm_path),
    )


@pytest.fixture(scope="session")
def synthetic_dataset_no_ld(synthetic) -> Dataset:
    """Same data with identity LD, for hand-checkable computations."""
    return Dataset(
        synthetic.variant_df,
        gene_df=synthetic.gene_df,
        beta_col="beta",
        se_col="se",
        sample_size=SYNTHETIC_SAMPLE_SIZE,
        ldm=None,
    )
