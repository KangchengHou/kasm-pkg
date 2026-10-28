# KASM (Kernel Allelic Series Model)

[![Tests](https://img.shields.io/github/actions/workflow/status/KangchengHou/kasm-pkg/tests.yml?branch=main&label=Tests&logo=github&style=flat-square)](https://github.com/KangchengHou/kasm-pkg/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/kasm-pkg?label=PyPI&logo=pypi&logoColor=white&style=flat-square)](https://pypi.org/project/kasm-pkg/)
[![Python](https://img.shields.io/pypi/pyversions/kasm-pkg?label=Python&logo=python&logoColor=white&style=flat-square)](https://pypi.org/project/kasm-pkg/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://github.com/KangchengHou/kasm-pkg/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-DOCS.md-blue?style=flat-square)](https://github.com/KangchengHou/kasm-pkg/blob/main/DOCS.md)

KASM is a method that integrates functional annotations to form a covariance
kernel describing how allelic effects within a gene covary. Using data from
exome association studies, KASM estimates the heritability of rare protein
coding variants, its enrichment in structural and functional annotations, and
the concordance of effects between pairs of variants. [Our paper](https://www.medrxiv.org/content/10.64898/2026.10.16.26365771v1)
describes the methodology and applications in detail.

## Installation

```bash
pip install kasm-pkg   # or: uv add kasm-pkg
```

KASM requires Python >= 3.12 and depends on NumPy, SciPy, Polars (>= 1.30) and Clarabel (>= 0.11.1). It is tested on Linux and macOS with Python 3.12–3.14.

## Quick start

Here is an example of using KASM to estimate heritability from exome-wide
association statistics. KASM takes variant-level association statistics: one
row per variant, with a canonical `chr:pos:ref:alt` ID, its gene, the effect
estimate and SE, allele frequency, and functional annotations.
Within-gene LD comes from an LD matrix file passed to `Dataset` as `ldm`;
without one, as in this toy example, KASM assumes no LD between variants.

```python
import kasm
import polars as pl

variant_df = pl.DataFrame(
    {
        "variant": [f"1:{pos}:A:G" for pos in range(100, 1000, 100)],
        "gene": ["G1"] * 3 + ["G2"] * 3 + ["G3"] * 3,
        "beta": [-0.12, 0.19, 0.15, 0.01, 0.17, 0.05, 0.15, 0.25, 0.22],
        "se": [0.10, 0.12, 0.09, 0.11, 0.10, 0.13, 0.08, 0.10, 0.12],
        "af": [2e-4, 5e-5, 3e-4, 1e-4, 2e-4, 4e-5, 5e-4, 2e-4, 6e-5],
        "constant": 1.0,
        "score1": [0.9, 0.1, 0.0, 0.9, 0.1, 0.0, 0.1, 0.8, 0.9],
        "score2": [0.8, 0.3, 0.1, 0.7, 0.2, 0.0, 0.3, 0.6, 0.9],
    }
)
print(variant_df)

dataset = kasm.Dataset(variant_df, beta_col="beta", se_col="se", sample_size=300_000)
model = kasm.Model(
    dataset,
    kasm.KernelOptions(
        maf_col="af",
        burden_cols=["constant", "score1", "score2"],
        overdispersion_cols=["constant"],
    ),
).compute_suffstats()
fit = kasm.fit_random_effects(model, exclude_genes=[], n_jk_blocks=3)
print(f"heritability: {fit.hsq:.2e} (SE {fit.hsq_se:.2e})")
print(fit.components.select("kernel", "kernel_coef", "kernel_coef_se"))
```

The example runs in ~1 second on a standard computer. Expected output:

<details>
<summary>Show output</summary>

```text
shape: (9, 8)
┌───────────┬──────┬───────┬──────┬─────────┬──────────┬────────┬────────┐
│ variant   ┆ gene ┆ beta  ┆ se   ┆ af      ┆ constant ┆ score1 ┆ score2 │
│ ---       ┆ ---  ┆ ---   ┆ ---  ┆ ---     ┆ ---      ┆ ---    ┆ ---    │
│ str       ┆ str  ┆ f64   ┆ f64  ┆ f64     ┆ f64      ┆ f64    ┆ f64    │
╞═══════════╪══════╪═══════╪══════╪═════════╪══════════╪════════╪════════╡
│ 1:100:A:G ┆ G1   ┆ -0.12 ┆ 0.1  ┆ 0.0002  ┆ 1.0      ┆ 0.9    ┆ 0.8    │
│ 1:200:A:G ┆ G1   ┆ 0.19  ┆ 0.12 ┆ 0.00005 ┆ 1.0      ┆ 0.1    ┆ 0.3    │
│ 1:300:A:G ┆ G1   ┆ 0.15  ┆ 0.09 ┆ 0.0003  ┆ 1.0      ┆ 0.0    ┆ 0.1    │
│ 1:400:A:G ┆ G2   ┆ 0.01  ┆ 0.11 ┆ 0.0001  ┆ 1.0      ┆ 0.9    ┆ 0.7    │
│ 1:500:A:G ┆ G2   ┆ 0.17  ┆ 0.1  ┆ 0.0002  ┆ 1.0      ┆ 0.1    ┆ 0.2    │
│ 1:600:A:G ┆ G2   ┆ 0.05  ┆ 0.13 ┆ 0.00004 ┆ 1.0      ┆ 0.0    ┆ 0.0    │
│ 1:700:A:G ┆ G3   ┆ 0.15  ┆ 0.08 ┆ 0.0005  ┆ 1.0      ┆ 0.1    ┆ 0.3    │
│ 1:800:A:G ┆ G3   ┆ 0.25  ┆ 0.1  ┆ 0.0002  ┆ 1.0      ┆ 0.8    ┆ 0.6    │
│ 1:900:A:G ┆ G3   ┆ 0.22  ┆ 0.12 ┆ 0.00006 ┆ 1.0      ┆ 0.9    ┆ 0.9    │
└───────────┴──────┴───────┴──────┴─────────┴──────────┴────────┴────────┘
heritability: 7.95e-05 (SE 5.08e-05)
shape: (7, 3)
┌───────────────────┬─────────────┬────────────────┐
│ kernel            ┆ kernel_coef ┆ kernel_coef_se │
│ ---               ┆ ---         ┆ ---            │
│ str               ┆ f64         ┆ f64            │
╞═══════════════════╪═════════════╪════════════════╡
│ B:constant        ┆ 0.009453    ┆ 0.051091       │
│ B:score1          ┆ -0.199445   ┆ 0.41096        │
│ B:score2          ┆ -0.501225   ┆ 0.838432       │
│ B:constant*score1 ┆ -0.09807    ┆ 0.094232       │
│ B:constant*score2 ┆ 0.101415    ┆ 0.177126       │
│ B:score1*score2   ┆ 0.368544    ┆ 0.538677       │
│ O:constant        ┆ -0.010676   ┆ 0.016743       │
└───────────────────┴─────────────┴────────────────┘
```

</details>

## Usage

To run KASM on your own data:

1. **Input.** A Polars table with one row per variant: a `chr:pos:ref:alt`
   variant ID, its gene, the effect estimate and its SE, allele frequency, and
   annotation columns used as burden weights. Wrap it in `kasm.Dataset`, with
   the GWAS sample size and, optionally, per-gene LD matrices via `ldm`.
2. **Kernel options.** Declare kernels with `kasm.KernelOptions` (burden columns, MAF
   bins, overdispersion, independent and within-gene kernels) and bind them
   with `kasm.Model(dataset, options).compute_suffstats()`.
3. **Fit.** `kasm.fit_random_effects` estimates heritability, kernel
   coefficients and jackknife SEs; `kasm.fit_fixed_effects` fits significant
   genes as fixed effects. Enrichment and effect concordance (covar) build on
   these fits.

The [documentation](https://github.com/KangchengHou/kasm-pkg/blob/main/DOCS.md)
covers each step, the LD matrix format, and examples on a 50-gene dataset.

## Development

Contributions are welcome: open an
[issue](https://github.com/KangchengHou/kasm-pkg/issues) for bugs and
questions, or a pull request for fixes and features.

```bash
git clone https://github.com/KangchengHou/kasm-pkg.git
cd kasm-pkg
uv run --extra dev pre-commit install
uv run --extra dev pytest
```
