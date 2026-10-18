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
the concordance of effects between pairs of variants. Our paper describes the
methodology and applications in detail.

KASM can be installed with `pip install kasm-pkg` or `uv add kasm-pkg`.

[Quick start](#quick-start) · [Development](#development)

## Quick start

Here is an example of using KASM to estimate heritability from exome-wide
association statistics. KASM takes variant-level association statistics: one
row per variant, with a canonical `chr:pos:ref:alt` ID, its gene, the effect
estimate and SE, allele frequency, and annotations used as burden weights.
Within-gene LD comes from an LD matrix file passed to `Dataset` as `ldm`;
without one, as in this toy example, KASM assumes no LD between variants.

```python
import kasm
import polars as pl

variant_df = pl.DataFrame(
    {
        "variant": [f"1:{pos}:A:G" for pos in range(100, 1000, 100)],
        "gene": ["G1"] * 3 + ["G2"] * 3 + ["G3"] * 3,
        "beta": [-0.12, 0.19, -0.15, 0.01, 0.17, -0.05, 0.15, 0.25, 0.22],
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

See more examples in the [documentation](https://github.com/KangchengHou/kasm-pkg/blob/main/DOCS.md).

## Development

Contributions are welcome: open an
[issue](https://github.com/KangchengHou/kasm-pkg/issues) for bugs and
questions, or a pull request for fixes and features. Development uses
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/KangchengHou/kasm-pkg.git
cd kasm-pkg
uv run --extra dev pre-commit install
uv run --extra dev pytest
```
