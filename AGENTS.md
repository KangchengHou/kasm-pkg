# kasm-pkg

Python library for the kernel allelic series model (KASM); distributed on PyPI
as `kasm-pkg`, imported as `kasm`. Keep analysis workflows out of this repository.
`DOCS.md` documents concepts, usage, and the public API; keep its examples
runnable against `tests/data/n50/` and update it with interface changes.

## Engineering

- Use the simplest durable design. Remove obsolete paths; no backward-compatibility
  code/tests, schema versions, or manifest-based provenance.
- Prefer reliable established libraries. Check existing dependency APIs and types
  before reimplementing capabilities or adding packages.
- Use Python, Polars tables, and NumPy/SciPy numerics.
  Numerical computations take NumPy arrays and SciPy CSR buffers, not pandas, Dataset, or Model.
- Run deterministically with one BLAS/OpenMP thread, including tests and benchmarks.

## Statistical contracts

- Covar inference is random-only.
- Enrichment extensions require a single kernel group; reject multi-group inputs
  before computation and assign extensions directly to the sole group.
- Outer nonconvergence is diagnostic-only. Keep finite final point/JK iterates and
  compute SEs from valid raw JK estimates. Never discard estimates, mask SEs, or
  increase budgets solely for nonconvergence. Missing/non-finite outputs, invalid
  geometry, rank failures, and inner-solver failures still raise.
- Dataset is immutable and Polars-native, with explicit `variant` and `gene`
  columns. Canonical `[chr]:[pos]:[ref]:[alt]` IDs define genomic ordering and LD alignment.
- Missing genes use identity LD with a warning, not an error; off-panel variants
  get identity diagonals. LD content identity belongs in stats provenance.
- Exons are within-gene burden groups under the parent gene LDM;
  genes remain the analysis and jackknife units.
- Random JK preparation requires explicit `exclude_genes`; prepared fits reuse
  stored exclusions. Fixed fits require explicit `include_genes`.
- Simulations transport unscaled per-allele effects across SE/MAF scales, not h².

## Development

- Use `uv run python ...` and `uv run --extra dev pytest ...`; keep each test under one minute.
- Validate every change with `uv run --extra dev pytest` (committed 50-gene
  fixture, ~10 s). Before handing off numerical changes, also run
  `uv run --extra dev pytest --full` (local 1000-gene fixture).
- Before validation, format only changed Python files with
  `uv run --extra dev ruff format <files>`.
- Install the hook once: `uv run --extra dev pre-commit install`.
  Review and restage any files it formats on commit.
- Unless asked to, do not commit.

## Terminology

- **covar_indep**: independent kernel D on variants participating in a covar kernel.
- **covar_membership**: binary variant participation in a covar kernel.
- **pair_degree**: each variant's summed pair weight; used for the variance denominator.
- **signif_genes**: significant-gene sets; use burden projection weights for learned one-dimensional weights.
- Use **covar kernel** for K and **covar extension** for its full addition; avoid
  “relation” for these concepts.
- **Burden extension**: an optional augmented burden design with compact moments
  and attribution data owned by the parent sufficient statistics.
- **Covar extension**: one optional covar kernel K and optional covar independent kernel D,
  with compact moments, matched-reference weights, and support owned by the parent.
- **ExtSuffStats**: one extension’s augmented layout, added moments, and typed
  payload, owned by either per-gene or JK parent statistics.
- **EnrichResult**: additive annotation readout; random diagonal attribution and
  fixed nested projections retain distinct estimands.
- **ExtraEnrichCols**: typed enrichment-extension payload owned by parent stats;
  finite continuous and signed annotations use the same path as binary annotations.
