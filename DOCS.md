# KASM documentation

KASM, the kernel allelic series model, estimates how rare coding variants
contribute to a trait's heritability, using exome association summary
statistics, variant annotations, gene annotations, and within-gene LD. It
models the covariance of variant effects within each gene as a combination of
functionally informed kernels, then reads heritability, enrichment, and effect
concordance off the fitted covariance.

A typical analysis runs:

```text
Dataset → KernelOptions / Model → sufficient statistics
        → fit_random_effects / fit_fixed_effects → result tables and moments
```

Every Python example in this document runs from the repository root, in order,
against the committed 50-gene test dataset in `tests/data/n50/`. Real analyses
use the full exome and 100 jackknife blocks; the examples use 10 blocks because
the test dataset has only 48 genes.

## Contents

1. [Concepts](#concepts)
2. [Logging](#logging)
3. [Datasets](#datasets)
4. [LDM](#ldm)
5. [Models and kernel options](#models-and-kernel-options)
6. [Heritability estimation](#heritability-estimation)
7. [Sufficient statistics and the jackknife](#sufficient-statistics-and-the-jackknife)
8. [Enrichment](#enrichment)
9. [Multi-model fitting](#multi-model-fitting)
10. [Simulation](#simulation)
11. [Covar: effect concordance](#covar-effect-concordance)
12. [Fixed-effect decomposition](#fixed-effect-decomposition)
13. [Scoring variants](#scoring-variants)
14. [Contracts](#contracts)
15. [API index](#api-index)

## Concepts

### The kernel model

For gene $g$ with $V_g$ rare coding variants, let $\beta_g$ be the vector of
per-allele effects. KASM assumes effects are independent across genes and
structured within each gene:

$$
\mathrm{Cov}(\tilde\beta_g) = \lambda_g \sum_{k \in \mathcal{H}} \tau_k K_{k,g}
  + \sum_{k \in \mathcal{O}} \tau_k K_{k,g}.
$$

1. **Heritable kernels** $\mathcal{H}$ encode assumptions about how variant
   effects covary within a gene; they are interpreted as biological.
2. **Overdispersion kernels** $\mathcal{O}$ are diagonal and absorb residual
   inflation, such as stratification; they are not counted as heritability and
   are not scaled by $\lambda_g$.
3. **Kernel coefficients** $\tau_k$ are shared across genes and unconstrained.
   In the API they are `kernel_coef`.
4. **Gene coefficients** $\eta$ scale the heritable kernels by gene annotation:
   $\lambda_g = \sum_z Z_{g,z}\,\eta_z$, where $Z_g$ is a one-hot gene annotation
   such as a constraint bin. They are nonnegative with mean one within each
   kernel group, which separates the overall magnitude (in $\tau$) from the
   relative scaling across genes (in $\eta$). In the API they are `gene_coef`.

$\tilde\beta_g$ denotes population effects: biological effects plus
non-biological contributions that overdispersion kernels are meant to absorb.

### Kernel types

1. **Burden kernel** $K_{bb,g} = w_{b,g} w_{b,g}^\top$ for a burden weight
   vector $w_b$, binary (a variant class indicator) or continuous (a
   deleteriousness score). Variants with larger weights share stronger effects
   in a common, gene-specific direction. Configured with `burden_cols`.
2. **Cross-burden kernel**
   $K_{bb',g} = w_{b,g} w_{b',g}^\top + w_{b',g} w_{b,g}^\top$ lets two burden
   directions covary. KASM adds every pairwise cross-burden kernel among the
   burden columns of a kernel group automatically.
3. **MAF-stratified burden** masks each weight by allele-frequency bin,
   $w_{b,m,g} = w_{b,g} \odot f_{m,g}$, so one annotation can act differently
   across frequency bins. Configured with `maf_bins` and `burden_by_maf=True`.
4. **Independent kernel** is diagonal and models variant effects that no burden
   direction captures. Configured with `independent_col`.
5. **Within-gene groups** split each burden kernel into independent per-group
   factors, such as exons, while keeping genes as the analysis unit and the
   full gene LD. Configured with `within_gene_group_col`.
6. **Covar kernel** is a binary, zero-diagonal matrix that selects pairs of
   variants, such as variants within 15 residues of each other. It measures
   whether selected pairs share effects. Configured with
   `extra_covar_kernel_fun` inside a kernel group.
7. **Overdispersion kernel** $\mathrm{diag}(f_k)$, usually one per MAF bin.
   Configured with `overdispersion_cols` and `overdispersion_by_maf=True`.

### Estimands

Let $R$ be the genotype covariance (LD) matrix and $\Sigma = E[\beta\beta^\top]$
the true effect covariance. KASM measures kernel similarity with the
inner product $\langle A, B\rangle_R = \mathrm{tr}(RARB)$ and projects $\Sigma$
onto the span $\mathcal{K}$ of the configured kernels. The projection
$K^* = P_\mathcal{K}(\Sigma)$ is the best-fitting kernel.

1. **Kernel heritability** is the heritability the best-fitting kernel explains,
   $h^2_\mathcal{K} = \mathrm{tr}[R\,P_\mathcal{K}(\Sigma)]$. It equals total
   heritability $\mathrm{tr}(R\Sigma)$ when $\Sigma \in \mathcal{K}$.
2. **Heritable component.** With both heritable and overdispersion kernels,
   KASM reports only the heritable part of the projection,
   $h^2_{h \mid o} = \mathrm{tr}[R\,C_h(\Sigma)]$. Confounding that projects
   onto the overdispersion space does not bias it.
3. **Heritability enrichment** of an annotation $\mathcal{A}$ compares its
   share of heritability with its share of heterozygosity, using the fitted
   heritable covariance $K$:
   $$
   \mathrm{Enrichment}(\mathcal{A}) =
   \frac{\sum_{v \in \mathcal{A}} R_{vv} K_{vv} \big/ \sum_v R_{vv} K_{vv}}
        {\sum_{v \in \mathcal{A}} R_{vv} \big/ \sum_v R_{vv}}.
   $$
   An enrichment of 1 means the annotation's variants have the average
   per-allele squared effect. `EnrichResult` also supports a per-variant
   (count) denominator.
4. **Effect concordance** for a set of variant pairs $\mathcal{P}_g$ is the
   mean effect cross-product over the geometric mean of the paired variances,
   aggregated across genes as a ratio of sums:
   $$
   \rho = \frac{E_g E_{(v,v') \in \mathcal{P}_g}[K_{vv'}]}
     {\sqrt{E_g E_{\mathcal{P}_g}[K_{vv}]\; E_g E_{\mathcal{P}_g}[K_{v'v'}]}}.
   $$
   Identical effects give $\rho = 1$; independent mean-zero effects give 0.
   For missense and pLoF classes,
   $r_{ML} = \rho_{ML} / \sqrt{\rho_{MM}\rho_{LL}}$ is the deattenuated
   correlation of the two classes' gene-level burden effects.

### Estimators

1. **Random effects** (subthreshold genes). KASM fits $\tau$ and $\eta$ by
   alternating Haseman–Elston regressions: with $\eta$ fixed the model is
   linear in $\tau$, and with $\tau$ fixed it is linear in $\eta$. The normal
   equations need only summary statistics:
   $T_{kl} = N^2\,\mathrm{tr}(K_k R K_l R)$,
   $t_k = N\,\mathrm{tr}(K_k R)$, and
   $q_k = N^2\,\hat\beta^\top D K_k D \hat\beta$, where $D = \mathrm{diag}(R)$.
   The estimate is
   $\hat h^2_\text{rand} = \sum_g \hat\lambda_g \sum_{k \in \mathcal{H}} \hat\tau_k\,\mathrm{tr}(R_g K_{k,g})$.
   With one burden kernel, no gene scaling, and no overdispersion, this is
   exactly burden heritability regression (BHR).
2. **Fixed effects** (significant genes). Each gene's burden effects
   $\alpha_g$ are fixed parameters of $y = X_g W_g \alpha_g + \epsilon$. With
   burden covariance $G_g = W_g^\top R_g W_g$ and
   $\hat\alpha_g = W_g^\top D_g \hat\beta_g$, the unbiased per-gene estimate is
   $\hat h^2_g = (N \hat\alpha_g^\top G_g^+ \hat\alpha_g - r_g)/(N - r_g)$ with
   $r_g = \mathrm{rank}(G_g)$, and an analytic standard error. Method
   `"indep"` replaces $W_g$ with the identity, one fixed effect per variant.
3. **Mixed** adds the random estimate over subthreshold genes and the fixed
   estimate over significant genes; the gene sets are disjoint, so the SEs
   add in quadrature.
4. **Significant genes** come from a one-dimensional burden test: one weight
   vector $\theta$ combines the burden columns, learned to maximize the summed
   per-gene $\chi^2$, and genes passing Bonferroni correction are treated as
   fixed effects.
5. **Block jackknife.** Genes are split into contiguous blocks, typically 100.
   Each block's sufficient statistics are subtracted and the full alternating
   fit is rerun; standard errors come from the delete-one-block estimates.

### Notation and API names

| Symbol | Meaning | API |
| --- | --- | --- |
| $\tau_k$ | kernel coefficient | `RandomFit.kernel_coef`, `kernel_coef_jk` |
| $\eta_z$ | gene coefficient | `RandomFit.gene_coef`, `gene_coef_dict` |
| $\lambda_g$ | per-gene scale | `RandomFit.gene_scale(model)` |
| $Z_g$ | gene annotation | `KernelOptions.gene_annot_cols` |
| $w_b$ | burden weight | `KernelOptions.burden_cols`, `Model.burden_wgts()` |
| $D_g^{1/2}$ | variant scale | `Model.variant_scale()` |
| $R_g$ | within-gene LD | `LDM`, `Dataset.load_ldm()` |
| $T, t, q$ | moment sums | `SuffStats`, `JKSuffStats` |
| $\hat h^2$ | heritability | `RandomFit.hsq`, `FixedFit.hsq` |

The variant scale squared is $1/(N\,\mathrm{se}^2)$ by default, or
$2p(1-p)$ with `variant_variance_from_maf=True`.

### Standard model designs

These designs cover common analyses. Column names refer to the test dataset;
`MAF_BINS` uses the three rare-variant bins and `OEBINS` the five constraint
bins.

```python
from pathlib import Path

import kasm
import numpy as np
import polars as pl

MAF_BINS = {
    "ultra_rare": (0, 1e-5),
    "rare": (1e-5, 1e-4),
    "low_frequency": (1e-4, 1e-3),
}
OEBINS = [f"oebin{i}" for i in range(1, 6)]
SCORES = ["vep.polyphen", "cadd.EsmScoreMissense", "am.pathogenicity"]
common = dict(
    maf_col="AF",
    maf_bins=MAF_BINS,
    gene_annot_cols=OEBINS,
    burden_by_maf=True,
    overdispersion_cols=["constant"],
    overdispersion_by_maf=True,
)

# Naive: one constant burden weight.
naive = kasm.KernelOptions(
    **common | {"burden_by_maf": False}, burden_cols=["constant"]
)
# Baseline: constant plus three missense scores, each stratified by MAF bin.
baseline = kasm.KernelOptions(**common, burden_cols=["constant", *SCORES])
# Baseline + Annotation: one more burden column, for enrichment.
plus_annotation = kasm.KernelOptions(
    **common, burden_cols=["constant", *SCORES, "netsurfp.rsa_G1"]
)
# Baseline x Exon: burden kernels split by exon within each gene.
by_exon = kasm.KernelOptions(
    **common, burden_cols=["constant", *SCORES], within_gene_group_col="cds_exon"
)
# Baseline + Indep: adds an independent per-variant kernel.
plus_indep = kasm.KernelOptions(
    **common, burden_cols=["constant", *SCORES], independent_col="constant"
)
```

A BHR-like design uses binary damaging and benign indicators as burden columns.
The position kernels of "Baseline + Indep + Position" are covar kernels; see
[Covar](#covar-effect-concordance).

## Logging

```python
kasm.configure_logging("INFO")
kasm.logger.info("analysis started")
```

`configure_logging(level, filename=None)` also writes to a file when
`filename` is given. Data alignment and fitting events are logged at INFO.

## Datasets

`Dataset` holds one trait's summary statistics, variant annotations, gene
annotations, and LD source, as Polars tables.

```python
DATA = "tests/data/n50"
variant_df = pl.read_parquet(f"{DATA}/variant_df.parquet")
gene_df = pl.read_csv(f"{DATA}/gene_df.tsv", separator="\t")
missense_df = variant_df.filter(pl.col("variant_class") == "missense")

dataset = kasm.Dataset(
    missense_df,
    gene_df=gene_df,
    beta_col="BETA",
    se_col="SE",
    sample_size=missense_df["N"].mean(),
    ldm=f"{DATA}/exome.ldm",
)
dataset.n_genes, len(dataset.genes)
```

1. **Variant table.** Requires explicit `variant` and `gene` columns, the
   effect estimate and standard error, and every annotation the model uses.
   Variant IDs are canonical `[chr]:[pos]:[ref]:[alt]` strings, which define
   genomic order and LD alignment. IDs must be unique, each gene's rows must
   be contiguous, and gene order is the order of first appearance.
2. **Gene table.** Optional; keyed by `gene`. `Dataset` keeps the genes present
   in both tables, without reordering variant rows, and fails when none
   overlap. The alignment counts are logged as `event=gene_df_alignment`.
3. **Types.** Numeric and Boolean columns are cast to `Float64`; null or
   non-finite values raise. The dataset is immutable: `variant_df()` and
   `gene_df()` return snapshots, and `variant_values()` and `gene_values()`
   return read-only NumPy views.
4. **Ordering helpers.** `parse_variant_ids()` splits IDs into chromosome rank,
   position, reference, and alternate. `sort_variant_df(frame, *by)` sorts by
   optional leading keys, then genomic position; `gene_col="gene"` orders genes
   by their first variant and keeps each gene contiguous.

```python
kasm.parse_variant_ids(missense_df["variant"].head(2))
```

## LDM

An LDM file (`exome.ldm`) stores one sparse within-gene LD matrix per gene.
`Dataset` loads each gene's matrix and aligns it to its variant rows:

1. A gene missing from the LDM uses identity LD, with a warning.
2. A variant missing from an otherwise present gene gets an identity diagonal.

`LDMLoader` gives direct read-only access.

```python
gene = dataset.genes[0]
gene_variants = dataset.variant_df().filter(pl.col("gene") == gene)["variant"]
loader = kasm.LDMLoader(f"{DATA}/exome.ldm")
ldm = loader.load_gene_ldm(gene, gene_variants.to_numpy())

design = np.ones((ldm.shape[0], 1))
propagated = ldm @ design
quadform = ldm.quadform(design)
diagonal = ldm.diag()
subset = ldm.subset(np.array([0, 1]))
loader.close()
```

`LDM` operations stay sparse; call `toarray()` only when a dense matrix is
needed.

## Models and kernel options

`KernelOptions` declares the kernel design, and `Model` binds it to a dataset
and validates the referenced columns, binary supports, and annotation rank.

```python
model = kasm.Model(dataset, baseline).compute_suffstats()
model.n_kernels, model.burden_wgts(gene).shape
```

The options are:

1. `burden_cols`: variant columns used as burden weights.
2. `maf_col`, `maf_bins`: allele-frequency column and named half-open bins,
   `low <= af < high`.
3. `burden_by_maf`: stratify each burden column by MAF bin. Default `True`.
4. `gene_annot_cols`: mutually exclusive one-hot gene columns for gene
   coefficients. Without them, all genes share one coefficient.
5. `independent_col`: adds one heritable diagonal kernel weighted by this
   nonnegative column. Use an all-ones column for unstructured per-variant
   effects.
6. `overdispersion_cols`, `overdispersion_by_maf`: diagonal overdispersion
   kernels, optionally stratified by MAF bin. `overdispersion_shared=True`
   shares one coefficient per MAF bin across the columns.
7. `within_gene_group_col`: split each burden kernel into independent
   within-gene groups, such as exons. Genes remain the analysis and jackknife
   units, and LD between groups is kept.
8. `kernel_groups`: several named heritable groups in one model, each a
   dictionary with `burden_cols` and optional `independent_col`,
   `overdispersion_cols`, `extra_covar_kernel_fun`, and
   `extra_covar_with_indep`. Each group has its own gene coefficients. Burden
   columns must be zero outside the support of the group's `independent_col`.
9. `covariance_between_kernel_groups`: maps a derived group name to two base
   groups and adds every cross-burden kernel between them, with its own gene
   coefficients. Independent kernels are never crossed.
10. `variant_variance_from_maf`: use $2p(1-p)$ instead of the observed-SE
    variant scale. Required for simulation.

## Heritability estimation

Compute sufficient statistics once, find significant genes with the burden
test, then fit subthreshold genes as random effects and significant genes as
fixed effects.

```python
pvalues = model.compute_gene_pvalue(method="burden_1d")
signif_genes = pvalues.filter(pl.col("pvalue") < 0.05 / len(pvalues))["gene"].to_list()

random_fit = kasm.fit_random_effects(model, exclude_genes=signif_genes, n_jk_blocks=10)
fixed_fit = kasm.fit_fixed_effects(model, include_genes=signif_genes)
mixed_hsq, mixed_hsq_se = kasm.combine_mixed_effects(
    random_fit=random_fit, fixed_fit=fixed_fit
)
random_fit.hsq, random_fit.hsq_se, fixed_fit.hsq, fixed_fit.hsq_se
```

1. **Gene selection.** Random fits require explicit `exclude_genes`; fixed fits
   require explicit `include_genes`. `combine_mixed_effects()` requires the
   fixed genes to equal the random exclusions exactly, so no gene is counted
   twice. Its SE is the quadrature sum of the random jackknife SE and the fixed
   analytic SE.
2. **Burden test.** `compute_gene_pvalue("burden_1d")` learns one projection
   across all genes and stores it on the model as `burden_1d_wgts`, which
   `fit_fixed_effects(method="burden_1d")` reuses. `"burden_full"` tests all
   burden columns jointly.
3. **Fixed methods.** `"burden"` (default) fits the full burden design jointly;
   `"burden_1d"` fits the one-dimensional projection; `"indep"` fits one
   effect per variant.
4. **Standalone fixed fit.** Pass a `Dataset` with `kernel_options=` instead of
   a `Model`. Use the `Model` form when the result goes to
   `combine_mixed_effects()`.

```python
standalone = kasm.fit_fixed_effects(
    dataset, kernel_options=baseline, include_genes=signif_genes, method="indep"
)
```

### Random-effect results

```python
random_fit.point  # total and per-group hsq, hsq_se, n_genes, n_variants
random_fit.jk  # delete-block hsq per kernel group
random_fit.components  # per-kernel coefficients and variances
random_fit.overdispersion_components  # overdispersion coefficients and variances
random_fit.maf_components  # additive allocation across MAF bins
random_fit.gene_annot_components  # additive allocation across gene annotations
random_fit.kernel_coef, random_fit.kernel_coef_se, random_fit.gene_coef
fixed_fit.per_gene, fixed_fit.maf_components
```

1. **MAF components** allocate heritability exactly across `maf_bins`: their
   sum equals `hsq`, and they can be negative. Cross terms between bins are
   split equally. Fixed MAF components have no separate SEs.
2. **Gene-annotation components** allocate random heritability across
   `gene_annot_cols`, with `het` (heterozygosity), counts, and jackknife
   replicates; they sum to `hsq` at the point and in every replicate.
3. **Coefficient shapes.** `kernel_coef` follows `kernel_labels`;
   `gene_coef` has shape `(n_annotations, n_kernel_groups)` and `gene_coef_jk`
   `(n_blocks, n_annotations, n_kernel_groups)`.
4. **Grouped fits** add one `point` and `jk` row per kernel group; group SEs
   come from summed delete-block estimates.

### Gene coefficients

By default gene coefficients are estimated from a deterministic multistart set.
`init_gene_coef` starts from given values, and `fix_gene_coef=True` holds them
fixed through the point fit and every jackknife replicate:

```python
fixed_scale_fit = kasm.fit_random_effects(
    model,
    exclude_genes=signif_genes,
    n_jk_blocks=10,
    init_gene_coef=random_fit.gene_coef_dict,
    fix_gene_coef=True,
)
```

`init_gene_coef` follows the model structure: `{annotation: value}` for an
ordinary model, `{group: {annotation: value}}` for a grouped model, and
`{label: ...}` for a model mapping. `gene_coef_dict` returns the same structure,
so one fit can initialize the next. Labels are `gene_annot_cols`, or
`constant` without gene annotations. Estimated values are renormalized to mean
one; fixed values keep their scale.

### Convergence

Random fits stop when gene coefficients change by less than `tol` (default
`1e-6`) or after `max_iter` iterations (default 50). Nonconvergence is
diagnostic only: `converged`, `n_iter`, `converged_jk`, and `n_iter_jk` record
it, final finite iterates are kept, and SEs use all delete-block estimates.
Missing or non-finite outputs, rank failures, and solver failures raise.

`psd_burden_coef=True` constrains the burden coefficient matrix to be positive
semidefinite, so the implied burden covariance is valid. The default is
`False`.

## Sufficient statistics and the jackknife

1. **`SuffStats`** holds per-gene moments, used by burden tests and fixed
   fits. `model.compute_suffstats(extra_enrich_cols=[...])` computes it in one
   pass over the LD, including any enrichment extensions and configured covar
   kernels.
2. **`JKSuffStats`** holds per-block moments for random fits. It is computed
   from `SuffStats` without touching LD, and stored as `model.jk_suffstats`.

```python
jk_stats = model.compute_jk_suffstats(exclude_genes=signif_genes, n_jk_blocks=10)
import tempfile

out = Path(tempfile.mkdtemp())
jk_stats.save(out / "trait.jk.npz")
loaded = kasm.JKSuffStats.load(out / "trait.jk.npz")
from_file = kasm.fit_random_effects(loaded, n_jk_blocks=10)
```

1. **Blocks.** Genes form contiguous blocks in dataset order, or follow
   explicit `gene_blocks` labels. Each block's moments are summed once, so the
   jackknife needs one exome-wide LD pass.
2. **Reuse.** `fit_random_effects(model, ...)` prepares and caches
   `jk_suffstats` on first use; a later call with the same exclusions and
   blocks reuses it, and different geometry raises until recomputed.
   `compute_suffstats()` clears the cache.
3. **Persistence.** `save()` and `load()` keep moments, layout, extensions, and
   exclusions together. A loaded `JKSuffStats` can be fitted without the LD or
   variant table.
4. **Extensions.** Enrichment and covar extensions are `ExtSuffStats` entries
   in the parents' `burden_extensions` and `covar_extensions`. Fitting selects
   one extension and assembles the augmented system internally.

## Enrichment

Prepare candidate annotations in the same LD pass, then select one per fit.
Random selection returns `(RandomFit, EnrichResult)`; so does fixed selection.

```python
model = kasm.Model(dataset, baseline).compute_suffstats(
    extra_enrich_cols=["netsurfp.rsa_G1", "tmhmm.feature_tmhelix"]
)
random_enrich_fit, random_enrich = kasm.fit_random_effects(
    model,
    exclude_genes=signif_genes,
    n_jk_blocks=10,
    extra_enrich_col="netsurfp.rsa_G1",
)
fixed_enrich_fit, fixed_enrich = kasm.fit_fixed_effects(
    model, include_genes=signif_genes, extra_enrich_col="netsurfp.rsa_G1"
)
random_enrich.enrichment(), random_enrich.enrichment_se()
```

1. **Random estimand.** One augmented model with the baseline burden columns,
   the annotation column, and their cross kernels. Enrichment is diagonal
   attribution: `annot_hsq` and `total_hsq` are $\sum R_{vv}K_{vv}$ over the
   annotation and over all variants. The augmented fit's own `hsq` is a
   different estimand from `total_hsq`.
2. **Fixed estimand.** Nested projections onto `[X, X * annotation]`, each with
   its own rank correction; the SE uses the delta method with the covariance of
   the two projections.
3. **Annotation values.** Binary columns give enrichment; continuous and signed
   columns use the same path and give weighted moments. Values enter
   heterozygosity linearly and are never squared or binarized.
4. **Normalization.** `enrichment()` divides by the heterozygosity share by
   default (`normalization="het"`). For a per-variant denominator, construct
   the result with `annot_count` and `total_count` (and their `_jk` replicates
   for random results) and pass `normalization="count"`.
5. **Mixed heritability** with an enrichment fit still uses each fit's ordinary
   `hsq`.
6. **Exons.** With `within_gene_group_col`, extensions use the same
   within-gene groups.

### Pooling across traits

Pool separately fitted results by adding moments before forming the ratio.
Random results pool delete-block moments with the same weights; fixed results
use analytic variances, and `independent=True` asserts independent inputs.

```python
pooled = kasm.pool_enrichment([random_enrich, random_enrich], weights=[0.5, 0.5])
pooled.enrichment(), pooled.enrichment_se()
```

Results save and load independently with `save(path)` and
`EnrichResult.load(path)`.

## Multi-model fitting

`fit_random_effects()` accepts a mapping of named models, such as variant
classes, traits, or cohorts, and returns a dictionary of `RandomFit` with the
same keys. Sharing is fixed:

1. Burden coefficients are shared when their kernel identities match.
2. Independent and overdispersion coefficients, and gene coefficients, are
   fitted per model.

To estimate total heritability across nonoverlapping variant classes, fit one
model per class jointly, then sum:

```python
class_models = {}
for variant_class in ("missense", "lof", "synonymous"):
    class_df = variant_df.filter(pl.col("variant_class") == variant_class)
    class_dataset = kasm.Dataset(
        class_df,
        gene_df=gene_df,
        beta_col="BETA",
        se_col="SE",
        sample_size=class_df["N"].mean(),
        ldm=f"{DATA}/exome.ldm",
    )
    class_models[variant_class] = kasm.Model(
        class_dataset,
        kasm.KernelOptions(**common, burden_cols=["constant"]),
    ).compute_suffstats()

class_fits = kasm.fit_random_effects(
    class_models, exclude_genes=signif_genes, n_jk_blocks=10
)
total_hsq, total_hsq_se = kasm.combine_random_hsq(class_fits)
```

1. `weights` weights each model's moment equations.
2. `combine_random_hsq()` sums aligned point and delete-block estimates. It
   does not check for overlapping variants; summing overlapping models would
   double count.
3. A flat `exclude_genes` is intersected with each model's genes; a mapping
   must use exactly the model labels. All models use one block count.
4. Combine mixed effects per model by pairing each `RandomFit` with that
   model's `FixedFit`.

## Simulation

Simulation draws effects from a fitted model and maps them to summary
statistics through the LD. It needs a model with
`variant_variance_from_maf=True`.

```python
simulation_options = kasm.KernelOptions(
    maf_col="AF",
    burden_cols=["constant", "am.pathogenicity"],
    independent_col="constant",
    variant_variance_from_maf=True,
)
source_model = kasm.Model(dataset, simulation_options).compute_suffstats()
source_fit = kasm.fit_random_effects(
    source_model, exclude_genes=signif_genes, n_jk_blocks=10
)

np.random.seed(0)
causal_effects = kasm.simulate_burden_effects(source_model, source_fit, n_sim=2)
betahat, simulated_se = kasm.simulate_sumstats(source_model, causal_effects[:, 0])
```

1. `simulate_burden_effects()` draws structured per-gene burden effects,
   without LD. `gene_scale=` overrides how effects are spread across genes.
2. `simulate_sumstats()` maps any causal vector through each gene's LD and
   adds correlated sampling noise, returning `(betahat, se)`. Independent and
   overdispersion effects are ordinary per-variant vectors to add before this
   call.
3. Both use NumPy's global random state; seed it per replicate.
4. Coefficients transfer between observed-SE and MAF-scaled models as unscaled
   per-allele effects, which does not preserve heritability. Define simulation
   truth from the transported effects.

## Covar: effect concordance

A covar kernel selects variant pairs; fitting it measures whether those pairs
share effects. Define covar kernels inside a kernel group with a module-level
function that takes one gene's variant table and returns a mapping of names to
binary, symmetric, zero-diagonal sparse matrices.

```python
from scipy import sparse


def near_pairs(gene_variant_df):
    position = (
        gene_variant_df["variant"].str.split(":").list.get(1).cast(pl.Int64).to_numpy()
    )
    distance = np.abs(position[:, None] - position[None, :])
    return {"near": sparse.csr_matrix((distance > 0) & (distance < 45), dtype=float)}


covar_model = kasm.Model(
    dataset,
    kasm.KernelOptions(
        gene_annot_cols=OEBINS,
        kernel_groups={
            "missense": {
                "burden_cols": ["constant", "am.pathogenicity"],
                "independent_col": "constant",
                "extra_covar_kernel_fun": near_pairs,
                "extra_covar_with_indep": True,
            }
        },
    ),
).compute_suffstats()
base_fit = kasm.fit_random_effects(
    covar_model, exclude_genes=signif_genes, n_jk_blocks=10
)
covar_fit, covar = kasm.fit_random_effects(
    covar_model,
    exclude_genes=signif_genes,
    n_jk_blocks=10,
    extra_covar_kernel="missense:near",
    init_gene_coef=base_fit.gene_coef_dict,
    fix_gene_coef=True,
)
concordance = covar.point[0] / covar.point[1]
```

1. **Selection.** Name one covar kernel as `"group:name"`; the call returns
   `(RandomFit, CovarResult)`. Loop to analyze several kernels. Gene
   coefficients must be fixed, usually from a baseline fit.
2. **Independent term.** `extra_covar_with_indep=True` adds a diagonal kernel on
   the variants participating in the pairs, with its own coefficient.
3. **Result.** `CovarResult.point` is `(numerator, variance denominator)` and
   `jk` has shape `(n_blocks, 2)`. Pool these additive moments across traits
   before forming the ratio and its jackknife SE. `n_genes` and `n_pairs`
   record support.
4. Covar inference is random-only.

### Concordance from a baseline fit

`compute_random_covar(model, fit, ...)` reads covariance moments for
indicator-defined pairs from an existing fit, without refitting.

```python
missense_class = dataset.variant_df()["am.pathogenicity"] > 0.564
covar_indicator_model = kasm.Model(
    kasm.Dataset(
        dataset.variant_df().with_columns(
            high=missense_class.cast(pl.Float64),
            other=(~missense_class).cast(pl.Float64),
        ),
        gene_df=gene_df,
        beta_col="BETA",
        se_col="SE",
        sample_size=dataset.sample_size,
        ldm=f"{DATA}/exome.ldm",
    ),
    baseline,
).compute_suffstats()
indicator_fit = kasm.fit_random_effects(
    covar_indicator_model, exclude_genes=signif_genes, n_jk_blocks=10
)
high_high = kasm.compute_random_covar(
    covar_indicator_model, indicator_fit, group_col="high"
)
high_other = kasm.compute_random_covar(
    covar_indicator_model, indicator_fit, group_col1="high", group_col2="other"
)
```

1. `group_col` gives within-indicator moments, `(covariance, variance)`;
   `group_col1` with `group_col2` gives the cross covariance only. Cross
   indicators must not overlap.
2. Moments are sums of per-gene pair and variance means. Pool across traits,
   then form $P/V$, or $P_{12}/\sqrt{V_1 V_2}$ for a cross ratio, using common
   gene support.
3. `include_genes` restricts support without changing the fit. Models with
   `within_gene_group_col` are not supported.

## Fixed-effect decomposition

`FixedFit.decompose(columns)` allocates one fixed-effect fit additively across
categorical variant columns, such as protein domains, using the fit's
heritability as the denominator for every group.

```python
gene_rows = (
    dataset.variant_df()
    .filter(pl.col("gene") == signif_genes[0])
    .with_columns(buried=pl.col("netsurfp.rsa_G1").cast(pl.Int64).cast(pl.String))
)
one_gene = kasm.Dataset(
    gene_rows,
    beta_col="BETA",
    se_col="SE",
    sample_size=dataset.sample_size,
    ldm=f"{DATA}/exome.ldm",
)
gene_fit = kasm.fit_fixed_effects(
    one_gene,
    kernel_options=kasm.KernelOptions(
        burden_cols=["constant", "am.pathogenicity"], maf_col="AF", maf_bins=MAF_BINS
    ),
    include_genes=[signif_genes[0]],
    method="indep",
)
gene_fit.decompose(["buried"])
```

The table reports group and total heritability with covariance-aware SEs,
heritability share, heterozygosity share, and enrichment. Binary columns are
two-level partitions. Pass several columns at once to share one LD pass.

## Scoring variants

A saved `RandomFit` can score variants in any aligned dataset without refitting:

```python
random_fit.save(out / "trait.fit.npz")
reloaded = kasm.RandomFit.load(out / "trait.fit.npz")
scores = reloaded.score_variants(dataset)
burden_scale = reloaded.gene_scale(model, burden_col="constant")
```

1. `score_variants()` returns `effsq`, the model-implied expected squared
   per-allele effect with unit gene coefficients, and
   `hsq = effsq * variant_scale**2`, the per-variant heritability attribution.
   It requires a single kernel group and is meant for ranking within genes.
2. `gene_scale(model)` returns the per-gene, per-group scale
   $\lambda_g$; `burden_col=` selects the group of one burden column.
3. A loaded fit needs no `Model`; reports that depend on geometry are computed
   at fit time and saved with it.

## Contracts

1. Random fits require explicit `exclude_genes`; prepared statistics reuse
   their stored exclusions. Fixed fits require explicit `include_genes`.
2. Enrichment extensions require a single kernel group; multi-group models are
   rejected before computation.
3. Covar inference is random-only; enrichment and covar cannot be selected in
   the same fit.
4. Nonconvergence is diagnostic only; missing or non-finite outputs, invalid
   geometry, rank failures, and solver failures raise.
5. Genes missing from the LDM use identity LD with a warning.
6. Exons are within-gene burden groups; genes remain the analysis and jackknife
   units.
7. Kernel coefficients are unconstrained; gene coefficients are nonnegative
   with mean one per kernel group.

## API index

| Name | Purpose | Section |
| --- | --- | --- |
| `Dataset` | summary statistics, annotations, LD source | [Datasets](#datasets) |
| `parse_variant_ids` | split canonical variant IDs | [Datasets](#datasets) |
| `sort_variant_df` | sort variants genomically | [Datasets](#datasets) |
| `LDM`, `LDMLoader` | within-gene LD matrices | [LDM](#ldm) |
| `KernelOptions` | kernel design | [Models](#models-and-kernel-options) |
| `KernelId`, `KernelLayout` | compiled kernel identities and order | [Models](#models-and-kernel-options) |
| `Model` | dataset plus design; statistics and burden tests | [Models](#models-and-kernel-options) |
| `fit_random_effects` | random-effects fit, enrichment, covar | [Heritability](#heritability-estimation) |
| `fit_fixed_effects` | fixed-effects fit and enrichment | [Heritability](#heritability-estimation) |
| `combine_mixed_effects` | random plus fixed heritability | [Heritability](#heritability-estimation) |
| `RandomFit`, `FixedFit` | fit results | [Heritability](#heritability-estimation) |
| `SuffStats`, `JKSuffStats` | per-gene and per-block moments | [Statistics](#sufficient-statistics-and-the-jackknife) |
| `compute_jk_suffstats` | array-level block aggregation | [Statistics](#sufficient-statistics-and-the-jackknife) |
| `ExtSuffStats`, `ExtraEnrichCols` | extension moments and payloads | [Statistics](#sufficient-statistics-and-the-jackknife) |
| `jackknife_se` | delete-one-block standard error | [Statistics](#sufficient-statistics-and-the-jackknife) |
| `EnrichResult`, `pool_enrichment` | enrichment moments and pooling | [Enrichment](#enrichment) |
| `combine_random_hsq` | sum joint random fits | [Multi-model](#multi-model-fitting) |
| `simulate_burden_effects`, `simulate_sumstats` | simulation | [Simulation](#simulation) |
| `CovarResult`, `compute_random_covar`, `covar` | effect concordance | [Covar](#covar-effect-concordance) |
| `configure_logging`, `logger` | logging | [Logging](#logging) |
