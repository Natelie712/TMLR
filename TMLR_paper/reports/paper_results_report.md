# Curveball TMLR revision: empirical results

## Main result

All 21 model/behavior test comparisons are reused from completed, validation-frozen runs. No test search was performed in this revision. Values are CAA-style all-option normalized target-label probability changes in percentage points, with a shared true no-hook baseline.

| Model | Behavior | Selected lambda | Linear | Curveball | Difference [paired 95% CI] |
| --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B | AI Coordination | 0.0 | +6.083 | +5.337 | -0.746 [-1.446, -0.219] |
| Qwen3-1.7B | Corrigibility | 0.0 | +22.786 | +23.801 | +1.016 [-1.852, +4.233] |
| Qwen3-1.7B | Hallucination | 0.01 | +8.968 | +28.955 | +19.988 [+11.398, +29.375] |
| Qwen3-1.7B | Myopic Reward | 0.0 | +19.586 | +44.735 | +25.150 [+13.656, +36.978] |
| Qwen3-1.7B | Survival Instinct | 0.0 | +12.365 | +5.613 | -6.752 [-11.017, -3.006] |
| Qwen3-1.7B | Sycophancy | 0.005 | +2.498 | +6.258 | +3.760 [-4.216, +11.844] |
| Qwen3-1.7B | Refusal | 0.0 | +4.975 | +11.813 | +6.838 [+1.719, +13.098] |
| Llama-3.2-3B-Instruct | AI Coordination | 0.001 | +0.611 | +0.613 | +0.001 [-0.020, +0.024] |
| Llama-3.2-3B-Instruct | Corrigibility | 0.0 | +2.418 | +2.448 | +0.030 [+0.005, +0.057] |
| Llama-3.2-3B-Instruct | Hallucination | 0.5 | -1.672 | -1.701 | -0.029 [-0.082, +0.014] |
| Llama-3.2-3B-Instruct | Myopic Reward | 0.005 | -0.292 | -0.270 | +0.021 [-0.015, +0.059] |
| Llama-3.2-3B-Instruct | Survival Instinct | 0.5 | -1.104 | -0.671 | +0.433 [-0.195, +1.147] |
| Llama-3.2-3B-Instruct | Sycophancy | 0.5 | +0.288 | +0.552 | +0.264 [-0.259, +0.847] |
| Llama-3.2-3B-Instruct | Refusal | 0.01 | +1.370 | +1.430 | +0.060 [-0.131, +0.306] |
| Llama-2-7B-Chat | AI Coordination | 0.0 | +23.015 | +23.053 | +0.038 [-0.576, +0.653] |
| Llama-2-7B-Chat | Corrigibility | 0.0 | +20.052 | +19.957 | -0.096 [-0.553, +0.375] |
| Llama-2-7B-Chat | Hallucination | 0.0 | +4.581 | +4.630 | +0.048 [+0.007, +0.115] |
| Llama-2-7B-Chat | Myopic Reward | 0.5 | +4.247 | +3.813 | -0.434 [-0.692, -0.203] |
| Llama-2-7B-Chat | Survival Instinct | 0.25 | +20.844 | +20.282 | -0.562 [-1.043, -0.111] |
| Llama-2-7B-Chat | Sycophancy | 0.5 | -5.394 | -12.548 | -7.153 [-14.143, -0.354] |
| Llama-2-7B-Chat | Refusal | 0.0 | +11.481 | +11.599 | +0.119 [-0.017, +0.303] |

## Current completion status

| Model | Response diagnostics | Completed behaviors |
| --- | --- | --- |
| qwen3 | complete | 7 |
| llama3b | complete | 7 |
| llama2 | complete | 7 |

CPU geometry complete: True. Complete response-curve cells currently available: 265. Pending panels are labelled pending and must not be submitted as completed figures.


## CPU geometric findings

All 189 fitted kernels retain less than 95% of centered-kernel variance with the frozen 20 components. The retained fraction ranges from 41.292% to 91.791%. This measures RKHS variance, not ambient activation variance. The manuscript must not claim an automatically satisfied 95% rule.

Qwen3-1.7B validation cross-prompt mean cosine spans 0.0108 to 0.2205 across the seven behaviors. This shows model/behavior dependence rather than a universal coherent contrast direction.
Llama-3.2-3B-Instruct validation cross-prompt mean cosine spans 0.0030 to 0.2982 across the seven behaviors. This shows model/behavior dependence rather than a universal coherent contrast direction.
Llama-2-7B-Chat validation cross-prompt mean cosine spans 0.0085 to 0.2443 across the seven behaviors. This shows model/behavior dependence rather than a universal coherent contrast direction.

Selected Qwen Hallucination lambda 0.01 has paired-negative-activation magnitude/distance group Spearman rho=0.03777234112230548. Its interval crosses zero. The strong MCQ effect therefore does not establish the original paper's proposed positive magnitude-distance correlation mechanism. This analysis is at completed-negative-label activations, a different location from the MCQ decision token.

Qwen Hallucination: mean cluster-direction cosine=0.4900; mean size-matched random-group-partition cosine=0.9818. These are descriptive fitting-data diagnostics, not independent behavioral validation.
Qwen Survival Instinct: mean cluster-direction cosine=0.1380; mean size-matched random-group-partition cosine=0.9225. These are descriptive fitting-data diagnostics, not independent behavioral validation.

Qwen Hallucination local clusters are strongly associated with target answer labels: 3/8 clusters contain only one target label; the remaining cluster majority fractions are at least 88.7%. This is empirical evidence of a label-geometry confound, not proof that every nonlinear update is a label shortcut. CAA paired examples differ in completed answer labels. Genuine one-to-one pairing alone does not remove this confound; do not equate local clustering with a learned behavioral manifold. Full counts are in cluster_target_label_audit.csv.

The corresponding in-sample training magnitude-distance correlation is rho=-0.0338, 95% CI [-0.1102, 0.0332] over 754 source groups. This also does not support a positive correlation for the selected Hallucination configuration.

For Qwen Hallucination, the common 2D unit-update projection has two-minus-one-component GMM BIC -37.35 at lambda 0.01 and -165.87 at lambda 1. A two-component mixture fits these projected samples better under this diagnostic, while lambda 0 is nearly constant. This does not establish genuine multimodality: a mixture can approximate skewed unimodal data, and projection and answer-label structure remain confounds.

Global frozen source-group overlap checks passed: {'fit_validation': 0, 'fit_benchmark': 0, 'validation_benchmark': 0}. No label corrections or exclusions were made.

## Claims supported and claims requiring revision

- The original broad claim that Curveball consistently outperforms Linear across models and behaviors is not supported by these CAA results. All weak/negative behaviors remain in the table.
- Qwen Hallucination supports a focused nonlinear-weight case: lambda 0.01 improves over lambda 0 on validation and the repeatedly used official benchmark. The verified validation crossover attributes the gain primarily to direction, not a larger L2 budget. These interventions are diagnostic counterfactuals, not proposed algorithms.
- A lambda 0 advantage is a linear-kernel KPCA pipeline effect; it does not demonstrate nonlinear steering benefit. In particular, differences in displacement, component truncation and inverse reconstruction remain relevant.
- Llama3B's tiny weight sensitivity applies to this block, kernel scale, fitting data and frozen native step. It does not prove that the model lacks nonlinear structure.
- On Llama2, four behaviors select lambda 0. Nonzero weights modestly improve Myopic Reward and Survival Instinct relative to lambda 0 while remaining below direct Linear. Sycophancy remains a negative steering result.
- Local direction variation, projected clusters or magnitude correlations do not themselves demonstrate causality, intrinsic manifold curvature or a behavioral advantage. Near-constant magnitudes are excluded from meaningful correlation claims. The two-component GMM diagnostic is descriptive and projection-dependent.
- New CAA data contain no Humor/Rudeness/Excitement/Sadness judge or classifier modality. Original claims about tone steering and tone-vs-behavior geometry must be removed from this empirical revision.
- Figure 2B VAE distortion is not reproduced. No new-data curvature–steering claim should be carried forward. Independent synthetic panels remain pending provenance confirmation, not silently relabelled as CAA geometry.

## Methods and statistical limits

Training uses only fitting groups and is evaluated in-sample. Validation selects lambda from the already tested grid; selected maxima can be optimistic. Official Test is an exploratory fixed benchmark because previous diagnoses repeatedly inspected it. Published mean effects and 10,000 paired source-group bootstrap intervals are reused without changing the metric or labels. Pointwise intervals do not imply multiplicity-corrected superiority.

The primary interventions use last-token block outputs, natural residual-preserved Curveball updates, fixed 20 KPCA components, weighted degree-2 kernel with gamma=0.01 and inverse ridge=1. Native Curveball steps are training-only RMS-calibrated at lambda 1 and then held fixed across weights; actual evaluation displacements are not assumed equal to Linear. Llama2 uses its validated parenthesis prompt; Qwen/Llama3B use the frozen assistant-answer prefill. All methods within each model use identical prompts and labels. No Angular values with a mismatched multi-token scope are included.

Paired-label geometry uses actual positive/nonmatching completed-answer activations from the same source question. Figure 6 MCQ magnitudes instead use the final input token at the answer-decision position. These locations are separately labelled and should not be conflated. Relative preimage reconstruction error is not residual-preserved steering error. Kernel variance fractions refer to the centered RKHS kernel spectrum, not raw activation-space explained variance.

Table 2 uses validation group-mean observations, 1,000 group bootstraps and 1,999 group permutations with Holm correction across all nonconstant tested rows. A separate full in-sample fitting-data correlation table uses genuine paired training activations and the identical diagnostic weights; its Holm family is separate from validation. No MCQ probability is used as a paired-target distance. Table 4 averages cross-prompt cosines across distinct source groups and uses group bootstrap. Figure 4 clusters fitting negative group-centroid activations (k=8, n_init=20, seed=20261111) and compares 500 size-matched random partitions. This group aggregation is a documented adaptation of the manuscript's negative-activation clustering.

## Placement and numbering

Open the [figure gallery](figure_gallery.html) for completed figures and the [scientific figure review](figure_refinement_review.md) for numerical checks and interpretation limits. Incomplete response figures are stored in `figures/drafts/` and excluded from the manuscript bundle.

See [artifact inventory](paper_artifact_inventory.md), [execution plan](experiment_execution_plan.md), and [VAE resource assessment](vae_resource_assessment.md). Tables 1–4 retain their original scientific roles. Figures 8–9 are removed; tone Figures 17–20 are replaced as documented. Figure 21 adds Llama2 response curves. All PDFs are vector Matplotlib output with embedded TrueType fonts. Use booktabs and graphicx for the LaTeX tables. Historical inputs remain read-only during this revision; no GitHub push.
