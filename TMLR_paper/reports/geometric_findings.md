# CPU geometric findings

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
