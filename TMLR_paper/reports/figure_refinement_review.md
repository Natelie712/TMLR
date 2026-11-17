# Figure refinement and scientific review

The existing paper figures were replaced in place after a preserved snapshot at `figures_before_refinement_20261112.zip`. No historical experiment output, fitted object, model configuration, label or scoring protocol was changed. Incomplete figures are segregated as drafts and excluded from the paper bundle.

## Corrections

| Figures | Problem corrected | Current presentation |
|---|---|---|
| 1 | Point estimates lacked uncertainty | Probability-change intervals and direct paired method-difference intervals; all 21 conditions retained |
| 3, 10, 11, 21 | Missing cross-panel legends, crowded labels, ambiguous zero, incomplete panels mixed with final figures | Explicit numeric weights, shared no-hook marker, fixed [0,1] scale, five-point description, draft gating |
| 4 | Histogram density hid the eight actual cluster values; label structure was invisible | ECDF control comparison and cluster target-label composition; no manifold claim |
| 5 | Separate automatic axis limits could magnify numerical variation | Same orthonormal axes and unit-disk limits for every weight |
| 6 | Near-constant norm histograms created extreme density peaks | ECDFs; display-only normalization by same-question Linear displacement; original L2 RMS values in legends |
| 7 | Different vertical ranges complicated model comparison; legend too small | Common row scales, external behavior legend, explicit reconstruction definition and kernel-space caveat |
| 12–19 | Density spikes, inconsistent colors and panel-specific magnification | ECDFs, fixed numeric-weight colors, common direction axes, explicit activation location and display-only norm scaling |
| 20 | Tiny RMS differences looked large on a cropped axis; missing behavioral uncertainty | Full-scale RMS/cosine view, probability-change CIs and retained paired diagnostic contribution intervals |

## Scientific interpretation

- Flat Llama3B curves and nearly constant lambda-zero updates are real under these frozen settings. They are retained, not smoothed or visually exaggerated.
- The two rows of Figure 7 measure different quantities: relative inverse-reconstruction error and retained centered-kernel variance. Neither is a behavioral probability. Changing lambda changes the feature space, so decreasing retained fractions is not by itself evidence that steering worsens.
- Figure 4 clusters have a strong association with answer-label composition. They cannot be treated as unconfounded evidence of a behavioral manifold. Figure 5 projected variation or a better mixture-model fit does not prove true multimodality.
- Figure 6 measures the MCQ decision token; Figures 5 and 12–19 use cached completed-answer-label activations. Their geometries are not interchangeable.
- Qwen Hallucination's selected nonlinear weight has a supported paired gain in the existing diagnostic comparison, while similar RMS magnitude and the crossover favor a direction contribution. This remains an exploratory selected case; it does not establish general superiority.
- Pointwise uncertainty intervals and validation-selected maxima do not provide family-wide significance guarantees. Negative results and intervals crossing zero remain visible.
- Equal signed strengths across methods do not imply equal activation displacement. Axis normalization in Figures 6 and 13–19 only changes the display units, never the intervention.

## Data checks

| Check | Status | Evidence |
| --- | --- | --- |
| Figure 1 / main table recomputation | passed | All 42 plotted method means recomputed from 50 official-test question rows each; no change to mean or labels. Maximum candidate-logit probability error: 1.1102230246251565e-16 |
| Figures 5 / 12–19 direction geometry | passed | Recomputed every plotted cosine and common-axis projection from actual cached updates; all projections lie in the unit disk. Maximum reconstruction discrepancy: 8.881784197001252e-16 |
| Constant-distribution handling | passed | 31 model/concept/weight conditions have near-constant magnitudes. ECDF steps replace misleading density peaks; common [-1,1] projection axes avoid numerical-noise magnification. |
| Figure 7 coverage | passed | All 3 models × 7 behaviors × 9 weights are present. Reconstruction error is relative Frobenius error on uncentered activations; kernel variance fractions refer to different feature spaces across weights. Fixed 20 components do not attain 95% in any of these fits. |
| Response probabilities and no-hook semantics | passed | 36285 saved question/condition rows checked, including partial checkpoints. Every displayed complete summary matches raw means. Zero has no intervention and identical probability across methods. Maximum logit recomputation error: 2.22e-16. No monotonicity constraint is imposed. |
| Publication gating | passed | Incomplete response figures live only in figures/drafts and are excluded from the Overleaf compilation and ZIP. Figure 3 is ready only when its three fixed panels are complete. No favorable-panel reselection. |
