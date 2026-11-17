# Negative steering: independent lambda selection

This study selects a negative-steering kernel weight independently for each of three models and seven CAA behaviors. Positive selections and historical results remain unchanged.

The fixed grid is {0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1}. Models, blocks, checkpoints, prompt variants, all-option scorer, last-token natural residual update, fits and Linear vectors are reused unchanged. Linear uses alpha=-1. Curveball uses the negative of the frozen positive native step calibrated with lambda1 on training-only probes. No negative-strength recalibration or strength search is performed. Equal multipliers do not imply equal actual displacement.

Select the smallest validation row-mean Delta P(target); exact ties choose the smaller numerical lambda. A positive minimum is reported as failure to suppress, not an improvement. Selection uses neither the official test nor training evaluation. Freeze all seven choices for each model before that model's official-test evaluation. Fit (in-sample) and official test then evaluate Linear, lambda0, selected negative lambda and lambda1. Row means and paired source-group bootstrap intervals follow the positive study. All 350 official questions and original labels remain unchanged. Test results are exploratory fixed-benchmark results, and validation maxima/minima are subject to selection bias.

Checkpoint each question/condition. Reuse response-curve endpoints only after checking frozen revision provenance, exact signed native parameter, target, candidate token IDs, group and a fresh positive/no-hook replay. Historical negative calibration results are not substituted. Verify every kernel's zero-step identity, projection/inverse parity and negative hook position before scoring. Store logits, candidate probability mass, native step, actual displacement and reconstruction diagnostics. Numerical assertions stop execution and are not automatically retried.

Outputs: `data/negative_lambda_search/`, `reports/negative_lambda_search.html`, and `tables/negative_lambda_selected.tex`. No GitHub push.
