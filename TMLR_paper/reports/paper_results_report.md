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

## Release status

The main-table numbers are unchanged. All manuscript figures are withdrawn pending scientific and manuscript review; no figure is approved as an original-paper replacement. See ../../RELEASE_AUDIT.md. Supporting numerical diagnostics remain available, but deterministic rendering is not scientific approval.

## Methods and statistical limits

Training uses only fitting groups and is evaluated in-sample. Validation selects lambda from the already tested grid; selected maxima can be optimistic. Official Test is an exploratory fixed benchmark because previous diagnoses repeatedly inspected it. Published mean effects and 10,000 paired source-group bootstrap intervals are reused without changing the metric or labels. Pointwise intervals do not imply multiplicity-corrected superiority.

The primary interventions use last-token block outputs, natural residual-preserved Curveball updates, fixed 20 KPCA components, weighted degree-2 kernel with gamma=0.01 and inverse ridge=1. Native Curveball steps are training-only RMS-calibrated at lambda 1 and then held fixed across weights; actual evaluation displacements are not assumed equal to Linear. Llama2 uses its validated parenthesis prompt; Qwen/Llama3B use the frozen assistant-answer prefill. All methods within each model use identical prompts and labels. No Angular values with a mismatched multi-token scope are included.
