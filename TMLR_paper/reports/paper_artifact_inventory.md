# Paper artifact inventory

Source: `paper/_NeurIPS26__kernelpca_steering.pdf`, SHA256 abf9780fd1fd7be44c538d495f386a7a5fbc79ac4d975e85a824caf3e942a403. The uploaded manuscript contains Tables 1–4 and Figures 1–20. Figure21 is newly added for the third model.

| Artifact / section | Scientific question | Original dependencies | Revision and source | GPU needed? | CPU sufficient? | Retain unchanged? |
| --- | --- | --- | --- | --- | --- | --- |
| Figure 1 / 1 overview | Algorithm overview and empirical motivation | Llama-3.2-1B-Instruct / Phi-3.5-mini; synthetic behavioral and tone datasets | Replace empirical overview with all 21 CAA model-behavior comparisons; retain Algorithms 1/2 conceptually; remove tone-judge panel and old curvature linkage. Sources: Main-results CSV | False | True | False |
| Figure 2 / 3 / B | Geometric distortion and synthetic curvature effects | A: conceptual; B: old VAE ensembles; C/D: independent hypersphere simulation | A/C/D eligible for retention only after provenance confirmation; B requires a new-data VAE study and is not reproduced. Do not substitute kNN detour ratios for VAE pullback distances. Sources: evals/toy_models_steering.py; VAE notebooks; manuscript PDF; no verified matching result archive found | Only B may need GPU after cost pilot | False | False |
| Figure 3 / 4.1 | Steering probability response to signed strength | Llama-3.2-1B-Instruct / Phi-3.5-mini; synthetic behavioral and tone datasets | Five-point validation diagnostics with true no-hook; fixed signed multiplier of the existing positive native step. Representative Qwen Hallucination, Llama3B Hallucination, Llama2 Myopic Reward panels, chosen from prior evidence and labelled exploratory. Sources: Cached validation no-hook/+1; Qwen prior negative steps differ and cannot be silently merged | True | False | False |
| Figure 4 / 5 | Local linear direction variation vs random partitions | Old Corrigibility negative clusters and paired positives | Qwen Hallucination plus Survival Instinct contrast; k=8 negative group-centroid clusters, matching real positive pairs; 500 size-matched random group partitions. Group averaging is explicitly documented. Sources: Fit paired activations | False | True | False |
| Figure 5 / 5 | Multimodal point-wise steering directions? | Old Corrigibility KPCA perturbations | Actual residual-preserved updates; unit direction projection on Linear and common orthogonal PC; report variability and descriptive projection-only GMM diagnostic without assuming multimodality. Sources: Frozen fits + paired validation activations | False | True | False |
| Figure 6 / 5 | Adaptive magnitude vs fixed Linear displacement | Old Corrigibility activations | MCQ decision-token displacement distributions from actual saved forwards; contrast Qwen Hallucination, Llama3B invariance, Llama2 nonzero-weight cases. Keep paired-label activation diagnostics separate. Sources: Saved per-question displacement logs | False | True | False |
| Figure 7 / D.2 | Reconstruction and kernel sensitivity | Old humorous trait reconstruction grid | Existing quadratic-weight reconstruction diagnostics and actual retained centered-kernel variance. No claim that fixed 20 components automatically attain 95%. This is a weight ablation, not a new degree/grid search. Sources: Frozen 189 fitting objects; saved decision-location recon errors | False | True | False |
| Figure 8 / E.2 | Open-ended classifier score vs strength | Llama-3.2-1B-Instruct / Phi-3.5-mini; synthetic behavioral and tone datasets | Remove: no corresponding new open-ended classifier evaluation. Reserve original numbers and explain gaps. Sources: Old figure is not valid for the CAA MCQ revision | False | False | False |
| Figure 9 / E.3 | Open-ended classifier score vs strength | Llama-3.2-1B-Instruct / Phi-3.5-mini; synthetic behavioral and tone datasets | Remove: no corresponding new open-ended classifier evaluation. Reserve original numbers and explain gaps. Sources: Old figure is not valid for the CAA MCQ revision | False | False | False |
| Figure 10 / E.4 | Behavior target-label probability vs signed strength | Llama-3.2-1B-Instruct / Phi-3.5-mini; synthetic behavioral and tone datasets | All seven CAA validation curves for Qwen3-1.7B; fixed five points, not a dense sweep. Sources: Cached no-hook/+1 and new response diagnostics | True | False | False |
| Figure 11 / E.5 | Behavior target-label probability vs signed strength | Llama-3.2-1B-Instruct / Phi-3.5-mini; synthetic behavioral and tone datasets | All seven CAA validation curves for Llama-3.2-3B-Instruct; fixed five points, not a dense sweep. Sources: Cached no-hook/+1 and new response diagnostics | True | False | False |
| Figure 12 / E.6 | Direction cosine distributions across concepts | Old eight behavioral/tone attributes | Three models by seven CAA concepts; residual-preserved update cosine distributions on paired validation negatives. Sources: CPU update caches | False | True | False |
| Figure 13 / F | Local cosine, update projection, magnitude diagnostics | Corrigibility on Llama-3.2-1B | Replace with Corrigibility on all three current models; completed-answer paired geometry is labelled separately from MCQ decision geometry. Sources: Fit activations, frozen kernels, saved MCQ updates | False | True | False |
| Figure 14 / F | Local cosine, update projection, magnitude diagnostics | Power-seeking on Llama-3.2-1B | Replace with AI Coordination on all three current models; completed-answer paired geometry is labelled separately from MCQ decision geometry. Sources: Fit activations, frozen kernels, saved MCQ updates | False | True | False |
| Figure 15 / F | Local cosine, update projection, magnitude diagnostics | Self-awareness on Llama-3.2-1B | Replace with Hallucination on all three current models; completed-answer paired geometry is labelled separately from MCQ decision geometry. Sources: Fit activations, frozen kernels, saved MCQ updates | False | True | False |
| Figure 16 / F | Local cosine, update projection, magnitude diagnostics | Wealth-seeking on Llama-3.2-1B | Replace with Myopic Reward on all three current models; completed-answer paired geometry is labelled separately from MCQ decision geometry. Sources: Fit activations, frozen kernels, saved MCQ updates | False | True | False |
| Figure 17 / F | Local cosine, update projection, magnitude diagnostics | Humor on Llama-3.2-1B | Replace with Survival Instinct on all three current models; completed-answer paired geometry is labelled separately from MCQ decision geometry. Sources: Fit activations, frozen kernels, saved MCQ updates | False | True | False |
| Figure 18 / F | Local cosine, update projection, magnitude diagnostics | Sadness on Llama-3.2-1B | Replace with Sycophancy on all three current models; completed-answer paired geometry is labelled separately from MCQ decision geometry. Sources: Fit activations, frozen kernels, saved MCQ updates | False | True | False |
| Figure 19 / F | Local cosine, update projection, magnitude diagnostics | Excitement on Llama-3.2-1B | Replace with Refusal on all three current models; completed-answer paired geometry is labelled separately from MCQ decision geometry. Sources: Fit activations, frozen kernels, saved MCQ updates | False | True | False |
| Figure 20 / F / new case study | Original Rudeness diagnostics; revised Hallucination case | Rudeness on Llama-3.2-1B | Replace tone panel with existing Qwen Hallucination direction/magnitude crossover and validation/test contrasts. Counterfactuals are diagnostics, not a new algorithm. Sources: Verified qwen_factorial_contrasts.csv and paired intervals / test_summary | False | True | False |
| Figure 21 / new Appendix E | Behavior target-label probability vs signed strength | New third-model panel | All seven CAA validation curves for Llama-2-7B-Chat; fixed five points, not a dense sweep. Sources: Cached no-hook/+1 and new response diagnostics | True | False | False |
| Table 1 / 4.1 | Does steering change target behavior probability? / trait judge scores? | Llama-3.2-1B-Instruct / Phi-3.5-mini; synthetic behavioral and tone datasets | Three CAA models, seven behaviors, validation-selected Curveball vs CAA Linear; explicit lambda and paired difference CI. Omit incompatible historical Angular. Sources: Completed small_lambda_positive, small_lambda_validation_test and llama2_small_lambda_positive summaries | False | True | False |
| Table 2 / 5 | Does local magnitude track paired target distance? | Old paired activations, eight concepts | Real CAA positive/negative pairs; validation source-group Spearman CI and family-corrected permutation tests, plus near-constant flags. Sources: Official generate activation caches and frozen fit objects | False | True | False |
| Table 3 / A.2 | How much data and how many pairs? | Llama-3.2-1B-Instruct / Phi-3.5-mini; synthetic behavioral and tone datasets | Count question records, unique groups and activation samples separately; document excluded generate/test overlaps. Sources: Frozen split_manifest and official JSON | False | True | False |
| Table 4 / F | How consistent are per-prompt contrast vectors? | Old eight concepts including tones | Cross-source-group cosine of genuine CAA contrast vectors; no tone-vs-behavior claim. Sources: Official activation caches | False | True | False |

## Numbering and modality changes

Tables1–4 and Figures1–7,10–16 keep their scientific slots, with explicitly revised datasets/metrics. Figures8–9 are removed, retaining numbering gaps in the artifact inventory. Figures17–19 replace absent tone panels with Survival Instinct, Sycophancy and Refusal. Figure20 becomes the Qwen Hallucination case study. Figure21 adds Llama2 probability curves. Figure2 is not marked reproduced. Final manuscript inclusion order must follow the supplied numbering map.

Appendix A/H synthetic generation templates must be removed or moved to historical background. Current CAA datasets were not generated with those templates. The caption repeated on Figures7–11 does not describe their content accurately; their section headings determine the artifact purpose.

## Algorithm and interpretation fidelity

The Section5 printed perturbation displacement `inverse(phi(a)+epsilon*d)-a` includes preimage reconstruction error. This revision instead measures Algorithm2 actual update `inverse(phi(a)+alpha*d)-inverse(phi(a))`, with the original activation residual preserved. This corrects the diagnostic definition, not the steering algorithm. Normalized direction projections separate direction from magnitude; raw 2D update projections mix both.

The manuscript has inconsistent component-selection descriptions. All completed experiments use fixed20 components. Actual retained centered-kernel trace fractions are computed for every fit; no automatic95% assertion is made. Polynomial gamma=.01, coef0=1, degree2, inverse ridge1 and frozen native strength follow the completed experiments, not an unperformed AppendixD reconstruction grid.

<!-- CURRENT_COMPLETION -->

## Current generated-artifact status

| Artifact | Status | File |
| --- | --- | --- |
| table 1 | complete | tables/main_results.tex |
| table 2 | complete | tables/magnitude_correlation_selected.tex |
| table 3 | complete | tables/dataset_statistics.tex |
| table 4 | complete | tables/cross_prompt_consistency_main.tex |
| figure 1 | complete | figures/figure_1.pdf |
| figure 2 | not reproduced: provenance / cost assessment only |  |
| figure 3 | complete | figures/figure_3.pdf |
| figure 4 | complete | figures/figure_4.pdf |
| figure 5 | complete | figures/figure_5.pdf |
| figure 6 | complete | figures/figure_6.pdf |
| figure 7 | complete | figures/figure_7.pdf |
| figure 8 | removed: open-ended modality absent |  |
| figure 9 | removed: open-ended modality absent |  |
| figure 10 | complete | figures/figure_10.pdf |
| figure 11 | complete | figures/figure_11.pdf |
| figure 12 | complete | figures/figure_12.pdf |
| figure 13 | complete | figures/figure_13.pdf |
| figure 14 | complete | figures/figure_14.pdf |
| figure 15 | complete | figures/figure_15.pdf |
| figure 16 | complete | figures/figure_16.pdf |
| figure 17 | complete | figures/figure_17.pdf |
| figure 18 | complete | figures/figure_18.pdf |
| figure 19 | complete | figures/figure_19.pdf |
| figure 20 | complete | figures/figure_20.pdf |
| figure 21 | complete | figures/figure_21.pdf |
