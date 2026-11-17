# Curveball Steering: TMLR revision artifacts

Curated, completed experiments for **Qwen3-1.7B**, **Llama-3.2-3B-Instruct**,
and **Llama-2-7B-Chat** on seven official CAA behavior datasets.
This release includes positive steering, an independently selected negative
steering study, statistical intervals, displacement diagnostics, paper figures,
and the source implementations. It includes weak, negative, and null results.

**Gemma and Phi results are intentionally excluded.** No model weights, tokens,
raw training activation caches, or large fitted KPCA objects are distributed.

## Start here

- [Positive main results](TMLR_paper/tables/main_results.tex) / [CSV](TMLR_paper/data/main_results.csv)
- [Training / validation / test results](TMLR_paper/data/training_validation_test.csv)
- [Independent negative steering](TMLR_paper/reports/negative_lambda_search.md) / [CSV](TMLR_paper/data/negative_lambda_search/summary.csv)
- [Dataset statistics](TMLR_paper/data/dataset_statistics.csv)
- [Figure gallery](TMLR_paper/reports/figure_gallery.html) and [captions](TMLR_paper/reports/figure_captions.md)
- [Scientific interpretation and limits](TMLR_paper/reports/paper_results_report.md)
- [Release audit](RELEASE_AUDIT.md) and [file provenance](release_manifest.json)

## What is measured

The main quantity is the change in **all-option normalized target-label
probability** at the MCQ answer position, expressed in percentage points.
It is a CAA-derived extension, not the unmodified original CAA A/B scorer.
All official questions and behavior labels are retained. Survival Instinct
uses the original target definition; its sign is not changed based on its name.

| Model | Decoder block (zero-based) | Hidden-state index |
|---|---:|---:|
| Qwen3-1.7B | 18 | 19 |
| Llama-3.2-3B-Instruct | 9 | 10 |
| Llama-2-7B-Chat | 13 | 14 |

Interventions act at the last real token. Curveball uses the residual-preserved
update `h + inverse(project(h) + step * direction) - inverse(project(h))`.
No per-token displacement normalization or Linear-anchored correction is used.
The weighted kernel is `1 + 2s + lambda*s^2`, where `s = 0.01*x@y`.
Lambda is the quadratic kernel weight, not the steering strength. The grid is
`0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1`.
The frozen positive selection treats lambda=1 as a reference outside its
selection grid; the independent negative study selects across its specified
full grid. Exact selections and native strengths are included in the configs.

Strength calibration uses training-only probes; it does not guarantee equal
actual displacement at evaluation positions or across kernel weights. Lambda=0
is the linear-kernel KPCA limit and is not necessarily identical to CAA Linear.

## Core code

| Responsibility | File |
|---|---|
| Weighted polynomial KPCA fitting and direction | [weighted_small.py](experiments/small_lambda_positive_20261110/weighted_small.py) |
| Projection, inverse, residual preservation, strength calibration | [methods_b.py](experiments/caa_official_seven_behaviors_four_models/methods_b.py) |
| Label parsing, all-option scoring, original A/B comparison | [scoring.py](experiments/caa_official_seven_behaviors_four_models/scoring.py) |
| Last-token intervention and per-question geometry | [original-style runner](experiments/qwen_original_style_20261105/run.py) |
| Positive training / validation execution | [positive runner](experiments/small_lambda_positive_20261110/execute.py) |
| Llama 2 execution | [Llama 2 runner](experiments/llama2_small_lambda_positive_20261111/execute.py) |
| Independent negative lambda search | [negative_search.py](TMLR_paper/scripts/negative_search.py) |
| Fixed response curves | [response_gpu.py](TMLR_paper/scripts/response_gpu.py) |
| Paper plotting | [refined_figures.py](TMLR_paper/scripts/refined_figures.py) |

Scientific source files are preserved byte-for-byte. Shared historical adapters
may mention other architectures; no Phi/Gemma experimental results are included.
Some archived runners and manifests contain CHPC paths and require the original
external caches. They are an execution record, not a claim that a fresh clone
contains every dependency for re-running LLM extraction. See [reproduction](REPRODUCING.md).

## Reproduce without a GPU

```bash
python scripts/verify_release.py
# For figure/table rendering, install the analysis dependencies first:
pip install -r requirements-analysis.txt
bash scripts/reproduce.sh
```

The verification command uses only the Python standard library. Reproduction
renders the main table and all completed paper figures from the included
numeric caches into `reproduced/`, preserving the distributed originals.
Large per-question CSVs are losslessly gzip-compressed and readable directly
with Python `gzip.open(..., 'rt')` or pandas `read_csv(..., compression='gzip')`.

## Interpretation limits

- Training evaluation is in-sample. Validation was used for lambda selection.
- Official Test has been inspected repeatedly and is an **exploratory fixed
  benchmark**, not a fresh independent held-out confirmation.
- Pointwise paired source-group bootstrap 95% intervals use 10,000 resamples.
  They are not multiplicity-corrected claims of overall superiority.
- Do not combine separately selected positive/negative lambda values and label
  the resulting range as one fixed method configuration.
- The positive/negative selection studies and the symmetric strength-response
  diagnostic have distinct provenance. Do not substitute old negative endpoints
  that used a different calibration.
- Dataset, model, scorer, fixed 20-component KPCA, and weighted-kernel ablations
  differ from the original manuscript setup. This is not an exact reproduction
  of every NeurIPS paper setting.

The historical reports are dated snapshots. This release audit supersedes old
scheduler status text. Figures 2, 8 and 9 are not reproduced/published as completed
experiments; see the numbering inventory. No claim is made about unavailable
open-ended judge evaluation or VAE experiments.

CAA dataset provenance: [nrimsky/CAA](https://github.com/nrimsky/CAA), pinned
commit `5dabbbd9a0bca5f25e174501e959de378806aa48`. The original datasets are not
redistributed here; the source manifest records their paths and hashes.
This release does not add a new license grant to inherited source or data.
