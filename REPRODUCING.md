# Reproduction levels

## 1. Audit and render the released artifacts

`python scripts/verify_release.py` checks all packaged hashes, allowed model
identifiers, main-table arithmetic, frozen validation selections, 350-question
test coverage, negative study completion, and the absence of symlinks or
large GitHub-incompatible files. It does not load a language model.

`bash scripts/reproduce.sh` runs that audit and then re-renders the main LaTeX
table and the 18 completed vector PDF figures using the included source and
numeric caches. Output goes to `reproduced/`. This is cache-based reproduction,
not newly run inference. Use Python 3.11 and the recorded analysis dependencies.
The archived `TMLR_paper/scripts/reproduce.sh` is the original CHPC workflow;
use the top-level wrapper for the portable cached release.

Original per-question probabilities/logits/displacements are in:

- `results/small_lambda_positive_20261110/per_question.csv.gz`: Qwen/Llama3B fit/validation.
- `results/small_lambda_validation_test_20261110/test_per_question.csv.gz`: their frozen test evaluation.
- `results/llama2_small_lambda_positive_20261111/per_question.csv.gz`: Llama2 splits.
- `TMLR_paper/data/negative_lambda_search/per_question.csv.gz`: independent negative study.

## 2. Inspect the algorithm without historical model caches

`scripts/check_algorithm.py` imports the released weighted KPCA and scorer,
fits a small synthetic contrastive dataset, verifies projection/inverse parity
against sklearn, zero-strength identity, nonzero finite updates, and A/B versus
all-option scoring agreement. This is a code-integrity test, not new research
evidence or a hyperparameter search.

## 3. Re-run model inference

The archived runners depend on the exact model snapshots, fitting activations,
fitted objects, local data paths and source hashes in their original manifests.
These large/gated assets are intentionally not on GitHub. Access the relevant
Hugging Face model with your own authorization, obtain the official CAA source
at the pinned commit, and regenerate the fitting caches using the retained
source-group split manifest. No repository file contains credentials.

The preserved historical code should be adapted through explicit path/config
changes in a new experiment namespace. Do not bypass hash assertions, overwrite
old outputs, silently change FP32/eager model inference or float64 KPCA, or
present regenerated caches as the original artifacts. Config snapshots preserve
the original native strengths, candidate-label rules and model-specific prompt
variants; merely using the same numeric alpha is insufficient.

Record new model revisions, source hashes, environment and GPU placement.
Validate extraction positions, label tokens, hidden-state indexing, hook scope,
no-hook parity, zero-step identity, and finite displacement before a sweep.
Select any new parameters on validation only. Official Test remains exploratory.
