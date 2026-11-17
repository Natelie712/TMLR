# Curveball TMLR empirical revision

Start with `reports/paper_results_report.html` and `data/artifact_completion.csv`.
They distinguish completed results from pending GPU response diagnostics.
For visually reviewed figures, open `reports/figure_gallery.html`; the scientific
checks and interpretation limits are in `reports/figure_refinement_review.md`.

The revision uses Qwen3-1.7B (block 18), Llama-3.2-3B-Instruct (block 9), and
Llama-2-7B-Chat (block 13), seven official CAA behaviors, frozen source-group
splits, and last-token natural residual-preserved Curveball updates.
Weights and native strengths are inherited without additional test tuning.
The primary metric is all-option normalized target-label probability, rather
than the unmodified CAA A/B score. Official Test is an exploratory fixed
benchmark because it was inspected during previous investigations.

## Files for manuscript preparation

- `tables/main_results.tex`: Table 1, including paired differences and CIs.
- `tables/magnitude_correlation_selected.tex`: Table 2 validation summary.
- `tables/dataset_statistics.tex`: Table 3, questions/groups/activations.
- `tables/cross_prompt_consistency_main.tex`: Table 4 validation summary.
- `tables/*_full.tex`: multipage fitting/validation geometry tables.
- `tables/appendix_*`: complete split results and evaluated kernel-weight grids.
- `figures/figure_*.pdf`: vector figures with embedded TrueType fonts.
- `reports/figure_captions.md`: manuscript captions and domain qualifications.
- `reports/overleaf_check/overleaf_preview.pdf`: compiled artifact preview.
- `data/overleaf_artifacts.zip`: tables, vector PDFs, captions, completion
  inventory, and `scripts/overleaf_preview.tex`. Upload the contents to Overleaf
  and select that TeX file as the main document. It is a preview, not the paper.

Response figures become complete only when all required GPU conditions finish.
Incomplete previews live exclusively in `figures/drafts/` and are excluded
from the Overleaf ZIP and compiled preview. Consult the completion inventory
for current status. Figure 2 remains an audit/cost assessment; Figures 8–9
are removed because no corresponding open-ended evaluation is present.
Original numbering is documented in `reports/paper_artifact_inventory.md`.

## Reproduction

Run from the project root using the existing project virtual environment:

```bash
bash TMLR_paper/scripts/reproduce.sh cpu
bash TMLR_paper/scripts/reproduce.sh freeze
bash TMLR_paper/scripts/reproduce.sh report
reproductions/remy_native_linear_20261012/venv/bin/python TMLR_paper/scripts/verify_artifacts.py
```

The CPU command regenerates tables, cache-only geometry if not already
complete, reports, and the Overleaf bundle. It does not load an LLM or submit
a GPU job. Requires the historical caches referenced in
`data/revision_config.json`, NumPy, PyTorch, scikit-learn, SciPy, joblib,
Matplotlib, markdown-it-py, and `pdflatex` with booktabs/graphicx/longtable.
Model inference additionally uses the frozen local Hugging Face snapshots
and Transformers. No token or network access is required for cached models.

`scripts/responses.slurm` initially ran as Notchpeak job **16852828** and was
preempted after Qwen3 finished and Llama3B partly finished. Recovery job
**16868125** resumes its checkpoints on one A40 48GB. Do not submit a duplicate. The job runs the
three models sequentially with FP32/eager inference, TF32 disabled, and the
fixed validation multiplier grid {-1, -0.5, 0, 0.5, 1}. It replays a small
sample of saved endpoints before reusing 0/+1, checks candidate IDs/targets,
last-token hook positions, finite logits and zero-step identity, and saves
each question/condition immediately. A failed parity check stops that run.
Historical Qwen negative KPCA steps used separate calibration; they are not
silently reused as the negative endpoint of the new symmetric grid.

For recovery on a suitable allocated GPU, after checking no job is active:

```bash
bash TMLR_paper/scripts/reproduce.sh gpu qwen3 llama3b llama2
```

Completed conditions are skipped on resume. Per-model locks prevent duplicate
local writers. GPU status, replay checks and raw logs are under
`data/response_curves/` and `data/execution/`. The job refreshes reports,
figures and the Overleaf package after each completed model. A stopped or
failed scientific check requires review; closing a laptop does not stop Slurm.

An independent CPU recovery controller was submitted as **16868203** on
November 12, 2026. It watches the experiment, resumes scheduler preemption,
node/boot failures and wall-time interruption, and rotates among compatible
Notchpeak A40/A6000/L40 48GB GPUs. It preserves the exact frozen experiment
script and avoids duplicate submissions. A dependent CPU successor protects
the controller's own continuity. CPU and GPU jobs remain subject to scheduling;
this mechanism cannot guarantee immediate resource availability.

Automatic recovery stops for Python tracebacks, scientific failures, OOM,
deliberate GPU cancellation, ambiguous submission, or an exhausted 72-hour /
12-retry window. State and the full event history are recorded in
`data/execution/automatic_recovery.json`. To disable recovery, set that file's
`status` to `stopped` before cancelling the controller and its recorded
successor; do not cancel the user's IDE allocation. Recovery operates only
on the identified TMLR experiment jobs.

## Storage and scientific controls

`scripts/` and this README are ordinary source files in the project home
directory. `data/`, `figures/`, `tables/`, and `reports/` are symlinks into
`/scratch/rai/vast1/u1557683/project_storage/kpca_steering/results/TMLR_paper/`.
Access all new outputs through `TMLR_paper/`; the links avoid the home quota.
The Overleaf ZIP contains actual files and does not depend on these symlinks.

Existing representations, model fits, labels, scorers, and historical outputs
remain unchanged. The 254-file freeze and table-source hashes are verified
separately. Completed-answer-label geometry is distinguished from MCQ
decision-token geometry. Reconstruction error is not interpreted as the
residual-preserved steering update. Fixed 20 KPCA components are not described
as satisfying a 95% variance rule. Pointwise CIs are not multiplicity-corrected
claims of overall superiority. No GitHub push, VAE training, or judge API run
is part of this execution.
