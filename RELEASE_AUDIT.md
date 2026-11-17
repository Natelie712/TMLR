# Public release audit and figure withdrawal — 2026-11-17

Repository: public `Natelie712/TMLR`, branch `linlin`.

## Current approved distribution scope

- Qwen3-1.7B, Llama-3.2-3B-Instruct, and Llama-2-7B-Chat; seven CAA behaviors.
- All 21 positive main-table conditions, including weak, null, and adverse effects.
- Training / Validation / exploratory Official Test summaries, separately frozen negative-lambda selection, paired intervals, and supporting per-question probabilities/logits/displacements.
- Scientific source, frozen configurations, supporting numeric diagnostics, and table reproduction instructions.
- No Gemma/Phi experimental results, model weights, credentials, raw training activations, or large fitted KPCA objects.

## Figure withdrawal

At the project owner's request, all 18 figure PDFs and their 18 PNG previews,
the gallery, captions, figure-review exports, and LaTeX figure-inclusion file
have been removed from the current branch. They were candidate revision figures,
not approved replacements for the original manuscript. The earlier description
of figure readiness was too strong: successful deterministic rendering is a
computational check, not sufficient scientific or manuscript review.

The portable reproduction command now renders the main table only. Archived
plotting source and supporting numeric caches remain for traceability, not as
approved manuscript figures. Local historical experiment outputs are unchanged.
This is a normal removal commit; earlier versions remain in Git history.

## Computational verification

The independent post-publication audit verified 271 originally published files
against the remote Git blobs; recomputed 233,296 per-question probabilities from
candidate logits, 494 positive and 348 negative summary rows; and reproduced
180 bootstrap CI bounds using the original 10,000-resample procedure. Maximum
probability error was 4.44e-16, maximum summary error 1.42e-14, and CI endpoints
matched exactly. Source-group split intersections were empty.

For this figure-withdrawal amendment, the numerical results and existing result
tables are preserved byte-for-byte. The retained main-table and negative-test
checks are rerun, and table-only rendering must exactly reproduce main_results.tex.
File hashes in release_manifest.json reflect this amendment. Historical source
hashes remain available even when release documentation has been corrected.

No new LLM inference is claimed by this packaging audit. Large historical fitted
objects are not distributed or freshly revalidated. Computational consistency
does not establish universal scientific validity. Official Test remains a
repeatedly inspected exploratory benchmark; intervals are pointwise and not
multiplicity-adjusted. Lambda zero is not evidence of nonlinear benefit.
