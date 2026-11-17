# Public release audit — 2026-11-17

Destination requested by the project owner: public `Natelie712/TMLR`, branch
`linlin`. This is a separate curated repository, not a push of the original
project's working tree or history.

## Included scope

- Qwen3-1.7B, Llama-3.2-3B-Instruct, Llama-2-7B-Chat; all seven CAA behaviors.
- 21 positive main-table model/behavior conditions, including null/adverse results.
- Full positive training/validation summaries and frozen exploratory test results.
- Completed independent negative-lambda search for all three models.
- Per-question probabilities, candidate logits and displacement records,
  losslessly gzip-compressed where large.
- Four main paper tables, appendix tables, and 18 completed vector PDF figures.
- Displacement/reconstruction diagnostics and fixed validation response curves.
- Scientific implementations, source configs, selection records, and file hashes.

Selection is based on completion and traceability, not favorable effect size.
The accepted historical figure review and scientific checks are included.

## Fresh verification performed for this release

1. Every packaged original file matches its source SHA-256; no source symlinks
   are shipped as substitutes for file contents.
2. All 21 positive main-table conditions are recomputed from bundled per-question
   records; 50 unique official questions per behavior and model are present.
3. All included negative test means are recomputed from their question records.
4. Positive and negative selections match the separately frozen validation
   configuration. No test-driven selection is introduced.
5. The split totals are 3,906 fit questions, 912 validation questions and 350
   official test questions per model, with source-group disjointness recorded.
6. All result CSV model identifiers are restricted to the three permitted models.
7. Weighted KPCA passes CPU projection/inverse parity, exact zero-step identity
   and finite nonzero-update checks. Scoring passes A/B agreement and non-A/B
   target inclusion checks.
8. The main table and all 18 figures re-render successfully from included caches.
   **All 18 re-rendered PNG files are byte-identical to their released originals.**
9. No file exceeds 95 MiB. No token/private-key patterns were detected in source
   and document files. Credentials and model downloads are outside this package.

The historical 254-input/189-fit audit certificate is preserved, but the large
fitted objects are not distributed or freshly revalidated by the portable
release check. That distinction is intentional.

## Excluded

Gemma experiments and source, Phi experimental results, all LLM weights and
training activation caches, large KPCA fit objects, credentials, Slurm runtime
logs, interrupted duplicate exports, previous unrefined figure backups, and
draft/incomplete figures. Shared legacy model adapters may still define Phi;
that is dependency source code, not a released Phi result.

Figures 2, 8 and 9 are not presented as completed experiments. Original numbering
and replacements are explained in the paper artifact inventory. No new inference
was run for packaging. No original result, training setting or Curveball update
was changed.

## Scientific status

These are verified computational artifacts, not a claim of statistical certainty.
Training is in-sample; validation is used for selection; Official Test is an
exploratory repeatedly inspected fixed benchmark. Bootstrap intervals are
pointwise, not adjusted for all comparisons. Equal native multipliers do not
imply equal activation displacement, and lambda=0 is not a nonlinear advantage.

The original experiment reports are preserved as historical snapshots. Old
pending-job prose in those snapshots does not override the completed source
records checked for this release.
