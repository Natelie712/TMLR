"""Attach evidence-based final conclusions without changing any result values."""
import csv
from support import *
from markdown_it import MarkdownIt

def main():
    cfg=verify();assert load(OUT/'completion_status.json')['complete']
    assert load(OUT/'post_geometry_status.json')['passed']
    tests=list(csv.DictReader((OUT/'test_summary.csv').open()));rows=list(csv.DictReader((OUT/'test_per_question.csv').open()))
    assert len(rows)==3200 and sum(r['selected']=='True' for r in tests)==14
    for key in ['qwen3','llama3b']:
        assert len({(r['concept'],r['index']) for r in rows if r['model']==key})==350
        run=Path(load(OUT/f'location_{key}.json')['path']);assert load(run/'manifest.json')['status']=='complete'
        assert load(run/'correctness_checks.json')['passed']
    notes='''## Confirmed findings

The full new GPU run completed successfully: both models, all seven behaviors, all350 official questions per model,14 validation-selected configurations and3,200 question-condition records. Selection and exact native steps were frozen before diagnostic or test inference. Repeated methods were deduplicated when the selected weight was0. Historical records are unchanged.

**Qwen Hallucination: direction, not increased magnitude, explains the local improvement.** Native lambda0 and0.01 have RMS updates257.486 and257.991. Their per-question magnitudes are not identical, so we ran a crossed2x2 diagnostic on all195 validation questions. Replacing only the direction at the original lambda0 magnitude improves target probability by4.787 pp (paired95% CI3.098 to6.670); replacing only the magnitude at the original direction changes it by-0.164 pp (CI-0.348 to-0.035). Averaging the two conditional paths assigns+4.766 pp to direction (CI3.085 to6.646) and-0.186 pp to magnitude (CI-0.374 to-0.059), summing to the observed+4.580 pp. These are local diagnostic counterfactuals; they do not modify the primary natural-magnitude method. For lambda0.05 and1, both magnitude and direction differ from lambda0; their individual causal contributions were not separately identified.

**Statistical evidence is positive within the analyzed data, with explicit reuse limits.** The original lambda0.01-versus0 paired interval is[2.940,6.432] pp. The post-hoc approximate112-comparison max-statistic band is[1.496,7.664] pp. The latter covers the specified two-model/seven-behavior/eight-nonzero-weight family, not all historical research choices. It does not make the selected validation estimate unbiased. On the frozen but previously inspected official50-question Hallucination test, Linear gives+8.968 pp, lambda0+19.953 pp, selected lambda0.01+28.955 pp, and lambda1+2.808 pp. Selected lambda0.01 exceeds lambda0 by9.002 pp (paired95% CI3.813 to15.057) and Linear by19.988 pp (CI11.398 to29.375). This is exploratory fixed-benchmark evidence, not an independent confirmatory holdout.

**Llama's near-invariance is empirically consistent with weak nonlinear terms at the fixed activation/kernel scale.** All hooks produced nonzero updates, all stored coefficients were applied, and fresh probability replays passed. Across the full validation sets, the range of lambda-dependent mean changes is only0.000121 to0.020687 pp. At lambda1 the centered forward quadratic/linear kernel-norm ratio is0.0248 to0.0388; the inverse training-kernel ratio is0.000697 to0.001047. More directly, on16 fixed validation groups per behavior, the mean quadratic/linear component-norm ratio of the actual inverse update is only0.000160 to0.000655 (0.016% to0.065%). Mean lambda1-versus-lambda0 update cosines are0.999126 to0.99999994. The largest observed relative update change is4.483% in Sycophancy, so this is near-invariance, not exact equality. Some tiny validation rankings are comparable to numerical replay differences and should not be interpreted as a reliably optimal lambda. These findings apply to this block, centering, coefficient, fitting data and decoder; they do not establish that Llama has no nonlinear representations.

**Interpretation limits.** The same nominal gamma0.01 and lambda grid do not impose the same effective nonlinearity across models. For Qwen Hallucination, even lambda0.01 has a mean inverse quadratic/linear update ratio of0.478. Forward projection, fitted latent direction and inverse mapping all vary with lambda; the crossed update experiment identifies the final activation-space geometry contribution, not which fitting subcomponent alone caused it. High preimage reconstruction error is distinct from residual-preserved intervention error: zero-strength identity checks passed. No coherence claim or new hyperparameter search follows from these MCQ diagnostics. All weak/negative behaviors remain in the complete comparison table. No test-driven retuning or GitHub push occurred.

[Inverse-update component decomposition](inverse_update_component_summary.csv), [per-question components](inverse_update_components_per_question.csv), and [floating-point cast errors](cast_error_per_question.csv) supplement the geometry diagnostics. The largest measured relative cast error was5.79e-8; component decomposition reconstructed the unrounded update within1.29e-10 relative error. The GPU replay maximum probability error was below1e-4 for both models.
'''
    (OUT/'scientific_conclusions.md').write_text('# Scientific conclusions\n\n'+notes)
    md=(OUT/'report.md').read_text();marker='## Confirmed findings'
    if marker in md:md=md[:md.index(marker)].rstrip()+'\n'
    md+='\n'+notes;(OUT/'report.md').write_text(md)
    (OUT/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><style>body{font:16px/1.6 system-ui;margin:32px}table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #ccd;padding:7px}img{max-width:100%}</style>'+MarkdownIt('commonmark').enable('table').render(md)+'</html>')
    verify()
    save(OUT/'integrity_check.json',dict(passed=True,protected_files_unchanged=len(cfg['protected_inputs']),official_questions_per_model=350,test_records=3200,selected_conditions=14,
        supplemental_analysis_sources={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'post_geometry.py',Path(__file__)]}))
    print(json.dumps(load(OUT/'integrity_check.json')),flush=True)

if __name__=='__main__':main()
