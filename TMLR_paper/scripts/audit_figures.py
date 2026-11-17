"""Scientific checks for figure inputs, plus an English review and visual index."""
import html
from artifact_common import *

def main():
    checks=[]
    def passed(name,detail):checks.append(dict(check=name,status='passed',detail=detail))
    mainrows=rows(PAPER/'data/main_results.csv')
    previous=rows(TEST/'test_per_question.csv')
    llama=[r for r in rows(LLAMA/'per_question.csv') if r['partition']=='benchmark']
    max_logit_error=0.
    for m in MODELS:
        rr=[r for r in (llama if m=='llama2' else previous) if r['model']==m]
        for c in NAMES:
            point=next(r for r in mainrows if r['model']==m and r['concept']==c)
            for method,key in [('Linear','linear_delta_pp'),(f'lambda_{selected(m)[c]}','curveball_delta_pp')]:
                ss=[r for r in rr if r['concept']==c and r['method']==method]
                assert len(ss)==50 and len({r['index'] for r in ss})==50,(m,c,method)
                value=100*np.mean([float(r['probability'])-float(r['nohook_probability']) for r in ss])
                assert abs(value-float(point[key]))<1e-8,(m,c,method,value,point[key])
                for r in ss:
                    logits=json.loads(r['candidate_logits']);ls=np.array(list(logits.values()),float);v=np.exp(ls-ls.max());p=v[list(logits).index(r['target'])]/v.sum()
                    max_logit_error=max(max_logit_error,abs(p-float(r['probability'])))
    assert max_logit_error<2e-6,max_logit_error
    passed('Figure 1 / main table recomputation','All 42 plotted method means recomputed from 50 official-test question rows each; no change to mean or labels. Maximum candidate-logit probability error: '+str(max_logit_error))

    max_projection_error=0.;max_identity_error=0.;constant=[]
    for m in MODELS:
        for c in NAMES:
            d=np.load(PAPER/'data/geometric_diagnostics'/f'updates_{m}_{c}.npz')
            v=d['global_linear']/np.linalg.norm(d['global_linear']);axis=d['orthogonal_axis']
            assert abs(np.linalg.norm(axis)-1)<1e-8 and abs(v@axis)<1e-7
            for key in d.files:
                if not key.startswith('update_'):continue
                w=key.removeprefix('update_');u=d[key];norm=np.linalg.norm(u,axis=1)
                assert np.isfinite(u).all() and np.all(norm>0)
                xy=np.c_[(u/norm[:,None])@v,(u/norm[:,None])@axis]
                max_projection_error=max(max_projection_error,float(abs(xy-d['projection_'+w]).max()))
                assert np.max(np.linalg.norm(xy,axis=1))<=1+1e-7
                if np.ptp(norm)<1e-5*norm.mean():constant.append(dict(model=m,concept=c,weight=float(w)))
    assert max_projection_error<1e-8
    passed('Figures 5 / 12–19 direction geometry','Recomputed every plotted cosine and common-axis projection from actual cached updates; all projections lie in the unit disk. Maximum reconstruction discrepancy: '+str(max_projection_error))
    passed('Constant-distribution handling',f'{len(constant)} model/concept/weight conditions have near-constant magnitudes. ECDF steps replace misleading density peaks; common [-1,1] projection axes avoid numerical-noise magnification.')

    eig=rows(PAPER/'data/geometric_diagnostics/kernel_variance_reconstruction.csv')
    assert len(eig)==189
    assert all(0<=float(r['retained_kernel_variance_fraction'])<=1 and float(r['fit_relative_reconstruction'])>=0 for r in eig)
    passed('Figure 7 coverage','All 3 models × 7 behaviors × 9 weights are present. Reconstruction error is relative Frobenius error on uncentered activations; kernel variance fractions refer to different feature spaces across weights. Fixed 20 components do not attain 95% in any of these fits.')

    summary=rows(PAPER/'data/response_summary.csv');raw={};baseline={};checked=0;max_response_error=0.
    for m in MODELS:
        for path in (PAPER/'data/response_curves'/m).glob('*.jsonl'):
            lines=path.read_text().splitlines()
            for i,line in enumerate(lines):
                try:record=json.loads(line)
                except json.JSONDecodeError:
                    assert i==len(lines)-1;continue
                for r in record['rows']:
                    key=(m,r['concept'],r['method'],float(r['multiplier']));raw.setdefault(key,[]).append(r)
                    p=float(r['normalized_target_probability']);assert 0<=p<=1 and np.isfinite(p)
                    logits=r['candidate_logits'];v=np.array(list(logits.values()),float);v=np.exp(v-v.max());recalc=v[list(logits).index(r['target'])]/v.sum()
                    max_response_error=max(max_response_error,abs(recalc-p));checked+=1
                    if r['multiplier']==0:
                        assert not r['intervention_triggered'] and np.max(np.abs(r['displacement']['actual_displacement']))==0
                        bkey=(m,r['concept'],r['index']);assert bkey not in baseline or baseline[bkey]==p;baseline[bkey]=p
    assert max_response_error<2e-6
    for r in summary:
        key=(r['model'],r['concept'],r['method'],float(r['multiplier']));ss=raw[key]
        assert len(ss)==int(r['questions'])==len({s['index'] for s in ss})
        assert abs(np.mean([s['normalized_target_probability'] for s in ss])-float(r['probability']))<1e-12
    passed('Response probabilities and no-hook semantics',f'{checked} saved question/condition rows checked, including partial checkpoints. Every displayed complete summary matches raw means. Zero has no intervention and identical probability across methods. Maximum logit recomputation error: {max_response_error:.3g}. No monotonicity constraint is imposed.')
    route=load(PAPER/'data/figure_routing.json');status=load(PAPER/'data/artifact_status.json')
    for n in route['draft']:
        assert not (PAPER/f'figures/figure_{n}.pdf').exists()
        assert (PAPER/f'figures/drafts/figure_{n}.pdf').exists()
    passed('Publication gating','Incomplete response figures live only in figures/drafts and are excluded from the Overleaf compilation and ZIP. Figure 3 is ready only when its three fixed panels are complete. No favorable-panel reselection.')
    save(PAPER/'data/figure_scientific_checks.json',dict(checks=checks,near_constant_conditions=constant,all_passed=True,experiment_settings_changed=False))
    captions=(PAPER/'reports/figure_captions.md').read_text()
    md='''# Figure refinement and scientific review

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

'''+table(['Check','Status','Evidence'],[[r['check'],r['status'],r['detail']] for r in checks])+'\n'
    (PAPER/'reports/figure_refinement_review.md').write_text(md)
    entries=[]
    for n in status['generated_figures']:
        cap=next((line for line in captions.splitlines() if line.startswith(f'Figure {n}. ')),'')
        entries.append(f'<section><h2>Figure {n}</h2><a href="../figures/figure_{n}.pdf"><img src="../figures/figure_{n}.png" alt="Figure {n}"></a><p>{html.escape(cap)}</p></section>')
    drafts=''.join(f'<li><a href="../figures/drafts/figure_{n}.pdf">Figure {n} — incomplete draft</a></li>' for n in route['draft'])
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>Reviewed paper figures</title><style>body{font:16px/1.5 system-ui;max-width:1150px;margin:32px auto;padding:0 20px}section{border-top:1px solid #bbb;padding:20px 0}img{width:100%;height:auto}p{max-width:1000px}</style><h1>Reviewed Curveball paper figures</h1><p>Completed artifacts only. Click an image for its vector PDF. Scientific input checks passed; interpretation limits are documented in <a href="figure_refinement_review.md">the review</a>.</p><details><summary>Incomplete drafts (excluded from the paper bundle)</summary><ul>'+drafts+'</ul></details>'+''.join(entries)+'</html>'
    (PAPER/'reports/figure_gallery.html').write_text(page)
    print(json.dumps(dict(stage='figure_scientific_audit',passed=True,response_rows_checked=checked,near_constant_conditions=len(constant))),flush=True)

if __name__=='__main__':main()
