"""Auditable English report, with diagnostic controls separated from test results."""
import csv,math
import numpy as np
from support import *
from analyze import write,stats,bootstrap
from markdown_it import MarkdownIt

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def readcsv(name):return list(csv.DictReader((OUT/name).open()))
def fmt(v):return f'{float(v):+.3f}'

def main():
    cfg=verify();tests=[];perq=[];geometry=[];factor=[];replays=[];progress=[];diagnostic_rows=[]
    for key in ['qwen3','llama3b']:
        loc=OUT/f'location_{key}.json'
        if not loc.exists():progress.append([key,'pending']);continue
        run=Path(load(loc)['path']);manifest=load(run/'manifest.json');progress.append([key,manifest['status']])
        for p in run.glob('geometry_*.json'):geometry+=load(p)
        for p in run.glob('replay_parity_*.json'):replays += [dict(model=key,**r) for r in load(p)]
        for p in run.glob('diagnostic_rows_*.json'):diagnostic_rows += [dict(r,model=key) for r in load(p)]
        if key=='qwen3' and (run/'diagnostic_rows_hallucination.json').exists():
            rr=load(run/'diagnostic_rows_hallucination.json');base={r['index']:r for r in rr if r['method']=='No-hook'};ids=sorted(base);by={(r['index'],r['method']):r for r in rr};boot,ng=bootstrap(base,ids,20261104+2*3+1)
            arr=lambda m:np.array([by[i,m]['normalized_target_probability'] for i in ids])
            p00=arr('lambda_0.0');p11=arr('lambda_0.01');p01=arr('direction0_magnitude001');p10=arr('direction001_magnitude0')
            effects={'native001_minus_native0':p11-p00,'direction_at_magnitude0':p10-p00,'direction_at_magnitude001':p11-p01,'magnitude_at_direction0':p01-p00,'magnitude_at_direction001':p11-p10,'interaction':p11-p10-p01+p00,'symmetric_direction_contribution':((p10-p00)+(p11-p01))/2,'symmetric_magnitude_contribution':((p01-p00)+(p11-p10))/2}
            for label,delta in effects.items():
                lo,hi=np.quantile(boot(100*delta),[.025,.975]);factor.append(dict(contrast=label,mean_pp=float(100*delta.mean()),ci_low=float(lo),ci_high=float(hi),questions=len(ids),groups=ng,diagnostic_only=True))
        for ci,c in enumerate(NAMES):
            p=run/f'test_rows_{c}.jsonl'
            if not p.exists():continue
            rows=[]
            for line in p.read_text().splitlines():
                try:d=load_line=json.loads(line)
                except json.JSONDecodeError:break
                assert d['config_sha256']==sha(OUT/'frozen_config.json');rows+=d['rows']
            by={(r['index'],r['method']):r for r in rows};assert len(by)==len(rows)
            base={r['index']:r for r in rows if r['method']=='No-hook'};ids=sorted(base)
            chosen=next(s['selected_weight'] for s in cfg['selected'] if s['model']==key and s['concept']==c)
            methods=['Linear']+[f'lambda_{w}' for w in sorted({0.,1.,chosen})]
            if len(ids)!=50 or any(sum(r['method']==m for r in rows)!=50 for m in methods):continue
            boot,ng=bootstrap(base,ids,20261104+ci*3+2);p0=np.array([base[i]['normalized_target_probability'] for i in ids]);arrays={m:np.array([by[i,m]['normalized_target_probability'] for i in ids]) for m in methods}
            for r in rows:
                t=r['index'];v=np.array([r['candidate_logits'][l] for l in r['option_labels']]);v=np.exp(v-v.max());v/=v.sum()
                assert abs(v[r['option_labels'].index(r['target'])]-r['normalized_target_probability'])<1e-12
                assert r['target']==base[t]['target'] and r['candidate_token_ids']==base[t]['candidate_token_ids']
                perq.append(dict(model=key,concept=c,index=t,group_id=r['group_id'],method=r['method'],target=r['target'],probability=r['normalized_target_probability'],nohook_probability=base[t]['normalized_target_probability'],delta_pp=100*(r['normalized_target_probability']-base[t]['normalized_target_probability']),candidate_logits=r['candidate_logits'],candidate_probability_mass=r['all_option_probability_mass'],displacement=r['displacement'],exploratory=True))
            for m,p1 in arrays.items():
                delta=100*(p1-p0);lo,hi=np.quantile(boot(delta),[.025,.975]);s=dict(model=key,concept=c,behavior=NAMES[c],method=m,selected=m==f'lambda_{chosen}',questions=50,groups=ng,nohook_probability=float(p0.mean()),positive_probability=float(p1.mean()),positive_delta_pp=float(delta.mean()),ci_low=float(lo),ci_high=float(hi),exploratory=True)
                for ref,label in [('Linear','vs_linear'),('lambda_0.0','vs_lambda0')]:
                    gap=100*(p1-arrays[ref]);l,h=np.quantile(boot(gap),[.025,.975]);s.update({label+'_pp':float(gap.mean()),label+'_ci_low':float(l),label+'_ci_high':float(h)})
                tests.append(s)
    write('test_summary.csv',tests);write('test_per_question.csv',perq);write('actual_update_geometry.csv',geometry);write('qwen_factorial_contrasts.csv',factor);write('gpu_replay_parity.csv',replays)
    # Retain the full diagnostic probability records, separately from benchmark rows.
    save(OUT/'diagnostic_probabilities.json',diagnostic_rows)
    derived=[]
    for key in ['qwen3','llama3b']:
        for c in NAMES:
            for m in sorted({r['method'] for r in geometry if r['model']==key and r['concept']==c}):
                gg=[r for r in geometry if r['model']==key and r['concept']==c and r['method']==m]
                derived.append(dict(model=key,concept=c,method=m,questions=len(gg),**stats([r['displacement'] for r in gg]),mean_cosine_to_lambda0=float(np.mean([r['cosine_to_lambda0'] for r in gg])),minimum_cosine_to_lambda0=float(np.min([r['cosine_to_lambda0'] for r in gg])),mean_relative_update_difference_to_lambda0=float(np.mean([r['relative_update_difference_to_lambda0'] for r in gg])),maximum_relative_update_difference_to_lambda0=float(np.max([r['relative_update_difference_to_lambda0'] for r in gg]))))
    write('geometry_summary.csv',derived)
    cached=readcsv('qwen_hallucination_displacement.csv');intervals=readcsv('paired_validation_intervals.csv');kernels=readcsv('kernel_component_diagnostics.csv');llama=readcsv('llama_sanity_cached.csv')
    hall=next(r for r in intervals if r['model']=='qwen3' and r['concept']=='hallucination' and float(r['weight'])==.01)
    oldsum=list(csv.DictReader((OLD/'summary.csv').open()));comparison=[]
    for s in cfg['selected']:
        k,c,w=s['model'],s['concept'],s['selected_weight'];row=[k,NAMES[c],w]
        for split in ['fit','validation']:
            for m in ['Linear',f'lambda_{w}']:
                r=next(r for r in oldsum if r['model']==k and r['concept']==c and r['partition']==split and r['method']==m);row.append(fmt(r['positive_delta_pp']))
        for m in ['Linear',f'lambda_{w}']:
            r=next((r for r in tests if r['model']==k and r['concept']==c and r['method']==m),None);row.append('pending' if r is None else fmt(r['positive_delta_pp']))
        comparison.append(row)
    plot=[]
    if geometry:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        gg=[r for r in geometry if r['model']=='qwen3' and r['method'].startswith('lambda_')]
        if gg:
            methods=[f'lambda_{w}' for w in [0.,.005,.01,.05,1.]];fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
            for ax,field,label in zip(axes,['displacement','norm_ratio_to_lambda0','cosine_to_lambda0'],['Actual displacement L2','Per-question norm / lambda0 norm','Update cosine to lambda0']):
                ax.boxplot([[r[field] for r in gg if r['method']==m] for m in methods],tick_labels=['0','.005','.01','.05','1'],showfliers=True);ax.set_xlabel('Quadratic weight');ax.set_ylabel(label)
            fig.savefig(OUT/'qwen_geometry.png',dpi=160);plt.close(fig);plot=['![Qwen Hallucination geometry](qwen_geometry.png)']
    complete=sum(r['selected'] for r in tests)==14
    save(OUT/'completion_status.json',dict(complete=complete,selected_test_cells=sum(r['selected'] for r in tests),expected_selected_test_cells=14,official_test_question_condition_rows=len(perq),gpu_states=progress))
    content=['# Small-lambda scientific diagnostics and exploratory official test',table(['Model','GPU status'],progress),
        '## Frozen selection and interpretation',
        'Candidate weights are selected exclusively from the completed validation search grid0..0.5; lambda1 remains the predeclared reference. Selection occurred before new diagnostic or test inference. All350 official questions per model and official labels are retained. No fitting, strength change, scorer adaptation or test-driven parameter update is performed. Selected validation maxima are optimistic; the official test has previously been inspected and is exploratory fixed-benchmark evaluation, not a fresh independent held-out confirmation.',
        '## Qwen Hallucination: natural displacement',table(['Weight','Positive delta (pp)','RMS','Median','P95','Maximum','Cosine to Linear','Relative reconstruction'],[[r['weight'],fmt(r['positive_delta_pp']),f"{float(r['rms']):.3f}",f"{float(r['median']):.3f}",f"{float(r['p95']):.3f}",f"{float(r['maximum']):.3f}",f"{float(r['mean_cosine_to_linear']):.6f}",f"{float(r['mean_relative_reconstruction']):.6f}"] for r in cached]),*plot,
        'Similar RMS alone does not establish identical per-question magnitudes or separate direction from magnitude. The replayed native updates and crossed direction/magnitude controls below operate on identical unsteered activations. They are diagnostic interventions only and never substitute for the selected original-style method on test.',
        '## Paired statistical evidence',f"Lambda0.01 minus lambda0: {fmt(hall['delta_vs_lambda0_pp'])} pp; paired pointwise95% CI [{fmt(hall['paired_ci_low'])}, {fmt(hall['paired_ci_high'])}]. The original row mean and10,000 source-group bootstrap resamples are retained (195 questions,184 groups).",
        f"Exploratory simultaneous bands: within-behavior8-comparison max-statistic band [{fmt(hall['within_concept_maxT_low'])}, {fmt(hall['within_concept_maxT_high'])}]; across112 comparisons [{fmt(hall['global_112_comparison_maxT_low'])}, {fmt(hall['global_112_comparison_maxT_high'])}]. These are centered standardized bootstrap approximations computed post hoc. They address this specified comparison family, not all prior research decisions, and do not eliminate selection optimism or data reuse.",
        '## Direction/magnitude counterfactuals',table(['Contrast','Mean (pp)','Paired95% CI'],[[r['contrast'],fmt(r['mean_pp']),f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]"] for r in factor]) if factor else 'GPU counterfactual results pending.',
        'Direction-at-magnitude0 holds each question\'s lambda0 norm fixed. Magnitude-at-direction0 holds its lambda0 direction fixed. The other conditional effects reverse the path; the interaction records nonadditivity. Symmetric contributions average the two paths and sum to the native effect difference; they are diagnostic decompositions, not new steering methods.',
        '## Llama sanity',table(['Behavior','Lambda mean range (pp)','Max per-question |lambda1-lambda0| (pp)','All hooks triggered'],[[NAMES[r['concept']],f"{float(r['lambda_mean_range_pp']):.6f}",f"{float(r['lambda1_vs0_max_abs_per_question_pp']):.6f}",r['all_methods_hook_triggered']] for r in llama]),
        table(['Model','Behavior','Centered forward quadratic/linear norm','Inverse quadratic/linear norm'],[[r['model'],NAMES[r['concept']],f"{float(r['centered_forward_quadratic_to_linear_frobenius']):.6g}",f"{float(r['inverse_quadratic_to_linear_frobenius']):.6g}"] for r in kernels]),
        'The kernel ratios measure lambda1 contributions on the actual centered fitting activations and inverse latent training coordinates. Their scale differs across models despite the identical nominal coefficient0.01. These are diagnostic measurements, not grounds to rescale frozen kernels. Actual per-question update-vector differences and replay errors are in geometry_summary.csv and gpu_replay_parity.csv. Small rounded probability differences are not automatically zero updates or an ignored lambda.',
        '## Training / validation / exploratory official test',table(['Model','Behavior','Selected weight','Train Linear','Train selected','Val Linear','Val selected','Test Linear','Test selected'],comparison),
        '## Limits and artifacts',
        'All primary updates retain natural adaptive magnitude. Reconstruction error means ||inverse(project(h))+center-h||2 / ||h||2; residual preservation still gives the original activation at zero steering. Large reconstruction error alone does not establish a causal behavioral defect. No generation coherence conclusions follow from this MCQ experiment.',
        '[Frozen configuration](frozen_config.json) | [Selected configurations](selected_configurations.csv) | [Test probabilities and intervals](test_summary.csv) | [Test per-question records](test_per_question.csv) | [All paired validation intervals](paired_validation_intervals.csv) | [Actual update geometry](actual_update_geometry.csv) | [Factorial contrasts](qwen_factorial_contrasts.csv)',
        'Historical results are unchanged. No GitHub push. No post-test tuning.']
    md='\n\n'.join(content)+'\n';(OUT/'report.md').write_text(md);(OUT/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><style>body{font:16px/1.6 system-ui;margin:32px}table{border-collapse:collapse;display:block;overflow:auto}th,td{border:1px solid #ccd;padding:7px}img{max-width:100%}</style>'+MarkdownIt('commonmark').enable('table').render(md)+'</html>')
    verify();print(json.dumps(dict(complete=complete,test_summary_rows=len(tests),test_records=len(perq),factorial_contrasts=factor)),flush=True)

if __name__=='__main__':main()
