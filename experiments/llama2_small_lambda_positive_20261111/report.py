"""English positive-steering curves with source-group paired bootstrap intervals."""
import csv, html, math
import numpy as np
from study_common import *
from markdown_it import MarkdownIt

def write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
        for r in rows:wr.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    cfg=verify();summary=[];perq=[];diagnostics=[];completeness=[]
    methods=['Linear']+[f'lambda_{w}' for w in WEIGHTS]
    for key in cfg['models']:
        for ci,c in enumerate(NAMES):
            for partition,pi in [('fit',0),('validation',1),('benchmark',2)]:
                methods=methods_for(c,partition)[1:]
                path=OUT/'rows'/key/f'{c}_{partition}.jsonl';rows=[]
                if path.exists():
                    for line in path.read_text().splitlines():
                        try:obj=json.loads(line)
                        except json.JSONDecodeError:break
                        assert obj['config_sha256']==sha(OUT/'experiment_config.json');rows+=obj['rows']
                lookup={(r['index'],r['method']):r for r in rows};assert len(lookup)==len(rows)
                expected={r['index'] for r in SPLIT if r['concept']==c and r['partition']==partition}
                base={r['index']:r for r in rows if r['method']=='No-hook'}
                complete={m:{r['index'] for r in rows if r['method']==m}==expected for m in ['No-hook']+methods}
                completeness.append(dict(model=key,concept=c,partition=partition,records=len(rows),expected_records=(len(methods)+1)*len(expected),complete=all(complete.values())))
                for r in rows:
                    b=base.get(r['index'])
                    if b is None:continue
                    assert r['target']==b['target'] and r['candidate_token_ids']==b['candidate_token_ids'] and r['group_id']==b['group_id']
                    p=np.array(list(r['candidate_logits'].values()));assert np.isfinite(p).all()
                    d=r['displacement'];alts=[v for l,v in r['candidate_logits'].items() if l!=r['target']];mx=max(alts)
                    row=dict(model=key,concept=c,partition=partition,index=r['index'],group_id=r['group_id'],method=r['method'],alpha=r['alpha'],
                        target=r['target'],probability=r['normalized_target_probability'],nohook_probability=b['normalized_target_probability'],
                        positive_delta_pp=100*(r['normalized_target_probability']-b['normalized_target_probability']),
                        candidate_logits=r['candidate_logits'],candidate_token_ids=r['candidate_token_ids'],candidate_probability_mass=r['all_option_probability_mass'],
                        full_vocab_target_probability=r['full_vocab_target_probability'],logit_margin=r['candidate_logits'][r['target']]-(mx+math.log(sum(math.exp(v-mx) for v in alts))),
                        native_parameter=r['native_parameter'],displacement=d,execution_provenance=r['execution_provenance'])
                    perq.append(row)
                if not complete['No-hook']:continue
                ids=sorted(expected);groups=sorted({base[i]['group_id'] for i in ids});gmap={g:j for j,g in enumerate(groups)}
                codes=np.array([gmap[base[i]['group_id']] for i in ids]);counts=np.bincount(codes)
                draws=np.random.default_rng(20261104+ci*3+pi).integers(0,len(groups),(10000,len(groups)));den=counts[draws].sum(1)
                def interval(values):return [float(v) for v in np.quantile(np.bincount(codes,weights=values)[draws].sum(1)/den,[.025,.975])]
                p0=np.array([base[i]['normalized_target_probability'] for i in ids]);arrays={m:np.array([lookup[i,m]['normalized_target_probability'] for i in ids]) for m in methods if complete[m]}
                for m,a in arrays.items():
                    delta=100*(a-p0);lo,hi=interval(delta)
                    s=dict(model=key,behavior=NAMES[c],concept=c,partition=partition,method=m,questions=len(ids),groups=len(groups),nohook_probability=float(p0.mean()),
                        positive_probability=float(a.mean()),positive_delta_pp=float(delta.mean()),ci_low=lo,ci_high=hi,ci_crosses_zero=lo<=0<=hi)
                    for ref,label in [('Linear','vs_linear'),('lambda_0.0','vs_lambda0')]:
                        if ref in arrays:
                            diff=100*(a-arrays[ref]);l,h=interval(diff)
                            s.update({label+'_pp':float(diff.mean()),label+'_ci_low':l,label+'_ci_high':h,label+'_ci_crosses_zero':l<=0<=h})
                    summary.append(s)
                    obs=[lookup[i,m]['displacement'] for i in ids];norms=np.array([d['actual_displacement'][0] for d in obs])
                    d=dict(model=key,concept=c,partition=partition,method=m,mean=float(norms.mean()),rms=float(np.sqrt(np.mean(norms**2))),median=float(np.median(norms)),p95=float(np.quantile(norms,.95)),maximum=float(norms.max()),
                        mean_activation_norm=float(np.mean([d['activation_norm'][0] for d in obs])),mean_relative_displacement=float(np.mean([d['relative_displacement'][0] for d in obs])),mean_cosine_to_linear=float(np.mean([d['cosine_to_actual_signed_linear'][0] for d in obs])))
                    if m!='Linear':d.update(mean_reconstruction_l2=float(np.mean([d['reconstruction_l2'][0] for d in obs])),mean_relative_reconstruction_l2=float(np.mean([d['relative_reconstruction_l2'][0] for d in obs])))
                    diagnostics.append(d)
    write('summary.csv',summary);write('per_question.csv',perq);write('displacement_statistics.csv',diagnostics);write('completion.csv',completeness)
    frozen=load(OUT/'selected_configurations.json') if (OUT/'selected_configurations.json').exists() else None
    expected_cells=140+sum(len(methods_for(c,'benchmark'))-1 for c in NAMES) if frozen else None
    complete=bool(frozen and all(r['complete'] for r in completeness))
    save(OUT/'completion_status.json',dict(complete=complete,available_cells=len(summary),expected_cells=expected_cells,training_validation_expected_cells=140,question_condition_rows=len(perq),official_test_evaluated=any(s['partition']=='benchmark' for s in summary)))
    methods=['Linear']+[f'lambda_{w}' for w in WEIGHTS]
    tables=[]
    for key in cfg['models']:
        for partition in ['fit','validation']:
            lookup={(s['concept'],s['method']):s for s in summary if s['model']==key and s['partition']==partition}
            body=[[NAMES[c]]+[f"{lookup[c,m]['positive_delta_pp']:+.3f}" if (c,m) in lookup else 'pending' for m in methods] for c in NAMES]
            tables.append(f'## {key}: {"Training (in-sample)" if partition=="fit" else "Validation"}\n\n'+table(['Behavior']+methods,body))
    plots=[]
    if summary:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for key in cfg['models']:
            fig,axes=plt.subplots(2,4,figsize=(17,8),constrained_layout=True)
            for ax,c in zip(axes.flat,NAMES):
                for part,color in [('fit','#246391'),('validation','#d17d1c')]:
                    ss={s['method']:s for s in summary if s['model']==key and s['concept']==c and s['partition']==part}
                    ws=[w for w in WEIGHTS if f'lambda_{w}' in ss]
                    if ws:ax.plot(ws,[ss[f'lambda_{w}']['positive_delta_pp'] for w in ws],'-o',ms=3,color=color,label=part)
                    if 'Linear' in ss:ax.axhline(ss['Linear']['positive_delta_pp'],color=color,ls='--',lw=1,label=part+' Linear')
                ax.set_xscale('symlog',linthresh=.001);ax.set_title(NAMES[c]);ax.set_xlabel('Quadratic weight lambda');ax.set_ylabel('Positive delta P (pp)');ax.axhline(0,c='gray',lw=.5)
            axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels()
            if handles:axes.flat[-1].legend(handles,labels,loc='center')
            name=f'{key}_positive_curves.png';fig.savefig(OUT/name,dpi=140);plt.close(fig);plots.append(f'![{key} curves]({name})')
    md='\n\n'.join(['# Small quadratic weights: positive-only steering',f'Completed cells: {len(summary)}/{expected_cells if expected_cells is not None else "140 training/validation + pending selected test"}. Complete: {complete}. Official test is exploratory and follows validation selection freeze.',
        'Llama-2-7B-Chat block13 (hidden-state index14); fixed all-option normalized target-label probability, last-token intervention, natural Curveball magnitude. Main values are mean positive minus true no-hook probability in percentage points. Lambda1 is a reference; lambda0 is the linear-kernel KPCA limit, not CAA Linear.',
        'Only training groups fit representations. The Llama2 prompt is the previously validated legacy parenthesis format, [INST] question [/INST] (, rather than the assistant-answer prefill used by Qwen/Llama3B. The same fixed splits and scoring rules apply to every method. Llama strength is calibrated once with lambda1 on training-only prefix probes, following the preceding Llama3B calibration convention; that step is then fixed for all weights. Evaluation scope is last-token. Neither per-token matching nor outcome-based strength selection is applied.',
        'Intervals are 10,000 paired source-group bootstrap intervals; point estimates are row means. Training is in-sample. Validation is used for exploratory lambda comparisons and is not an independent confirmation after selection. No uncorrected multiple-comparison superiority claims. All curves, including weak outcomes, are retained.',
        *tables,*plots,*selected_tables(summary,frozen),
        '[Detailed probabilities and paired intervals](summary.csv) | [Every question](per_question.csv) | [Displacement and reconstruction](displacement_statistics.csv) | [Frozen configuration](experiment_config.json)',
        'Relative reconstruction error is ||inverse(project(h))+center-h||2 / ||h||2, measured before steering; this is not the residual-preserved intervention error. Per-question records retain actual update norms, signed Linear cosine, logits, margins and candidate mass. No coherence claims are inferred from MCQ scores.',
        'Historical results and sources are unchanged. No GitHub push. This is an original-style weighted-kernel ablation under CAA-derived evaluation, not a claim of exact reproduction of all NeurIPS manuscript settings.'])+'\n'
    (OUT/'report.md').write_text(md);(OUT/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><style>body{font:16px/1.6 system-ui;margin:32px}table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #ccd;padding:7px}img{max-width:100%}</style>'+MarkdownIt('commonmark').enable('table').render(md)+'</html>')
    print(json.dumps(dict(summary_cells=len(summary),records=len(perq),complete=complete)),flush=True)


def selected_tables(summary,frozen):
    if frozen is None:return ['## Validation-selected configurations and official test\n\nPending complete validation; no test inference is permitted before the selection freeze.']
    lookup={(s['concept'],s['partition'],s['method']):s for s in summary}
    headers=['Behavior','Selected lambda']+[f'{part} {m}' for part in ['Training','Validation','Test'] for m in ['Linear','lambda0','selected','lambda1']]
    body=[];selected_rows=[]
    for c in NAMES:
        w=frozen['selected'][c];row=[NAMES[c],w]
        for part in ['fit','validation','benchmark']:
            for m in ['Linear','lambda_0.0',f'lambda_{w}','lambda_1.0']:
                r=lookup.get((c,part,m));row.append(f"{r['positive_delta_pp']:+.3f}" if r else 'pending')
            r=lookup.get((c,part,f'lambda_{w}'))
            if r:selected_rows.append(dict(r,selected_weight=w))
        body.append(row)
    write('selected_summary.csv',selected_rows)
    latex=['\\begin{tabular}{ll'+'r'*12+'}', ' & '.join(headers)+' \\\\', '\\hline']
    latex+=[' & '.join(map(str,r))+' \\\\' for r in body];latex+=['\\end{tabular}']
    (OUT/'main_table.tex').write_text('\n'.join(latex)+'\n')
    return ['## Frozen validation selection: Training / Validation / Official Test\n\n'+table(headers,body),
        'Official test values are exploratory fixed-benchmark results. The benchmark has been used in prior diagnoses. Validation maxima may be selection-biased. Paired intervals are pointwise and do not establish multiplicity-corrected superiority. The same chosen lambda is shown across all three partitions; test performance never changes the choice.']

if __name__=='__main__':main()
