"""Report-only correction: frozen inference, selection and statistics unchanged.

Preserves original negative_search.py and configuration hashes.
The row mapping is copied before metadata overrides, avoiding duplicate keywords.
"""
from negative_search import *

def report():
    from markdown_it import MarkdownIt
    summary=[];perq=[];status=[]
    for model in MODELS:
        st=DEST/f'{model}_status.json';status.append(dict(model=model,**(load(st) if st.exists() else dict(status='not_started'))))
        selected_file=DEST/f'selected_{model}.json';chosen=load(selected_file)['selected'] if selected_file.exists() else {}
        for ci,c in enumerate(NAMES):
            for pi,part in enumerate(['fit','validation','benchmark']):
                path=DEST/'rows'/model/f'{c}_{part}.jsonl'
                if not path.exists():continue
                records={}
                for line in path.read_text().splitlines():
                    try:r=json.loads(line)['row']
                    except json.JSONDecodeError:break
                    records[r['index'],r['method']]=r
                ids=sorted(r['index'] for r in split_rows(c,part))
                if any((i,'No-hook') not in records for i in ids):continue
                groups=sorted({records[i,'No-hook']['group_id'] for i in ids});codes=np.array([groups.index(records[i,'No-hook']['group_id']) for i in ids]);cnt=np.bincount(codes)
                draws=np.random.default_rng(20261104+ci*3+pi).integers(0,len(groups),(10000,len(groups)));den=cnt[draws].sum(1)
                def interval(a):return np.quantile(np.bincount(codes,weights=a)[draws].sum(1)/den,[.025,.975]).tolist()
                p0=np.array([records[i,'No-hook']['normalized_target_probability'] for i in ids]);arrays={m:np.array([records[i,m]['normalized_target_probability'] for i in ids]) for m in {m for _,m in records} if m!='No-hook' and all((i,m) in records for i in ids)}
                for method,probs in sorted(arrays.items()):
                    delta=100*(probs-p0);lo,hi=interval(delta);disp=np.array([records[i,method]['displacement']['actual_displacement'][0] for i in ids])
                    s=dict(model=model,concept=c,behavior=NAMES[c],partition=part,method=method,questions=len(ids),groups=len(groups),nohook_probability=float(p0.mean()),negative_probability=float(probs.mean()),negative_delta_pp=float(delta.mean()),ci_low=lo,ci_high=hi,ci_crosses_zero=lo<=0<=hi,selected=method==f'lambda_{chosen.get(c)}',selected_lambda=chosen.get(c),displacement_mean=float(disp.mean()),displacement_rms=float(np.sqrt(np.mean(disp**2))),displacement_median=float(np.median(disp)),displacement_p95=float(np.quantile(disp,.95)),displacement_max=float(disp.max()))
                    for ref,label in [('Linear','vs_linear'),('lambda_0.0','vs_lambda0')]:
                        if ref in arrays:
                            d=100*(probs-arrays[ref]);l,h=interval(d);s.update({label+'_pp':float(d.mean()),label+'_ci_low':l,label+'_ci_high':h})
                    summary.append(s)
                    for j,i in enumerate(ids):perq.append(dict(records[i,method],model=model,concept=c,nohook_probability=float(p0[j]),negative_delta_pp=float(delta[j])))
    if summary:write_csv(DEST/'summary.csv',summary)
    if perq:write_csv(DEST/'per_question.csv',perq)
    save(DEST/'completion.json',dict(complete=all(s['status']=='complete' for s in status),models=status))
    body=[]
    for m in MODELS:
        for c in NAMES:
            frozen=DEST/f'selected_{m}.json';w=load(frozen)['selected'][c] if frozen.exists() else None
            line=[MODELS[m],NAMES[c],w if w is not None else 'pending']
            for part in ['fit','validation','benchmark']:
                for meth in ['Linear',f'lambda_{w}']:
                    r=next((r for r in summary if r['model']==m and r['concept']==c and r['partition']==part and r['method']==meth),None)
                    line.append(f"{r['negative_delta_pp']:+.3f}" if r else 'pending')
            body.append(line)
    headers=['Model','Behavior','Negative-selected lambda','Training Linear','Training selected','Validation Linear','Validation selected','Test Linear','Test selected']
    md='# Independent negative-steering lambda search\n\n'+table(['Model','Status'],[[s['model'],s['status']] for s in status])+'\n\n'+table(headers,body)+'\n\nValues are negative steering minus true no-hook target probability, in percentage points. More negative means stronger suppression of the official behavior target, not a negative kernel weight. Selection uses the full frozen validation grid only; pending cells are never filled from positive results. Training is in-sample. Test is an exploratory repeatedly used fixed benchmark. Pointwise 10,000 paired source-group bootstrap intervals are in summary.csv; validation selection bias and multiple comparisons limit inference. Natural displacement is not matched across kernels.\n'
    (PAPER/'reports/negative_lambda_search.md').write_text(md)
    (PAPER/'reports/negative_lambda_search.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><style>body{font:15px/1.6 system-ui;margin:30px}td,th{padding:6px;border:1px solid #ccc}table{border-collapse:collapse}</style>'+MarkdownIt('commonmark').enable('table').render(md)+'</html>')
    tex_table(PAPER/'tables/negative_lambda_selected.tex',headers,body,'Independent validation-selected negative steering; all-option target probability changes in percentage points. More negative indicates stronger suppression. Official Test is exploratory.','tab:negative_selected')


if __name__=="__main__":report()
