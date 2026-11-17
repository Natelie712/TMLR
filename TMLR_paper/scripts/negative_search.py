"""Independent validation-only negative-lambda selection; immutable positive study."""
import argparse, fcntl, os, time, traceback
from artifact_common import *
import response_gpu as reference
import torch, joblib

DEST=PAPER/'data/negative_lambda_search'
PLAN=DEST/'experiment_config.json'

def prepare():
    reference.verify_frozen()
    sources=[Path(__file__),PAPER/'scripts/response_gpu.py',PAPER/'scripts/artifact_common.py',PAPER/'data/revision_config.json']
    for m in MODELS:
        sources += [base(m)/'fits'/m/'linear_vectors.pt',base(m)/'fits'/m/'positive_steps.json']
        sources += [fit_path(m,c,w) for c in NAMES for w in WEIGHTS]
    cfg=dict(models=MODELS,weights=WEIGHTS,behaviors=NAMES,multiplier=-1,
        strength='Negative of frozen positive lambda1 training-calibrated native step; shared across weights; no displacement matching.',
        selection='Minimum validation row-mean target probability change; exact ties choose smaller numeric lambda. No official-test selection.',
        aggregation='Row-mean point estimates; paired source-group bootstrap, matching the positive study.',
        validation_methods=['Linear']+[f'lambda_{w}' for w in WEIGHTS],
        followup='Freeze all seven choices per model, then evaluate fit and official test for Linear, lambda0, selected negative lambda, lambda1.',
        bootstrap_resamples=10000,seed=20261104,exploratory_fixed_benchmark=True,
        frozen_inputs={str(p.relative_to(ROOT)):sha(p) for p in sources})
    if PLAN.exists():assert load(PLAN)==cfg,'Negative plan already frozen; do not overwrite'
    else:save(PLAN,cfg)
    md='''# Negative steering: independent lambda selection

This study selects a negative-steering kernel weight independently for each of three models and seven CAA behaviors. Positive selections and historical results remain unchanged.

The fixed grid is {0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1}. Models, blocks, checkpoints, prompt variants, all-option scorer, last-token natural residual update, fits and Linear vectors are reused unchanged. Linear uses alpha=-1. Curveball uses the negative of the frozen positive native step calibrated with lambda1 on training-only probes. No negative-strength recalibration or strength search is performed. Equal multipliers do not imply equal actual displacement.

Select the smallest validation row-mean Delta P(target); exact ties choose the smaller numerical lambda. A positive minimum is reported as failure to suppress, not an improvement. Selection uses neither the official test nor training evaluation. Freeze all seven choices for each model before that model's official-test evaluation. Fit (in-sample) and official test then evaluate Linear, lambda0, selected negative lambda and lambda1. Row means and paired source-group bootstrap intervals follow the positive study. All 350 official questions and original labels remain unchanged. Test results are exploratory fixed-benchmark results, and validation maxima/minima are subject to selection bias.

Checkpoint each question/condition. Reuse response-curve endpoints only after checking frozen revision provenance, exact signed native parameter, target, candidate token IDs, group and a fresh positive/no-hook replay. Historical negative calibration results are not substituted. Verify every kernel's zero-step identity, projection/inverse parity and negative hook position before scoring. Store logits, candidate probability mass, native step, actual displacement and reconstruction diagnostics. Numerical assertions stop execution and are not automatically retried.

Outputs: `data/negative_lambda_search/`, `reports/negative_lambda_search.html`, and `tables/negative_lambda_selected.tex`. No GitHub push.
'''
    (PAPER/'reports/negative_lambda_protocol.md').write_text(md)
    print('Negative search plan frozen.',flush=True)

def verify():
    cfg=load(PLAN)
    for p,h in cfg['frozen_inputs'].items():assert sha(ROOT/p)==h,p
    reference.verify_frozen()
    return cfg

def checkpoint(model,c,part):
    path=DEST/'rows'/model/f'{c}_{part}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    found={};good=0
    if path.exists():
        with path.open('rb') as f:
            for line in f:
                try:o=json.loads(line)
                except json.JSONDecodeError:break
                assert o['config_sha256']==sha(PLAN)
                r=o['row'];key=(r['index'],r['method']);assert key not in found
                found[key]=r;good+=len(line)
        if good<path.stat().st_size:
            with path.open('r+b') as f:f.truncate(good)
    return path,found

def choose(scores):
    assert set(scores)==set(WEIGHTS) and np.isfinite(list(scores.values())).all()
    return min(WEIGHTS,key=lambda w:(scores[w],w))

def freeze_selection(model):
    selected={};evidence={};hashes={}
    for c in NAMES:
        path,rs=checkpoint(model,c,'validation');ids={r['index'] for r in split_rows(c,'validation')}
        means={}
        for w in WEIGHTS:
            method=f'lambda_{w}';assert {i for i,m in rs if m==method}==ids
            assert {i for i,m in rs if m=='No-hook'}==ids
            means[w]=float(np.mean([100*(rs[i,method]['normalized_target_probability']-rs[i,'No-hook']['normalized_target_probability']) for i in sorted(ids)]))
        selected[c]=choose(means);evidence[c]=means;hashes[str(path.relative_to(PAPER))]=sha(path)
    result=dict(selected=selected,validation_delta_pp=evidence,validation_hashes=hashes,config_sha256=sha(PLAN))
    path=DEST/f'selected_{model}.json'
    if path.exists():assert load(path)==json.loads(json.dumps(result))
    else:save(path,result)
    return selected

class Run(reference.Run):
    def __init__(self,model):
        reference.OUT=DEST
        super().__init__(model)
        self.manifest.update(protocol='independent negative lambda selection',validation_only=False,grid=WEIGHTS,negative_config_sha256=sha(PLAN))
    def tasks(self,c,part):
        from runtime_b import data,task
        split='test' if part=='benchmark' else 'generate';src=data(c,split)
        return [dict(task(self.tok,self.key,c,r['index'],src[r['index']],self.variant,split),group_id=r['group_id'],partition=part) for r in split_rows(c,part)]
    def restore(self,c):
        CB=fit_class();self.cbs[c]={w:CB.restore(joblib.load(fit_path(self.key,c,w)),'cuda') for w in WEIGHTS}
        step=float(load(base(self.key)/'fits'/self.key/'positive_steps.json')['steps'][c]['1'])
        assert step>0 and np.isfinite(step)
        self.steps[c]={'-1':-step,'1':step}
    def checks(self,c):
        cache={(r['index'],r['method']):r for r in raw_rows(self.key,c,'validation')};checks=[]
        for t in sorted(self.tasks(c,'validation'),key=lambda x:x['index'])[:2]:
            self.reference={};b=self.condition([t],'No-hook',0)[0]
            for method in ['No-hook','Linear',f'lambda_{selected(self.key)[c]}']:
                r=b if method=='No-hook' else self.condition([t],method,1)[0];old=cache[t['index'],method]
                assert r['target']==old['target'] and r['candidate_token_ids']==old['candidate_token_ids']
                error=abs(r['normalized_target_probability']-old['normalized_target_probability']);assert error<1e-4,error
                checks.append(dict(index=t['index'],method=method,replay_error=error))
            for w,cb in self.cbs[c].items():
                zrow=self.condition([t],f'lambda_{w}',0,step_override=0.)[0]
                assert zrow['normalized_target_probability']==b['normalized_target_probability']
                h=self.reference[t['index']].unsqueeze(0).cuda();z=cb.project(h)
                np.testing.assert_allclose(z.cpu().numpy(),cb.fit.transform(h.cpu().numpy()-cb.mean),rtol=1e-6,atol=1e-6)
                np.testing.assert_allclose(cb.inverse(z).cpu().numpy(),cb.fit.inverse_transform(z.cpu().numpy()),rtol=1e-6,atol=1e-5)
                neg=self.condition([t],f'lambda_{w}',-1)[0]
                assert neg['native_parameter']==self.steps[c]['-1'] and neg['intervention_triggered']
                assert neg['displacement']['positions']==[len(t['ids'])-1]
        save(self.out/f'checks_{c}.json',dict(passed=True,rows=checks,all_kernel_zero_identity_and_projection_inverse_checked=True))
    def evaluate_partition(self,c,part,methods):
        tasks=self.tasks(c,part);save(self.out/f'prompts_{c}_{part}.json',tasks)
        path,done=checkpoint(self.key,c,part);reusable={}
        if part=='validation':
            p=PAPER/'data/response_curves'/self.key/f'{c}.jsonl'
            if p.exists():
                for line in p.read_text().splitlines():
                    try:o=json.loads(line)
                    except json.JSONDecodeError:break
                    assert o['revision_sha256']==sha(PAPER/'data/revision_config.json')
                    for r in o['rows']:
                        if r['multiplier'] in [-1.,0.]:reusable[r['index'],'No-hook' if r['multiplier']==0 else r['method']]=r
        with path.open('a') as f:
            for t in tasks:
                self.reference={}
                for method in ['No-hook']+methods:
                    key=(t['index'],method)
                    if key in done:continue
                    sign=0 if method=='No-hook' else -1
                    expected=0 if not sign else -1 if method=='Linear' else self.steps[c]['-1']
                    old=reusable.get(key)
                    if old is not None:
                        assert old['native_parameter']==expected and old['target']==t['target'] and old['group_id']==t['group_id']
                        assert old['candidate_token_ids']==t['decision']['option_ids']
                        r=dict(old,provenance='reused verified symmetric negative response endpoint')
                    else:
                        r=self.condition([t],method,sign)[0];r['provenance']='new negative-search GPU inference'
                    r.update(method=method,alpha=sign,multiplier=sign,partition=part,negative_config_sha256=sha(PLAN))
                    assert r['native_parameter']==expected
                    f.write(json.dumps(dict(config_sha256=sha(PLAN),row=r))+'\n');f.flush();os.fsync(f.fileno());done[key]=r
                if t['index']%25==0:print(json.dumps(dict(model=self.key,concept=c,partition=part,rows=len(done),expected=len(tasks)*(len(methods)+1))),flush=True)
        assert len(done)==len(tasks)*(len(methods)+1)
        self.completed.append(dict(concept=c,partition=part));self.persist('running')
    def run(self):
        verify();self.load_model()
        for c in NAMES:
            self.restore(c);self.checks(c)
            self.evaluate_partition(c,'validation',['Linear']+[f'lambda_{w}' for w in WEIGHTS])
            self.cbs[c]={};torch.cuda.empty_cache()
        choice=freeze_selection(self.key);self.persist('selection_frozen')
        report()
        for c in NAMES:
            self.restore(c)
            for part in ['fit','benchmark']:
                self.evaluate_partition(c,part,['Linear']+[f'lambda_{w}' for w in sorted({0.,1.,choice[c]})])
            self.cbs[c]={};torch.cuda.empty_cache()
        verify();self.persist('complete');report()

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
                    for j,i in enumerate(ids):perq.append(dict(model=model,concept=c,**records[i,method],nohook_probability=float(p0[j]),negative_delta_pp=float(delta[j])))
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

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','report']);p.add_argument('--model',choices=list(MODELS));a=p.parse_args()
    if a.action=='prepare':prepare();report();return
    if a.action=='report':report();return
    assert a.model;verify();(DEST/'execution').mkdir(parents=True,exist_ok=True)
    lock=(DEST/'execution'/f'{a.model}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=DEST/f'{a.model}_status.json'
    if status.exists() and load(status)['status']=='complete':report();return
    torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ex=Run(a.model)
    try:
        with torch.inference_mode():ex.run()
    except Exception:
        ex.manifest['error']=traceback.format_exc();ex.persist('failed');raise

if __name__=='__main__':main()
