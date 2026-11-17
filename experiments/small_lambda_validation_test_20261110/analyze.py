"""Cached diagnostics, paired intervals, and outcome-independent test freeze."""
import csv,datetime
import numpy as np, torch,joblib
from support import *

def write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})

def stats(a):
    a=np.asarray(a);return dict(mean=float(a.mean()),rms=float(np.sqrt(np.mean(a*a))),median=float(np.median(a)),p05=float(np.quantile(a,.05)),p95=float(np.quantile(a,.95)),maximum=float(a.max()),minimum=float(a.min()))

def readrows(key,c,partition):
    rows=[]
    for line in (OLD/'rows'/key/f'{c}_{partition}.jsonl').read_text().splitlines():rows+=json.loads(line)['rows']
    return rows

def bootstrap(base,ids,seed):
    groups=sorted({base[i]['group_id'] for i in ids});mp={g:j for j,g in enumerate(groups)}
    codes=np.array([mp[base[i]['group_id']] for i in ids]);counts=np.bincount(codes)
    draws=np.random.default_rng(seed).integers(0,len(groups),(10000,len(groups)));den=counts[draws].sum(1)
    def boot(a):return np.bincount(codes,weights=a)[draws].sum(1)/den
    return boot,len(groups)

def main():
    torch.set_num_threads(2);prev.verify()
    summary=list(csv.DictReader((OLD/'summary.csv').open()));selection=[]
    # lambda1 was explicitly a reference in the preceding study; search candidates <=0.5.
    for key in ['qwen3','llama3b']:
        for c in NAMES:
            candidates=[r for r in summary if r['model']==key and r['concept']==c and r['partition']=='validation' and r['method'].startswith('lambda_') and float(r['method'].split('_')[1])<=.5]
            best=max(float(r['positive_delta_pp']) for r in candidates)
            chosen=min((r for r in candidates if best-float(r['positive_delta_pp'])<=1e-6),key=lambda r:float(r['method'].split('_')[1]))
            selection.append(dict(model=key,concept=c,selected_weight=float(chosen['method'].split('_')[1]),validation_delta_pp=float(chosen['positive_delta_pp']),tie_tolerance_pp=1e-6))
    protected={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments/small_lambda_positive_20261110').glob('*.py')}
    for p in list(HERE.glob('*.py'))+[OUT/'protocol.md',OLD/'summary.csv',OLD/'displacement_statistics.csv',OLD/'experiment_config.json']:
        protected[str(p.relative_to(ROOT))]=sha(p)
    for key in ['qwen3','llama3b']:
        for p in (OLD/'fits'/key).glob('*'):
            if p.is_file():protected[str(p.relative_to(ROOT))]=sha(p)
        for c in NAMES:
            p=OLD/'rows'/key/f'{c}_validation.jsonl';protected[str(p.relative_to(ROOT))]=sha(p)
    cfg=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected=selection,
        selection_rule='Maximize prior validation mean delta among prespecified search weights 0..0.5; ties within1e-6 pp select smaller lambda. Lambda1 remains the predeclared reference, not a new search candidate. This resolves the prior generic max rule consistently with its explicit lambda1-reference role.',
        official_test_methods=['No-hook','Linear','lambda_0.0','selected_weight','lambda_1.0'],deduplicate_identical_methods=True,
        models=prev.config()['models'],steps={k:load(OLD/'fits'/k/'positive_steps.json')['steps'] for k in ['qwen3','llama3b']},
        fitting=False,steering_scope='last real token',scorer='unchanged all-option normalized target-label probability',exploratory=True,
        qwen_diagnostic=dict(concept='hallucination',partition='validation',weights=[0.,.005,.01,.05,1.],crossed_updates=['direction0_magnitude001','direction001_magnitude0'],scope='diagnostic only; no test configuration selection from these outcomes'),
        llama_sanity=dict(partition='validation',groups_per_concept=16,selection='first16 sorted unique source-group IDs',weights=[0.,.005,.01,.05,1.]),
        protected_inputs=protected)
    fp=OUT/'frozen_config.json'
    if fp.exists():
        old=load(fp);cfg['created_utc']=old['created_utc'];assert cfg==old
    else:save(fp,cfg)
    write('selected_configurations.csv',selection)
    distributions=[];pairwise=[];all_boot=[];families=[];llama=[]
    for key in ['qwen3','llama3b']:
        for ci,c in enumerate(NAMES):
            rows=readrows(key,c,'validation');by={(r['index'],r['method']):r for r in rows};base={r['index']:r for r in rows if r['method']=='No-hook'};ids=sorted(base)
            boot,ng=bootstrap(base,ids,20261104+ci*3+1)
            p0=np.array([by[i,'lambda_0.0']['normalized_target_probability'] for i in ids]);means=[];local=[]
            for w in prev.WEIGHTS:
                method=f'lambda_{w}';p=np.array([by[i,method]['normalized_target_probability'] for i in ids]);means.append(float(p.mean()))
                if w>0:
                    delta=100*(p-p0);rep=boot(delta);se=float(rep.std(ddof=1));lo,hi=np.quantile(rep,[.025,.975]);j=len(pairwise)
                    pairwise.append(dict(model=key,concept=c,weight=w,questions=len(ids),groups=ng,delta_vs_lambda0_pp=float(delta.mean()),paired_ci_low=float(lo),paired_ci_high=float(hi),bootstrap_se=se))
                    standardized=(rep-delta.mean())/max(se,1e-12);all_boot.append(standardized);local.append(standardized);families.append((key,c))
                if key=='qwen3' and c=='hallucination' and w in [0.,.005,.01,.05,1.]:
                    obs=[by[i,method]['displacement'] for i in ids];norm=np.array([d['actual_displacement'][0] for d in obs]);norm0=np.array([by[i,'lambda_0.0']['displacement']['actual_displacement'][0] for i in ids])
                    rec=dict(model=key,concept=c,weight=w,**stats(norm),mean_cosine_to_linear=float(np.mean([d['cosine_to_actual_signed_linear'][0] for d in obs])),mean_relative_reconstruction=float(np.mean([d['relative_reconstruction_l2'][0] for d in obs])),positive_delta_pp=100*float(np.mean([by[i,method]['normalized_target_probability']-base[i]['normalized_target_probability'] for i in ids])),norm_ratio_median=float(np.median(norm/norm0)),norm_ratio_p05=float(np.quantile(norm/norm0,.05)),norm_ratio_p95=float(np.quantile(norm/norm0,.95)))
                    distributions.append(rec)
            critical=float(np.quantile(np.max(np.abs(local),axis=0),.95))
            for p in pairwise:
                if p['model']==key and p['concept']==c:
                    p['within_concept_maxT_low']=p['delta_vs_lambda0_pp']-critical*p['bootstrap_se'];p['within_concept_maxT_high']=p['delta_vs_lambda0_pp']+critical*p['bootstrap_se']
            if key=='llama3b':
                one=np.array([by[i,'lambda_1.0']['normalized_target_probability'] for i in ids]);gap=one-p0
                llama.append(dict(concept=c,lambda_mean_range_pp=100*(max(means)-min(means)),lambda1_vs0_max_abs_per_question_pp=100*float(np.max(np.abs(gap))),lambda1_vs0_mean_abs_per_question_pp=100*float(np.mean(abs(gap))),all_methods_hook_triggered=all(r['intervention_triggered'] for r in rows if r['method']!='No-hook'),nohook_mean=float(np.mean([base[i]['normalized_target_probability'] for i in ids])),nohook_floor_fraction=float(np.mean([base[i]['normalized_target_probability']<.01 for i in ids])),nohook_ceiling_fraction=float(np.mean([base[i]['normalized_target_probability']>.99 for i in ids]))))
    max_t=np.max(np.abs(np.stack(all_boot)),axis=0);critical=float(np.quantile(max_t,.95))
    for r in pairwise:
        r.update(global_112_comparison_maxT_low=r['delta_vs_lambda0_pp']-critical*r['bootstrap_se'],global_112_comparison_maxT_high=r['delta_vs_lambda0_pp']+critical*r['bootstrap_se'],global_critical=critical)
    write('paired_validation_intervals.csv',pairwise);write('qwen_hallucination_displacement.csv',distributions);write('llama_sanity_cached.csv',llama)
    # Compare centered linear/quadratic kernel contributions in forward and inverse fits.
    kernels=[]
    for key in ['qwen3','llama3b']:
        for c in NAMES:
            cb=prev.AttenuatedCurveball.restore(joblib.load(prev.fit_path(key,c,1.)))
            def center(a):return a-a.mean(0,keepdims=True)-a.mean(1,keepdims=True)+a.mean()
            x=cb.fit.X_fit_;s=.01*x@x.T;lin=center(2*s);quad=center(s*s)
            z=cb.fit.X_transformed_fit_;si=.01*z@z.T
            kernels.append(dict(model=key,concept=c,fit_weight=1.,centered_forward_quadratic_to_linear_frobenius=float(np.linalg.norm(quad)/np.linalg.norm(lin)),inverse_quadratic_to_linear_frobenius=float(np.linalg.norm(si*si)/np.linalg.norm(2*si)),centered_activation_norm_rms=float(np.sqrt(np.mean(np.sum(x*x,axis=1)))),latent_norm_rms=float(np.sqrt(np.mean(np.sum(z*z,axis=1)))),fit_weight_attribute=float(cb.fit.quadratic_weight)))
    write('kernel_component_diagnostics.csv',kernels);verify()
    print(json.dumps(dict(selection=selection,qwen_hallucination=[r for r in pairwise if r['model']=='qwen3' and r['concept']=='hallucination' and r['weight']==.01],displacements=distributions)),flush=True)

if __name__=='__main__':main()
