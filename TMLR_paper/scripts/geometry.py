"""CPU-only paired-activation geometry; no LLM fitting, inference, or test tuning."""
import time
import joblib,torch
from scipy.stats import spearmanr,rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from artifact_common import *

def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-30)
def group_arrays(values,groups):
 gs=sorted(set(groups));return np.stack([values[np.array(groups)==g].mean(0) for g in gs]),gs

def corr_stats(mag,distance,groups,seed):
 x,gs=group_arrays(mag,groups);y,_=group_arrays(distance,groups);rng=np.random.default_rng(seed)
 out=dict(questions=len(mag),groups=len(gs),sample_unit='source-group means of genuine paired question distances and update norms',row_spearman=None,group_spearman=None,ci_low=None,ci_high=None,permutation_p=None,near_constant_magnitude=bool(np.ptp(x)<=1e-5*max(abs(x.mean()),1e-30)))
 if out['near_constant_magnitude'] or np.ptp(y)==0:return out
 rho=float(spearmanr(x,y).statistic);boots=[]
 for _ in range(1000):
  idx=rng.integers(0,len(x),len(x));v=float(spearmanr(x[idx],y[idx]).statistic)
  if np.isfinite(v):boots.append(v)
 rx=rankdata(x);ry=rankdata(y);rx-=rx.mean();ry-=ry.mean();den=np.linalg.norm(rx)*np.linalg.norm(ry)
 perm=np.array([np.dot(rx,rng.permutation(ry))/den for _ in range(1999)])
 out.update(row_spearman=float(spearmanr(mag,distance).statistic),group_spearman=rho,ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)),permutation_p=float((1+np.sum(abs(perm)>=abs(rho)))/2000),bootstrap_resamples=1000,permutation_resamples=1999)
 return out

def cross_prompt(delta,groups,seed):
 u=unit(delta);names=sorted(set(groups));codes=np.array([names.index(g) for g in groups]);S=np.zeros((len(names),u.shape[1]));np.add.at(S,codes,u);n=np.bincount(codes);ss=(S*S).sum(1)
 denominator=float(n.sum()**2-(n*n).sum());point=float(((S.sum(0)**2).sum()-ss.sum())/denominator)
 rng=np.random.default_rng(seed);boot=[]
 for _ in range(10):
  mult=rng.multinomial(len(names),np.ones(len(names))/len(names),100).astype(float);agg=mult@S
  num=(agg*agg).sum(1)-(mult*mult)@ss;dn=(mult@n)**2-(mult*mult)@(n*n)
  boot.extend((num/dn).tolist())
 return dict(questions=len(u),groups=len(names),distinct_source_group_question_pairs=int(denominator/2),mean_cosine=point,ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),bootstrap_resamples=1000,definition='Mean cosine of actual paired positive-minus-negative vectors across distinct source groups; row pairs weighted equally; source-group bootstrap')

def main():
 start=time.time();torch.set_num_threads(2);CB=fit_class();plt=plot_style();correlations=[];consistency=[];local=[];eigrows=[];geos=[];mcq=[];modality=[];checks=[]
 for mi,model in enumerate(MODELS):
  cfg=config(model);folder=Path(cfg['activation_sources'][model]);vec=torch.load(base(model)/'fits'/model/'linear_vectors.pt',weights_only=True)['normalized'];steps=load(base(model)/'fits'/model/'positive_steps.json')['steps'];choice=selected(model)
  prompts=[json.loads(s) for s in (folder/'training_prompts.jsonl').read_text().splitlines()];pix={(r['concept'],r['index'],r['sign']):r for r in prompts}
  for ci,c in enumerate(NAMES):
   cache=torch.load(folder/f'train_activations_{c}.pt',weights_only=True);src=load(ROOT/'experiments/caa_official_seven_behaviors_four_models/upstream/datasets/generate'/c/'generate_dataset.json')
   assert cache['source_indices']==list(range(len(src)));assert cache['positive'].shape==cache['negative'].shape
   for i,r in enumerate(src):
    assert pix[c,i,1]['answer_label']==r['answer_matching_behavior'].strip('()')
    assert pix[c,i,-1]['answer_label']==r['answer_not_matching_behavior'].strip('()')
   checks.append(dict(model=model,concept=c,paired_rows=len(src),positive_negative_source_indices_identical=True,all_prompt_labels_verified=True,cache_sha256=sha(folder/f'train_activations_{c}.pt')))
   fitrs=split_rows(c,'fit');valrs=split_rows(c,'validation');fi=[r['index'] for r in fitrs];vi=[r['index'] for r in valrs]
   fp=cache['positive'][fi].double().numpy();fn=cache['negative'][fi].double().numpy();vp=cache['positive'][vi].double().numpy();vn=cache['negative'][vi].double().numpy()
   fitgroups=[r['group_id'] for r in fitrs];valgroups=[r['group_id'] for r in valrs];v=vec[c].double().numpy();vhat=unit(v)
   for part,p,n,g in [('fit',fp,fn,fitgroups),('validation',vp,vn,valgroups)]:consistency.append(dict(model=model,concept=c,behavior=NAMES[c],partition=part,**cross_prompt(p-n,g,20261111+mi*100+ci)))
   # Match cluster sizes exactly in random group-partition controls. No fabricated pairs.
   gn,gnames=group_arrays(fn,fitgroups);gd,_=group_arrays(fp-fn,fitgroups)
   km=KMeans(n_clusters=8,n_init=20,random_state=20261111).fit(gn)
   dirs=np.stack([gd[km.labels_==k].mean(0) for k in range(8)]);cos=unit(dirs)@vhat
   rng=np.random.default_rng(20261111+ci);rand=[]
   for _ in range(500):
    labels=rng.permutation(km.labels_);d=np.stack([gd[labels==k].mean(0) for k in range(8)]);rand.append(unit(d)@vhat)
   rand=np.array(rand)
   for k in range(8):local.append(dict(model=model,concept=c,cluster=k,source_groups=int((km.labels_==k).sum()),cosine_to_global_linear=float(cos[k]),random_control_mean=float(rand[:,k].mean()),random_control_p025=float(np.quantile(rand[:,k],.025)),random_control_p975=float(np.quantile(rand[:,k],.975)),random_partitions=500,clustering_input='Within-source-group mean negative completed-answer activation; k=8; kmeans n_init=20',local_direction='Mean of corresponding genuine paired positive-minus-negative group differences'))
   np.savez_compressed(PAPER/'data/geometric_diagnostics'/f'local_{model}_{c}.npz',directions=dirs,global_direction=v,cosines=cos,random_cosines=rand,group_ids=np.array(gnames),cluster_labels=km.labels_)
   updates={};weights=sorted({0.,.01,1.,choice[c]});h=torch.from_numpy(vn.astype(np.float32));dist=np.linalg.norm(vp-vn,axis=1)
   for w in WEIGHTS:
    obj=CB.restore(joblib.load(fit_path(model,c,w)));x=obj.fit.X_fit_;sn=.01*np.einsum('ij,ij->i',x,x)
    trace=float(np.sum(1+2*sn+w*sn*sn)-len(x)*obj.fit._centerer.K_fit_all_);fraction=float(obj.fit.eigenvalues_.sum()/trace)
    assert 0<=fraction<=1+1e-7
    eigrows.append(dict(model=model,concept=c,weight=w,components=len(obj.fit.eigenvalues_),retained_kernel_variance_fraction=fraction,centered_kernel_trace=trace,fit_relative_reconstruction=float(obj.fit_diagnostics['relative_reconstruction_l2']),meets_95_percent_kernel_variance=fraction>=.95))
    if w not in weights:continue
    assert torch.equal(obj.intervention(h[:4],0),h[:4]);parts=[];errs=[]
    for starti in range(0,len(h),64):
     hh=h[starti:starti+64];z=obj.project(hh);after=obj.intervention(hh,float(steps[c]['1']));delta=(after.double()-hh.double()).numpy();parts.append(delta);errs.extend((obj.inverse(z)+obj.center-hh.double()).norm(dim=-1).numpy().tolist())
    delta=np.concatenate(parts);updates[w]=delta;mag=np.linalg.norm(delta,axis=1);cosine=unit(delta)@vhat
    for j,r in enumerate(valrs):geos.append(dict(model=model,concept=c,partition='validation',index=r['index'],group_id=r['group_id'],weight=w,selected=w==choice[c],activation_location='negative completed-answer-label activation',native_step=steps[c]['1'],activation_norm=float(np.linalg.norm(vn[j])),displacement=float(mag[j]),cosine_to_linear=float(cosine[j]),paired_target_distance=float(dist[j]),absolute_reconstruction_error=errs[j],relative_reconstruction_error=errs[j]/float(np.linalg.norm(vn[j]))))
    correlations.append(dict(model=model,concept=c,behavior=NAMES[c],weight=w,selected=w==choice[c],activation_location='negative completed-answer-label activation',**corr_stats(mag,dist,valgroups,20261111+ci)))
   # One common orthogonal direction axis for all weights, based on normalized updates.
   allu=np.concatenate([unit(u) for u in updates.values()]);orth=allu-np.outer(allu@vhat,vhat)
   axis=PCA(n_components=1,svd_solver='randomized',random_state=42).fit(orth).components_[0]
   projections={w:np.c_[unit(u)@vhat,unit(u)@axis] for w,u in updates.items()}
   for w,xy in projections.items():
    if np.max(np.std(xy,axis=0))<1e-6:bic=None
    else:
     scale=np.maximum(xy.std(0),1e-8);xx=(xy-xy.mean(0))/scale
     bic=float(GaussianMixture(2,random_state=42,n_init=5,reg_covar=1e-5).fit(xx).bic(xx)-GaussianMixture(1,random_state=42,n_init=5,reg_covar=1e-5).fit(xx).bic(xx))
    modality.append(dict(model=model,concept=c,weight=w,cosine_std=float(np.std(xy[:,0])),normalized_orthogonal_projection_std=float(np.std(xy[:,1])),projection_gmm_bic_2_minus_1=bic,interpretation='Projection-only descriptive diagnostic. A lower two-component BIC does not establish full-space multimodality or causal steering gains.'))
   np.savez_compressed(PAPER/'data/geometric_diagnostics'/f'updates_{model}_{c}.npz',indices=np.array(vi),group_ids=np.array(valgroups),global_linear=v,orthogonal_axis=axis,**{f'update_{w}':u for w,u in updates.items()},**{f'projection_{w}':xy for w,xy in projections.items()})
   # Actual MCQ decision-position displacement, read from completed frozen GPU runs.
   for r in raw_rows(model,c,'validation'):
    if r['method']=='No-hook':continue
    d=r['displacement'];mcq.append(dict(model=model,concept=c,index=r['index'],group_id=r['group_id'],method=r['method'],activation_location='last real MCQ decision token',displacement=d['actual_displacement'][0],activation_norm=d['activation_norm'][0],cosine_to_linear=d['cosine_to_actual_signed_linear'][0],reconstruction_l2=d.get('reconstruction_l2',[None])[0] if r['method']!='Linear' else None))
   save(PAPER/'data/geometric_diagnostics/progress.json',dict(model=model,concept=c,elapsed_seconds=time.time()-start,completed=len(checks),expected=21))
   print(json.dumps(dict(stage='geometry',model=model,concept=c,completed=len(checks))),flush=True)
 # Correct the family of permutation tests, retaining raw values for transparency.
 pvals=[(i,r['permutation_p']) for i,r in enumerate(correlations) if r['permutation_p'] is not None];order=sorted(pvals,key=lambda t:t[1]);last=0
 for rank,(i,p) in enumerate(order):last=max(last,min(1.,(len(order)-rank)*p));correlations[i]['holm_adjusted_p']=last
 write_csv(PAPER/'data/geometric_diagnostics/magnitude_correlation.csv',correlations);write_csv(PAPER/'data/geometric_diagnostics/cross_prompt_consistency.csv',consistency)
 write_csv(PAPER/'data/geometric_diagnostics/local_directions.csv',local);write_csv(PAPER/'data/geometric_diagnostics/paired_activation_updates.csv',geos)
 write_csv(PAPER/'data/geometric_diagnostics/mcq_displacements.csv',mcq);write_csv(PAPER/'data/geometric_diagnostics/kernel_variance_reconstruction.csv',eigrows);write_csv(PAPER/'data/geometric_diagnostics/direction_modality.csv',modality)
 stats=[]
 for source,rs in [('paired_negative',geos),('mcq_decision',mcq)]:
  key='weight' if source=='paired_negative' else 'method'
  for model in MODELS:
   for c in NAMES:
    for method in sorted({str(r[key]) for r in rs if r['model']==model and r['concept']==c}):
     selectedrows=[r for r in rs if r['model']==model and r['concept']==c and str(r[key])==method];a=np.array([r['displacement'] for r in selectedrows]);stats.append(dict(model=model,concept=c,activation_location=source,method=method,n=len(a),mean=float(a.mean()),rms=float(np.sqrt(np.mean(a*a))),median=float(np.median(a)),p05=float(np.quantile(a,.05)),p95=float(np.quantile(a,.95)),maximum=float(a.max()),coefficient_of_variation=float(a.std()/a.mean())))
 write_csv(PAPER/'data/geometric_diagnostics/displacement_statistics.csv',stats)
 fmt=lambda x:'--' if x is None else f'{x:.3f}'
 tex_table(PAPER/'tables/magnitude_correlation.tex',['Model','Behavior','lambda','Q / G','rho','95% CI','Holm p'],[[MODELS[r['model']],r['behavior'],r['weight'],f"{r['questions']} / {r['groups']}",fmt(r['group_spearman']),f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]",fmt(r.get('holm_adjusted_p'))] for r in correlations],'Validation-only Spearman correlations between actual residual-preserved update norms and genuine paired positive-minus-negative activation distances at the completed negative answer label. Group-mean observations, 1,000 group-bootstrap resamples and 1,999 two-sided group permutations. Holm adjustment covers all nonconstant rows in this table. Near-constant update magnitudes are marked unavailable; quantization noise is not interpreted as adaptive magnitude. These are activation-space diagnostics, not MCQ probability correlations.','tab:magnitude_correlation')
 tex_table(PAPER/'tables/cross_prompt_consistency.tex',['Model','Behavior','Partition','Q / G','Mean cosine','95% CI'],[[MODELS[r['model']],r['behavior'],r['partition'],f"{r['questions']} / {r['groups']}",fmt(r['mean_cosine']),f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]"] for r in consistency],'Mean pairwise cosine between genuine paired contrast vectors from different source groups. Question pairs are equally weighted; intervals use 1,000 source-group bootstrap samples. Fit is in-sample. No tonal-versus-behavioral comparison is made because tone datasets are absent.','tab:cross_prompt_consistency')
 save(PAPER/'data/geometric_diagnostics/completion.json',dict(complete=True,model_concepts=len(checks),paired_source_checks=checks,elapsed_seconds=time.time()-start,bootstrap_resamples=1000,random_partition_controls=500,no_gpu_inference=True,official_test_used=False))
if __name__=='__main__':main()
