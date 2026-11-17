"""Additional read-only scientific checks and compact geometric findings."""
from artifact_common import *

def main():
 sets={p:{r['group_id'] for r in SPLIT if r['partition']==p} for p in ['fit','validation','benchmark']}
 overlap={a+'_'+b:len(sets[a]&sets[b]) for a,b in [('fit','validation'),('fit','benchmark'),('validation','benchmark')]};assert not any(overlap.values())
 save(PAPER/'data/global_split_leakage_check.json',dict(group_overlaps=overlap,passed=True,limitations='Frozen grouping catches specified lexical/source overlap, not every possible semantic relation.'))
 results=[]
 for model in MODELS:
  for c in NAMES:
   z=np.load(PAPER/'data/geometric_diagnostics'/f'local_{model}_{c}.npz');source=load(ROOT/'experiments/caa_official_seven_behaviors_four_models/upstream/datasets/generate'/c/'generate_dataset.json');fitrs=split_rows(c,'fit');gid2cluster=dict(zip(z['group_ids'],z['cluster_labels']))
   for k in range(8):
    selectedrows=[r for r in fitrs if gid2cluster[r['group_id']]==k];labels=[source[r['index']]['answer_matching_behavior'] for r in selectedrows]
    counts={l:labels.count(l) for l in sorted(set(labels))};results.append(dict(model=model,concept=c,cluster=k,question_records=len(labels),target_label_counts=counts,dominant_target_label_fraction=max(counts.values())/len(labels)))
 write_csv(PAPER/'data/geometric_diagnostics/cluster_target_label_audit.csv',results)
 eig=rows(PAPER/'data/geometric_diagnostics/kernel_variance_reconstruction.csv');corr=rows(PAPER/'data/geometric_diagnostics/magnitude_correlation.csv');consistency=rows(PAPER/'data/geometric_diagnostics/cross_prompt_consistency.csv');local=rows(PAPER/'data/geometric_diagnostics/local_directions.csv')
 lines=['# CPU geometric findings','',f'All {len(eig)} fitted kernels retain less than95% of centered-kernel variance with the frozen20 components. The retained fraction ranges from {min(float(r["retained_kernel_variance_fraction"]) for r in eig):.3%} to {max(float(r["retained_kernel_variance_fraction"]) for r in eig):.3%}. This measures RKHS variance, not ambient activation variance. The manuscript must not claim an automatically satisfied95% rule.','']
 for model in MODELS:
  rs=[r for r in consistency if r['model']==model and r['partition']=='validation'];values=[float(r['mean_cosine']) for r in rs];lines.append(f'{MODELS[model]} validation cross-prompt mean cosine spans {min(values):.4f} to {max(values):.4f} across the seven behaviors. This shows model/behavior dependence rather than a universal coherent contrast direction.')
 lines+=['','Selected Qwen Hallucination lambda0.01 has paired-negative-activation magnitude/distance group Spearman rho='+next(r['group_spearman'] for r in corr if r['model']=='qwen3' and r['concept']=='hallucination' and r['selected']=='True')+'. Its interval crosses zero. The strong MCQ effect therefore does not establish the original paper\'s proposed positive magnitude-distance correlation mechanism. This analysis is at completed-negative-label activations, a different location from the MCQ decision token.','']
 for c in ['hallucination','survival-instinct']:
  rs=[r for r in local if r['model']=='qwen3' and r['concept']==c];a=np.mean([float(r['cosine_to_global_linear']) for r in rs]);b=np.mean([float(r['random_control_mean']) for r in rs]);lines.append(f'Qwen {NAMES[c]}: mean cluster-direction cosine={a:.4f}; mean size-matched random-group-partition cosine={b:.4f}. These are descriptive fitting-data diagnostics, not independent behavioral validation.')
 audited=[r for r in results if r['model']=='qwen3' and r['concept']=='hallucination']
 purity=[r['dominant_target_label_fraction'] for r in audited]
 lines+=['',f'Qwen Hallucination local clusters are strongly associated with target answer labels: {sum(x==1 for x in purity)}/8 clusters contain only one target label; the remaining cluster majority fractions are at least {min(purity):.1%}. This is empirical evidence of a label-geometry confound, not proof that every nonlinear update is a label shortcut. CAA paired examples differ in completed answer labels. Genuine one-to-one pairing alone does not remove this confound; do not equate local clustering with a learned behavioral manifold. Full counts are in cluster_target_label_audit.csv.']
 fitfile=PAPER/'data/geometric_diagnostics/fit_magnitude_correlation.csv'
 if fitfile.exists():
  r=next(r for r in rows(fitfile) if r['model']=='qwen3' and r['concept']=='hallucination' and r['selected']=='True')
  lines+=['',f"The corresponding in-sample training magnitude-distance correlation is rho={float(r['group_spearman']):.4f}, 95% CI [{float(r['ci_low']):.4f}, {float(r['ci_high']):.4f}] over {r['groups']} source groups. This also does not support a positive correlation for the selected Hallucination configuration."]
 modes=rows(PAPER/'data/geometric_diagnostics/direction_modality.csv')
 vals={float(r['weight']):r for r in modes if r['model']=='qwen3' and r['concept']=='hallucination'}
 lines+=['',f"For Qwen Hallucination, the common 2D unit-update projection has two-minus-one-component GMM BIC {float(vals[.01]['projection_gmm_bic_2_minus_1']):.2f} at lambda 0.01 and {float(vals[1.]['projection_gmm_bic_2_minus_1']):.2f} at lambda 1. A two-component mixture fits these projected samples better under this diagnostic, while lambda 0 is nearly constant. This does not establish genuine multimodality: a mixture can approximate skewed unimodal data, and projection and answer-label structure remain confounds.",'',f'Global frozen source-group overlap checks passed: {overlap}. No label corrections or exclusions were made.']
 lines=[s.replace('than95','than 95').replace('frozen20','frozen 20').replace('satisfied95','satisfied 95').replace('lambda0.01','lambda 0.01') for s in lines]
 (PAPER/'reports/geometric_findings.md').write_text('\n'.join(lines)+'\n');print('Additional geometry and global leakage checks complete.',flush=True)
if __name__=='__main__':main()
