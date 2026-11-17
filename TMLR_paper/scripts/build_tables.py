"""Compile fixed-benchmark tables without LLM inference or test selection."""
from artifact_common import *

def main():
 mainrows=[];appendix=[];sourcefiles=set()
 for model in MODELS:
  src=base(model)/'summary.csv';sourcefiles.add(src)
  summaries=[r for r in rows(src) if r['model']==model]
  if model!='llama2':
   sourcefiles.add(TEST/'test_summary.csv')
   summaries += [dict(r,partition='benchmark') for r in rows(TEST/'test_summary.csv') if r['model']==model]
  for r in summaries:r['exploratory_fixed_benchmark']=r['partition']=='benchmark'
  appendix+=summaries
  ix={(r['concept'],r['partition'],r['method']):r for r in summaries}
  for c,w in selected(model).items():
   a=ix[c,'benchmark',f'lambda_{w}'];l=ix[c,'benchmark','Linear'];z=ix[c,'benchmark','lambda_0.0']
   r=dict(model=model,model_name=MODELS[model],concept=c,behavior=NAMES[c],selected_lambda=w,lambda_zero_is_linear_kernel_limit=w==0,questions=int(a['questions']),groups=int(a['groups']),linear_delta_pp=float(l['positive_delta_pp']),curveball_delta_pp=float(a['positive_delta_pp']),lambda0_delta_pp=float(z['positive_delta_pp']),difference_pp=float(a['vs_linear_pp']),difference_ci_low=float(a['vs_linear_ci_low']),difference_ci_high=float(a['vs_linear_ci_high']),vs_lambda0_pp=float(a['vs_lambda0_pp']),vs_lambda0_ci_low=float(a['vs_lambda0_ci_low']),vs_lambda0_ci_high=float(a['vs_lambda0_ci_high']),nohook_probability=float(a['nohook_probability']),positive_probability=float(a['positive_probability']),exploratory_fixed_benchmark=True)
   r['difference_ci_crosses_zero']=r['difference_ci_low']<=0<=r['difference_ci_high'];mainrows.append(r)
 write_csv(PAPER/'data/main_results.csv',mainrows);write_csv(PAPER/'data/training_validation_test.csv',appendix)
 body=[]
 for c in NAMES:
  line=[NAMES[c]]
  for m in MODELS:
   r=next(r for r in mainrows if r['model']==m and r['concept']==c)
   line += [f"{r['linear_delta_pp']:+.3f}",f"{r['curveball_delta_pp']:+.3f}",f"{r['selected_lambda']:g}",f"{r['difference_pp']:+.3f} [{r['difference_ci_low']:+.3f}, {r['difference_ci_high']:+.3f}]"]
  body.append(line)
 caption='Exploratory fixed-benchmark evaluation on the seven official CAA behaviors. Values are positive minus true no-hook CAA-style all-option normalized target-label probability, in percentage points. Curveball uses the validation-selected quadratic weight and natural residual-preserved displacement. Each difference is Curveball minus CAA Linear with a pointwise paired source-group bootstrap 95\\% interval (10,000 resamples), without a multiplicity-corrected significance claim. Lambda zero is the linear-kernel KPCA limit, not evidence of nonlinear improvement. Model-specific blocks: Qwen3 18, Llama3B 9, Llama2 13; all interventions affect only the last real token. This is not the unmodified CAA A/B scorer.'
 tex_table(PAPER/'tables/main_results.tex',['Behavior']+[f'{MODELS[m]} {x}' for m in MODELS for x in ['Linear','Curveball','lambda','Difference [95% CI]']],body,caption,'tab:main_results')
 from format_tables import format_main
 format_main(caption)
 for model in MODELS:
  ix={(r['concept'],r['partition'],r['method']):r for r in appendix if r['model']==model}
  body=[]
  for c,w in selected(model).items():
   vals=[NAMES[c],f'{w:g}']
   for part in ['fit','validation','benchmark']:
    for method in ['Linear','lambda_0.0',f'lambda_{w}','lambda_1.0']:vals.append(f"{float(ix[c,part,method]['positive_delta_pp']):+.3f}")
   body.append(vals)
  tex_table(PAPER/f'tables/appendix_splits_{model}.tex',['Behavior','Selected lambda']+[f'{p}: {m}' for p in ['Training','Validation','Test'] for m in ['Linear','lambda0','selected','lambda1']],body,MODELS[model]+': fixed selected configurations across all splits; positive delta probability in pp. Training is in-sample. Official Test is exploratory.','tab:splits_'+model)
  for part in ['fit','validation']:
   methods=['Linear']+[f'lambda_{w}' for w in WEIGHTS]
   tex_table(PAPER/f'tables/appendix_grid_{model}_{part}.tex',['Behavior']+methods,[[NAMES[c]]+[f"{float(ix[c,part,m]['positive_delta_pp']):+.3f}" for m in methods] for c in NAMES],MODELS[model]+': complete prespecified '+part+' quadratic-weight grid. Lambda1 is a reference excluded from selection; native positive step is fixed across weights.','tab:grid_'+model+'_'+part)
 stats=[]
 cfg=config('llama2')
 upstream=ROOT/'experiments/caa_official_seven_behaviors_four_models/upstream/datasets'
 for c in NAMES:
  row=dict(concept=c,behavior=NAMES[c]);allsets={}
  for part in ['fit','validation','benchmark']:
   rs=split_rows(c,part);g={r['group_id'] for r in rs};allsets[part]=g
   row[part+'_questions']=len(rs);row[part+'_source_groups']=len(g)
  assert not any(allsets[a]&allsets[b] for a,b in [('fit','validation'),('fit','benchmark'),('validation','benchmark')])
  ng=len(load(upstream/'generate'/c/'generate_dataset.json'));row['official_generate_questions']=ng
  row['excluded_generate_questions']=ng-row['fit_questions']-row['validation_questions']
  row.update(fit_positive_activations=row['fit_questions'],fit_negative_activations=row['fit_questions'],fit_total_activation_samples=2*row['fit_questions'],split_group_overlap=0,counts_per_model=True)
  stats.append(row)
 assert sum(r['benchmark_questions'] for r in stats)==350
 write_csv(PAPER/'data/dataset_statistics.csv',stats)
 tex_table(PAPER/'tables/dataset_statistics.tex',['Behavior','Fit Q / G','Val Q / G','Test Q / G','Fit pos.','Fit neg.','Excluded generate Q'],[[r['behavior']]+[f"{r[p+'_questions']} / {r[p+'_source_groups']}" for p in ['fit','validation','benchmark']]+[r['fit_positive_activations'],r['fit_negative_activations'],r['excluded_generate_questions']] for r in stats],'CAA data statistics, identical for all three models. Q denotes question records and G denotes unique source groups within each concept. Each fitting question supplies one matching and one nonmatching completed-answer activation; these are two activation samples, not two independent questions. Test-linked generate groups are excluded before fitting. All 350 official test questions and labels are retained.','tab:dataset_statistics')
 write_csv(PAPER/'data/dataset_split_records.csv',SPLIT)
 save(PAPER/'data/table_source_manifest.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in sorted(sourcefiles)},main_rows=len(mainrows),split_manifest_sha256=sha(ROOT/'results/qwen_quadratic_attenuation_20261103/split_manifest.json'),group_disjoint=True,metric='all-option normalized target-label probability; pp',no_new_inference=True))
 from refined_figures import main_results_figure
 main_results_figure()
 # Cache the existing causal diagnostic; do not recompute or rename it a steering method.
 for f in ['qwen_factorial_contrasts.csv','paired_validation_intervals.csv','qwen_hallucination_displacement.csv']:
  import shutil;shutil.copy2(TEST/f,PAPER/'data'/f)
 contrast=rows(TEST/'qwen_factorial_contrasts.csv');save(PAPER/'data/hallucination_case_sources.json',dict(files={str((TEST/f).relative_to(ROOT)):sha(TEST/f) for f in ['qwen_factorial_contrasts.csv','paired_validation_intervals.csv','qwen_hallucination_displacement.csv']}))
 print(json.dumps(dict(stage='tables_complete',main_conditions=len(mainrows),appendix_rows=len(appendix),datasets=len(stats))),flush=True)
if __name__=='__main__':main()
