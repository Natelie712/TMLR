"""Publication-ready vector figures and a report; no experiment selection."""
import fcntl,html,math
from markdown_it import MarkdownIt
from sklearn.decomposition import PCA
from artifact_common import *
COLORS=['#2b6f8e','#dc8b26','#8a4d8b','#507a40']

def geometry_figures():
 p=PAPER/'data/geometric_diagnostics'
 if not (p/'completion.json').exists():return False
 from refined_figures import geometry_figures as refined_geometry
 refined_geometry()
 # Keep main Table2 readable, with the full weight comparison as an appendix.
 full=rows(p/'magnitude_correlation.csv');sel=[r for r in full if r['selected']=='True'];fmt=lambda x:'--' if x=='' else f'{float(x):.3f}'
 tex_table(PAPER/'tables/magnitude_correlation_selected.tex',['Model','Behavior','lambda','Q / G','rho','95% CI','Holm p'],[[MODELS[r['model']],r['behavior'],r['weight'],r['questions']+' / '+r['groups'],fmt(r['group_spearman']),'['+fmt(r['ci_low'])+', '+fmt(r['ci_high'])+']',fmt(r.get('holm_adjusted_p',''))] for r in sel],'Validation-selected Curveball magnitude versus genuine paired activation distance. Source-group means; 1,000 group-bootstrap intervals; Holm adjustment over the complete tested geometry family. Near-constant magnitudes are undefined, shown as --. Full lambda 0/nonzero/lambda 1 comparisons accompany the data.','tab:magnitude_selected')
 return True

def case_figure():
 from refined_figures import case_figure as refined_case
 refined_case()


def response_figures():
 plt=plot_style();root=PAPER/'data/response_curves';stats=[];statuses=[]
 for model in MODELS:
  status=load(root/f'{model}_status.json') if (root/f'{model}_status.json').exists() else dict(status='not_started',completed=[])
  statuses.append(dict(model=model,**status))
  for ci,c in enumerate(NAMES):
   p=root/model/f'{c}.jsonl'
   if not p.exists():continue
   rr=[]
   for line in p.read_text().splitlines():
    try:rr.extend(json.loads(line)['rows'])
    except json.JSONDecodeError:break
   expected={r['index'] for r in split_rows(c,'validation')}
   for method in sorted({r['method'] for r in rr}):
    for a in [-1.,-.5,0.,.5,1.]:
     rs=sorted([r for r in rr if r['method']==method and r['multiplier']==a],key=lambda r:r['index'])
     if {r['index'] for r in rs}!=expected:continue
     assert len(rs)==len(expected);groups=sorted({r['group_id'] for r in rs});codes=np.array([groups.index(r['group_id']) for r in rs]);pval=np.array([r['normalized_target_probability'] for r in rs]);cnt=np.bincount(codes);sums=np.bincount(codes,weights=pval);rng=np.random.default_rng(20261104+ci*3+1);draws=rng.integers(0,len(groups),(10000,len(groups)));boot=sums[draws].sum(1)/cnt[draws].sum(1);lo,hi=np.quantile(boot,[.025,.975]);disp=np.concatenate([r['displacement']['actual_displacement'] for r in rs])
     stats.append(dict(model=model,concept=c,method=method,multiplier=a,questions=len(rs),groups=len(groups),probability=float(pval.mean()),ci_low=float(lo),ci_high=float(hi),displacement_rms=float(np.sqrt(np.mean(disp**2))),displacement_median=float(np.median(disp)),displacement_p05=float(np.quantile(disp,.05)),displacement_p95=float(np.quantile(disp,.95)),provenance=rs[0]['provenance']))
 write_csv(PAPER/'data/response_summary.csv',stats);save(PAPER/'data/execution/response_status.json',statuses)
 from refined_figures import response_panels
 response_panels(stats,statuses)
 return statuses,stats


def main():
 lock=(PAPER/'data/execution/render.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
 complete_geometry=geometry_figures();case_figure();statuses,curves=response_figures();mainrows=rows(PAPER/'data/main_results.csv')
 body=[[r['model_name'],r['behavior'],r['selected_lambda'],f"{float(r['linear_delta_pp']):+.3f}",f"{float(r['curveball_delta_pp']):+.3f}",f"{float(r['difference_pp']):+.3f} [{float(r['difference_ci_low']):+.3f}, {float(r['difference_ci_high']):+.3f}]"] for r in mainrows]
 md='# Curveball TMLR revision: empirical results\n\n## Main result\n\nAll 21 model/behavior test comparisons are reused from completed, validation-frozen runs. No test search was performed in this revision. Values are CAA-style all-option normalized target-label probability changes in percentage points, with a shared true no-hook baseline.\n\n'+table(['Model','Behavior','Selected lambda','Linear','Curveball','Difference [paired 95% CI]'],body)
 md+='\n\n## Current completion status\n\n'+table(['Model','Response diagnostics','Completed behaviors'],[[s['model'],s['status'],len(s['completed'])] for s in statuses])+f'\n\nCPU geometry complete: {complete_geometry}. Complete response-curve cells currently available: {len(curves)}. Pending panels are labelled pending and must not be submitted as completed figures.\n'
 if (PAPER/'reports/geometric_findings.md').exists():md+='\n\n'+(PAPER/'reports/geometric_findings.md').read_text().replace('# CPU geometric findings','## CPU geometric findings',1)
 md+='''
## Claims supported and claims requiring revision

- The original broad claim that Curveball consistently outperforms Linear across models and behaviors is not supported by these CAA results. All weak/negative behaviors remain in the table.
- Qwen Hallucination supports a focused nonlinear-weight case: lambda 0.01 improves over lambda 0 on validation and the repeatedly used official benchmark. The verified validation crossover attributes the gain primarily to direction, not a larger L2 budget. These interventions are diagnostic counterfactuals, not proposed algorithms.
- A lambda 0 advantage is a linear-kernel KPCA pipeline effect; it does not demonstrate nonlinear steering benefit. In particular, differences in displacement, component truncation and inverse reconstruction remain relevant.
- Llama3B's tiny weight sensitivity applies to this block, kernel scale, fitting data and frozen native step. It does not prove that the model lacks nonlinear structure.
- On Llama2, four behaviors select lambda 0. Nonzero weights modestly improve Myopic Reward and Survival Instinct relative to lambda 0 while remaining below direct Linear. Sycophancy remains a negative steering result.
- Local direction variation, projected clusters or magnitude correlations do not themselves demonstrate causality, intrinsic manifold curvature or a behavioral advantage. Near-constant magnitudes are excluded from meaningful correlation claims. The two-component GMM diagnostic is descriptive and projection-dependent.
- New CAA data contain no Humor/Rudeness/Excitement/Sadness judge or classifier modality. Original claims about tone steering and tone-vs-behavior geometry must be removed from this empirical revision.
- Figure 2B VAE distortion is not reproduced. No new-data curvature–steering claim should be carried forward. Independent synthetic panels remain pending provenance confirmation, not silently relabelled as CAA geometry.

## Methods and statistical limits

Training uses only fitting groups and is evaluated in-sample. Validation selects lambda from the already tested grid; selected maxima can be optimistic. Official Test is an exploratory fixed benchmark because previous diagnoses repeatedly inspected it. Published mean effects and 10,000 paired source-group bootstrap intervals are reused without changing the metric or labels. Pointwise intervals do not imply multiplicity-corrected superiority.

The primary interventions use last-token block outputs, natural residual-preserved Curveball updates, fixed 20 KPCA components, weighted degree-2 kernel with gamma=0.01 and inverse ridge=1. Native Curveball steps are training-only RMS-calibrated at lambda 1 and then held fixed across weights; actual evaluation displacements are not assumed equal to Linear. Llama2 uses its validated parenthesis prompt; Qwen/Llama3B use the frozen assistant-answer prefill. All methods within each model use identical prompts and labels. No Angular values with a mismatched multi-token scope are included.

Paired-label geometry uses actual positive/nonmatching completed-answer activations from the same source question. Figure 6 MCQ magnitudes instead use the final input token at the answer-decision position. These locations are separately labelled and should not be conflated. Relative preimage reconstruction error is not residual-preserved steering error. Kernel variance fractions refer to the centered RKHS kernel spectrum, not raw activation-space explained variance.

Table 2 uses validation group-mean observations, 1,000 group bootstraps and 1,999 group permutations with Holm correction across all nonconstant tested rows. A separate full in-sample fitting-data correlation table uses genuine paired training activations and the identical diagnostic weights; its Holm family is separate from validation. No MCQ probability is used as a paired-target distance. Table 4 averages cross-prompt cosines across distinct source groups and uses group bootstrap. Figure 4 clusters fitting negative group-centroid activations (k=8, n_init=20, seed=20261111) and compares 500 size-matched random partitions. This group aggregation is a documented adaptation of the manuscript's negative-activation clustering.

## Placement and numbering

Open the [figure gallery](figure_gallery.html) for completed figures and the [scientific figure review](figure_refinement_review.md) for numerical checks and interpretation limits. Incomplete response figures are stored in `figures/drafts/` and excluded from the manuscript bundle.

See [artifact inventory](paper_artifact_inventory.md), [execution plan](experiment_execution_plan.md), and [VAE resource assessment](vae_resource_assessment.md). Tables 1–4 retain their original scientific roles. Figures 8–9 are removed; tone Figures 17–20 are replaced as documented. Figure 21 adds Llama2 response curves. All PDFs are vector Matplotlib output with embedded TrueType fonts. Use booktabs and graphicx for the LaTeX tables. Historical inputs remain read-only during this revision; no GitHub push.
'''
 (PAPER/'reports/paper_results_report.md').write_text(md);(PAPER/'reports/paper_results_report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><style>body{font:15px/1.6 system-ui;margin:30px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}</style>'+MarkdownIt('commonmark').enable('table').render(md)+'</html>')
 route=load(PAPER/'data/figure_routing.json')
 numbers=[1,20]+([4,5,6,7,12]+list(range(13,20)) if complete_geometry else [])+route['complete']
 captions={1:'All three models and seven CAA behaviors: official-test positive target probability changes. Curveball weights were selected on validation. Exploratory fixed-benchmark evaluation.',3:'Five-point validation response diagnostics. Signed multipliers scale each method\'s frozen native strength; they do not imply matched activation displacement. Zero is a true no-hook baseline. Pending cells are explicitly marked.',4:'Local contrast directions from k=8 negative-activation group clusters and corresponding genuine positive pairs, versus 500 size-matched random group partitions. Qwen Hallucination and Survival Instinct, fitting data only.',5:'Unit residual-preserved Curveball update directions at negative completed-answer validation activations. Horizontal axis is the CAA Linear direction; vertical axis is a common leading orthogonal component. Direction variation does not alone establish multimodality.',6:'Actual last-token MCQ activation displacement distributions from saved validation forwards. Native strengths remain fixed; no per-token normalization is applied.',7:'Quadratic-weight reconstruction sensitivity and the actual fraction of centered-kernel variance retained by 20 components. The 95 percent reference line is not a guaranteed selection criterion.',12:'Cosine distributions of Algorithm 2 actual updates relative to CAA Linear across all three models and seven behaviors. Completed-negative-answer validation activations; no assumption of multimodality.',20:'Qwen Hallucination case study, reusing verified results. Similar RMS displacement at lambda 0 and 0.01 accompanies a direction change and stronger probability steering. Direction/magnitude crossover intervals are diagnostic counterfactuals, not a new algorithm.'}
 for n in [10,11,21]:captions[n]='All seven CAA validation behaviors: fixed five-point signed-strength response diagnostics for '+{10:'Qwen3',11:'Llama3B',21:'Llama2'}[n]+'. Not a dense sweep. No test-based strength selection.'
 for n,c in zip(range(13,20),['corrigible-neutral-HHH','coordinate-other-ais','hallucination','myopic-reward','survival-instinct','sycophancy','refusal']):captions[n]=NAMES[c]+': completed-negative-answer validation update cosines, unit-direction projections and natural displacement magnitudes across the three models. These are paired-activation geometry diagnostics, distinct from MCQ decision-token geometry.'
 captions.update({
  1:'Exploratory official-test positive probability changes (top) and paired Curveball-minus-Linear differences (bottom). All seven behaviors and three models are retained. Pointwise source-group bootstrap 95 percent intervals; selected weights are in Table 1. Equal visual scales are used across models within each row.',
  4:'Qwen fitting-data local unit contrast directions (left) and empirical CDFs of cosine to global CAA Linear (right). Eight negative-activation clusters versus 500 size-matched random partitions. Point colors show the fraction of questions with target label A and reveal a strong label-composition confound. Cluster IDs are presentation labels, not classes. PCA views do not establish behavioral manifold structure.',
  5:'Qwen Hallucination validation completed-answer activations. Actual residual-preserved updates are normalized only for direction visualization and projected onto the same orthonormal axes for all three weights. The unit circle bounds the projection; common limits prevent quantization-scale variation from being magnified. This does not establish full-space multimodality.',
  6:'Validation MCQ decision-token displacement ECDFs. The horizontal axis divides each observed Curveball update norm by the same-question actual Linear norm for display; interventions themselves are not displacement-matched. All panels share the same axis. Legend RMS values remain in original activation L2 units. Near-constant distributions appear as legitimate vertical steps.',
  7:'Fitting-data inverse reconstruction sensitivity and centered-kernel variance retained by 20 components. Reconstruction error is the Frobenius error divided by the Frobenius norm of the original uncentered activation matrix. Common vertical scales are used across models. The horizontal axis is linear near zero and logarithmic beyond 0.001. Each weight defines a different kernel, so retained fractions are not explained variance in one common feature space. The dotted 95 percent line is a reference, not a satisfied selection rule.',
  12:'Validation completed-answer update cosine ECDFs across all seven behaviors and three models. Colors identify fixed numerical kernel weights consistently. Each panel states its validation-selected weight. The ECDF avoids density spikes for nearly constant updates. There is no inference that every distribution is multimodal.',
  20:'Verified Qwen Hallucination case study. Positive probability changes with pointwise 95 percent intervals; RMS and update cosine shown on axes that do not magnify the roughly 0.2 percent RMS difference; direction/magnitude crossover contributions with paired intervals. Official Test is exploratory. The crossover is a diagnostic counterfactual, not a new steering algorithm.'})
 for n,c in zip(range(13,20),['corrigible-neutral-HHH','coordinate-other-ais','hallucination','myopic-reward','survival-instinct','sycophancy','refusal']):
  captions[n]=NAMES[c]+': validation completed-answer activation diagnostics. Columns show update cosine ECDFs, projections on common unit-direction axes, and displacement ECDFs scaled by the CAA Linear vector norm for display only. The actual intervention retains natural magnitude. All numerical weights use the same colors across figures; selected weights are marked in each model legend. These activations differ from the MCQ decision token.'
 (PAPER/'reports/figure_captions.md').write_text('# Figure captions\n\n'+'\n\n'.join(f'Figure {n}. '+captions[n] for n in sorted(numbers+route['draft'])))
 tex=[]
 for n in sorted(numbers):tex += [r'\setcounter{figure}{'+str(n-1)+'}',r'\begin{figure*}[p]',r'\centering',r'\includegraphics[width=0.95\textwidth,height=0.85\textheight,keepaspectratio]{figures/figure_'+str(n)+'.pdf}',r'\caption{'+captions[n]+'}',r'\label{fig:revision_'+str(n)+'}',r'\end{figure*}']
 (PAPER/'tables/figure_includes.tex').write_text('\n'.join(tex)+'\n')
 save(PAPER/'data/artifact_status.json',dict(main_table_complete=True,main_conditions=21,geometry_complete=complete_geometry,response_complete=all(s['status']=='complete' for s in statuses),response_status=statuses,figure2_reproduced=False,removed_original_figures=[8,9],generated_figures=sorted(numbers),draft_figures=route['draft'],historical_test_exploratory=True))
 print(json.dumps(dict(stage='rendered',geometry_complete=complete_geometry,response_cells=len(curves))),flush=True)
if __name__=='__main__':main()
