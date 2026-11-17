"""Compact two-panel main table without changing scientific values."""
from artifact_common import *
def format_main(caption=None):
 rs=rows(PAPER/'data/main_results.csv')
 if caption is None:
  caption='Exploratory fixed-benchmark CAA-style all-option normalized target-label probability changes, in percentage points. Top: CAA Linear and validation-selected Curveball at the frozen positive strength. Bottom: paired Curveball-minus-Linear differences and pointwise95\\% source-group bootstrap intervals (10,000 resamples). Lambda0 is the linear-kernel KPCA limit, not a nonlinear improvement. Interventions affect only the last real token and retain natural Curveball displacement. Test questions were previously inspected; no test-based tuning is performed. This is not the original unmodified CAA A/B scorer.'
 lines=[r'\begin{table*}[t]',r'\centering',r'\small',r'\setlength{\tabcolsep}{3pt}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{lrrrrrrrrr}',r'\toprule', ' & '+' & '.join(r'\multicolumn{3}{c}{'+MODELS[m]+'}' for m in MODELS)+r' \\', 'Behavior & '+' & '.join([r'Linear & Curveball & $\lambda$']*3)+r' \\',r'\midrule']
 for c in NAMES:
  rr=[next(r for r in rs if r['model']==m and r['concept']==c) for m in MODELS]
  lines.append(NAMES[c]+' & '+' & '.join(f"{float(r['linear_delta_pp']):+.3f} & {float(r['curveball_delta_pp']):+.3f} & {float(r['selected_lambda']):g}" for r in rr)+r' \\')
 lines += [r'\bottomrule',r'\end{tabular}}',r'\par\medskip',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{lrrr}',r'\toprule','Curveball minus Linear & '+' & '.join(MODELS.values())+r' \\',r'\midrule']
 for c in NAMES:
  rr=[next(r for r in rs if r['model']==m and r['concept']==c) for m in MODELS]
  lines.append(NAMES[c]+' & '+' & '.join(f"{float(r['difference_pp']):+.3f} [{float(r['difference_ci_low']):+.3f}, {float(r['difference_ci_high']):+.3f}]" for r in rr)+r' \\')
 lines += [r'\bottomrule',r'\end{tabular}}',r'\caption{'+caption.replace('pointwise95', 'pointwise 95').replace('Lambda0', 'Lambda 0')+'}',r'\label{tab:main_results}',r'\end{table*}']
 (PAPER/'tables/main_results.tex').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':format_main()
