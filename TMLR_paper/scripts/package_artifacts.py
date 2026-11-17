"""Build a self-contained Overleaf preview and transparent completion inventory."""
import subprocess, zipfile, platform
from artifact_common import *
from format_tables import format_main

def long_table(path, headers, body, caption, label):
    lines=[r'\begingroup\small\setlength{\tabcolsep}{4pt}',r'\begin{longtable}{'+'l'*len(headers)+'}',r'\caption{'+caption+r'}\label{'+label+r'}\\',r'\toprule',
           ' & '.join(headers)+r' \\',r'\midrule',r'\endfirsthead',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',r'\endhead']
    lines += [' & '.join(map(str,r))+r' \\' for r in body]
    lines += [r'\bottomrule',r'\end{longtable}',r'\endgroup']
    path.write_text('\n'.join(lines)+'\n')

def main():
    format_main()
    allrows=rows(PAPER/'data/training_validation_test.csv')
    short={'qwen3':'Qwen3','llama3b':'Llama3B','llama2':'Llama2'}
    for m in MODELS:
        ix={(r['concept'],r['partition'],r['method']):r for r in allrows if r['model']==m}
        # Short grouped column headings keep the complete split tables legible.
        lines=[r'\begin{table*}[p]\centering\small\setlength{\tabcolsep}{3pt}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{llrrrrrrrrrrrr}',r'\toprule',
               r' & & \multicolumn{4}{c}{Training} & \multicolumn{4}{c}{Validation} & \multicolumn{4}{c}{Official Test} \\',
               r'Behavior & $\lambda^*$ & '+(' & '.join([r'Linear & $0$ & $\lambda^*$ & $1$']*3))+r' \\',r'\midrule']
        for c in NAMES:
            w=selected(m)[c]
            vals=[NAMES[c],f'{w:g}']+[f"{float(ix[c,p,k]['positive_delta_pp']):+.3f}" for p in ['fit','validation','benchmark'] for k in ['Linear','lambda_0.0',f'lambda_{w}','lambda_1.0']]
            lines.append(' & '.join(vals)+r' \\')
        lines += [r'\bottomrule\end{tabular}}',r'\caption{'+MODELS[m]+': positive target probability changes (pp). Curveball columns identify kernel weight, not strength. The selected weight is frozen across splits; training is in-sample and Official Test is exploratory.}',r'\label{tab:splits_'+m+'}',r'\end{table*}']
        (PAPER/f'tables/appendix_splits_{m}.tex').write_text('\n'.join(lines)+'\n')
        for p in ['fit','validation']:
            tex_table(PAPER/f'tables/appendix_grid_{m}_{p}.tex',['Behavior','Linear']+[f'{w:g}' for w in WEIGHTS],
                      [[NAMES[c]]+[f"{float(ix[c,p,k]['positive_delta_pp']):+.3f}" for k in ['Linear']+[f'lambda_{w}' for w in WEIGHTS]] for c in NAMES],
                      MODELS[m]+': '+p+' positive target probability changes (pp). Numerical column labels are quadratic weights. Lambda 1 is a reference excluded from selection. Native positive strength is fixed across weights.','tab:grid_'+m+'_'+p)
    fmt=lambda x:'--' if x=='' else f'{float(x):.3f}'
    for part,filename in [('validation','magnitude_correlation'),('fit','fit_magnitude_correlation')]:
        p=PAPER/f'data/geometric_diagnostics/{filename}.csv'
        if p.exists():
            rr=rows(p)
            long_table(PAPER/f'tables/{filename}_full.tex',['Model','Behavior',r'$\lambda$','Q / G',r'$\rho$','95\% CI','Holm p'],
                       [[short[r['model']],r['behavior'],r['weight'],r['questions']+' / '+r['groups'],fmt(r['group_spearman']),'['+fmt(r['ci_low'])+', '+fmt(r['ci_high'])+']',fmt(r.get('holm_adjusted_p',''))] for r in rr],
                       part.capitalize()+' genuine paired-target magnitude correlations. In-sample fit and out-of-fit validation families are corrected separately. Undefined near-constant magnitudes are shown as --.','tab:'+filename+'_full')
    rr=rows(PAPER/'data/geometric_diagnostics/cross_prompt_consistency.csv')
    long_table(PAPER/'tables/cross_prompt_consistency_full.tex',['Model','Behavior','Partition','Q / G','Cosine','95\% CI'],
               [[short[r['model']],r['behavior'],r['partition'],r['questions']+' / '+r['groups'],fmt(r['mean_cosine']),'['+fmt(r['ci_low'])+', '+fmt(r['ci_high'])+']'] for r in rr],
               'Full cross-prompt directional consistency. Source-group bootstrap; fit is in-sample.','tab:cross_prompt_full')
    val=[r for r in rr if r['partition']=='validation']
    tex_table(PAPER/'tables/cross_prompt_consistency_main.tex',['Behavior']+[short[m]+' cosine [95% CI]' for m in MODELS],
              [[NAMES[c]]+[next(fmt(r['mean_cosine'])+' ['+fmt(r['ci_low'])+', '+fmt(r['ci_high'])+']' for r in val if r['model']==m and r['concept']==c) for m in MODELS] for c in NAMES],
              'Validation cross-prompt cosine of genuine paired contrast vectors from distinct source groups, with 1,000 group-bootstrap intervals. Counts appear in Table 3; the full fitting/validation comparison is supplied separately. No tone comparison is inferred.','tab:cross_prompt_main')
    status=load(PAPER/'data/artifact_status.json')
    completion=[]
    for i in [1,2,3,4]: completion.append(dict(kind='table',number=i,status='complete',file={1:'tables/main_results.tex',2:'tables/magnitude_correlation_selected.tex',3:'tables/dataset_statistics.tex',4:'tables/cross_prompt_consistency_main.tex'}[i]))
    for i in range(1,22):
        s='removed: open-ended modality absent' if i in [8,9] else 'not reproduced: provenance / cost assessment only' if i==2 else 'complete'
        if i in [3,10,11,21]:
            model={10:'qwen3',11:'llama3b',21:'llama2'}.get(i)
            complete=status['response_complete'] if model is None else any(r['model']==model and r['status']=='complete' for r in status['response_status'])
            if i in status.get('draft_figures',[]):s='incomplete; draft excluded from the paper bundle'
            elif i in status['generated_figures']:s='complete'
            elif not complete:s='pending GPU inference'
        completion.append(dict(kind='figure',number=i,status=s,file=f'figures/figure_{i}.pdf' if i in status['generated_figures'] else f'figures/drafts/figure_{i}.pdf' if i in status.get('draft_figures',[]) else ''))
    write_csv(PAPER/'data/artifact_completion.csv',completion)
    inventory=PAPER/'reports/paper_artifact_inventory.md'
    if inventory.exists():
        original=inventory.read_text().split('\n<!-- CURRENT_COMPLETION -->')[0]
        current='\n<!-- CURRENT_COMPLETION -->\n\n## Current generated-artifact status\n\n'+table(['Artifact','Status','File'],[[r['kind']+' '+str(r['number']),r['status'],r['file']] for r in completion])+'\n'
        inventory.write_text(original+current)
    tex=[r'\documentclass[10pt]{article}',r'\usepackage[margin=1in]{geometry}',r'\usepackage{booktabs,graphicx,longtable,hyperref}',r'\begin{document}',
         r'\section*{Curveball TMLR empirical revision: artifact preview}',
         'This is an artifact compilation check, not the revised manuscript. Official Test is an exploratory fixed benchmark. Incomplete response figures are excluded from this compilation and kept separately as drafts. Figure 2 remains unreproduced; Figures 8--9 are removed.',r'\clearpage']
    for i,f in [(1,'main_results'),(2,'magnitude_correlation_selected'),(3,'dataset_statistics'),(4,'cross_prompt_consistency_main')]:
        tex += [r'\setcounter{table}{'+str(i-1)+'}',r'\input{tables/'+f+'}',r'\clearpage']
    tex += [r'\input{tables/figure_includes}',r'\clearpage',r'\appendix',r'\section{Complete split and grid tables}']
    for m in MODELS:
        for suffix in [f'splits_{m}',f'grid_{m}_fit',f'grid_{m}_validation']:
            tex += [r'\input{tables/appendix_'+suffix+'}',r'\clearpage']
    for f in ['magnitude_correlation_full','fit_magnitude_correlation_full','cross_prompt_consistency_full']:
        if (PAPER/f'tables/{f}.tex').exists():tex += [r'\input{tables/'+f+'}',r'\clearpage']
    tex += [r'\end{document}']
    (PAPER/'scripts/overleaf_preview.tex').write_text('\n'.join(tex)+'\n')
    out=PAPER/'reports/overleaf_check';out.mkdir(exist_ok=True)
    for _ in range(2):
        p=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={out}','scripts/overleaf_preview.tex'],cwd=PAPER,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
        (out/'compile_stdout.log').write_text(p.stdout)
        if p.returncode:raise RuntimeError('Overleaf preview failed; see reports/overleaf_check/compile_stdout.log')
    warnings=[line for line in (out/'overleaf_preview.log').read_text(errors='replace').splitlines() if 'Overfull' in line or 'Float too large' in line]
    save(PAPER/'data/publication_checks.json',dict(latex_compile_passed=True,layout_warnings=warnings,python=platform.python_version(),historical_data_modified=False))
    files=list((PAPER/'tables').glob('*.tex'))+list((PAPER/'figures').glob('*.pdf'))+[PAPER/'scripts/overleaf_preview.tex',PAPER/'reports/figure_captions.md',PAPER/'data/artifact_completion.csv',PAPER/'README.md']
    files += [PAPER/'reports/figure_refinement_review.md', PAPER/'data/figure_scientific_checks.json']
    with zipfile.ZipFile(PAPER/'data/overleaf_artifacts.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p.relative_to(PAPER)))
    print(json.dumps(dict(stage='packaged',latex_passed=True,layout_warnings=warnings)),flush=True)

if __name__=='__main__':main()
